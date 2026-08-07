//! Kernel-based Delta table scanning with optimized query execution.
//!
//! This module provides efficient table scanning using Delta Kernel, integrating with
//! DataFusion's query engine. It supports:
//!
//! - **Physical scan execution** ([`DeltaScanExec`]) - Reads Parquet data files and applies
//!   Delta protocol transformations (column mapping, deletion vectors, partition values)
//! - **Metadata-only scans** ([`DeltaScanMetaExec`]) - Answers queries like `COUNT(*)`
//!   using file statistics without reading data files
//! - **Predicate pushdown** - Pushes filters to both kernel file skipping and Parquet readers
//!   for efficient data pruning
//! - **Multi-store support** - Handles files across different object stores in a single query
//!
//! The scan planning process in [`plan`] determines which files to read and how to apply
//! predicates, while execution plans handle the actual data reading and transformation.

use std::{
    collections::{HashSet, VecDeque},
    pin::Pin,
    sync::Arc,
};

use arrow::datatypes::UInt16Type;
use arrow_array::{
    ArrayRef, DictionaryArray, RecordBatch, StringArray, StringViewArray, UInt16Array,
};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType, FieldRef, Schema, SchemaBuilder, SchemaRef};
use chrono::{TimeZone as _, Utc};
use dashmap::DashMap;
use datafusion::{
    catalog::Session,
    common::{
        ColumnStatistics, HashMap, Result, ScalarValue, Statistics, ToDFSchema,
        internal_datafusion_err, plan_err, stats::Precision,
    },
    config::TableParquetOptions,
    datasource::physical_plan::{
        ParquetSource,
        parquet::{
            ParquetAccessPlan, RowGroupAccess,
            metadata::{DFParquetMetadata, ordering_from_parquet_metadata},
        },
    },
    error::DataFusionError,
    execution::{cache::cache_manager::FileMetadataCache, object_store::ObjectStoreUrl},
    physical_expr::LexOrdering,
    physical_plan::{
        ExecutionPlan,
        empty::EmptyExec,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder},
        union::UnionExec,
    },
    prelude::Expr,
};
use datafusion_datasource::{
    PartitionedFile, TableSchema, compute_all_files_statistics,
    file_groups::FileGroup,
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
    source::DataSourceExec,
};
use datafusion_physical_expr_adapter::{
    BatchAdapter, BatchAdapterFactory, DefaultPhysicalExprAdapterFactory,
    PhysicalExprAdapterFactory,
};
use delta_kernel::{
    Engine, Expression, engine::arrow_data::ArrowEngineData, expressions::StructData,
    scan::ScanMetadata, table_features::TableFeature,
};
use futures::{Stream, StreamExt as _, TryStreamExt as _, future::ready};
use itertools::Itertools as _;
use object_store::{ObjectMeta, ObjectStore, path::Path};
use tracing::debug;
use url::Url;

pub use self::exec::DeltaScanExec;
use self::exec_meta::DeltaScanMetaExec;
pub(crate) use self::plan::{KernelScanPlan, ProjectedScanContract, supports_filters_pushdown};
use self::replay::{ScanFileContext, ScanFileStream};
use super::{FileSelection, ResolvedFileSelection};
use crate::{
    DeltaTableError,
    delta_datafusion::{
        DeltaScanConfig,
        engine::{AsObjectStoreUrl as _, to_datafusion_scalar},
        file_id::wrap_file_id_value,
        table_provider::next::DeletionVectorSelection,
    },
    kernel::LogicalFileView,
};

mod exec;
mod exec_meta;
mod plan;
mod replay;

type ScanMetadataStream = Pin<Box<dyn Stream<Item = Result<ScanMetadata, DeltaTableError>> + Send>>;
type PublicFileIdMap = HashMap<String, String>;

struct ReplayedScanFiles {
    files: Vec<ScanFileContext>,
    transforms: HashMap<String, Arc<Expression>>,
    dvs: DashMap<String, Vec<bool>>,
    public_file_ids: PublicFileIdMap,
    metrics: ExecutionPlanMetricsSet,
}

pub(super) async fn execution_plan(
    config: &DeltaScanConfig,
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    stream: ScanMetadataStream,
    engine: Arc<dyn Engine>,
    limit: Option<usize>,
    file_selection: Option<&ResolvedFileSelection>,
    row_ordinal_selections: Option<&std::collections::HashMap<String, Vec<u64>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(selection) = file_selection
        && selection.active_file_ids.is_empty()
    {
        return Ok(Arc::new(EmptyExec::new(
            scan_plan.contract.result_schema.clone(),
        )));
    }

    let replayed = replay_files(engine, &scan_plan, config.clone(), stream, file_selection).await?;

    let file_id_field = scan_plan.contract.file_id_field.clone();
    if scan_plan.is_metadata_only() && !scan_plan.contract.retain_row_index {
        let map_file = |(file_index, f): (usize, &ScanFileContext)| {
            Ok((
                compact_internal_file_id(file_index),
                match &f.stats.num_rows {
                    Precision::Exact(n) => *n,
                    _ => {
                        return plan_err!(
                            "Expected exact row counts in file: {}",
                            super::redact_url_for_error(&f.file_url)
                        );
                    }
                },
            ))
        };

        let maybe_file_rows = replayed
            .files
            .iter()
            .enumerate()
            .map(map_file)
            .try_collect::<_, VecDeque<_>, _>();
        if let Ok(file_rows) = maybe_file_rows {
            let retain_file_id = scan_plan.contract.retain_file_id;
            let ReplayedScanFiles {
                transforms,
                dvs,
                public_file_ids,
                metrics,
                ..
            } = replayed;
            let exec = DeltaScanMetaExec::new(
                Arc::new(scan_plan),
                vec![file_rows],
                Arc::new(transforms),
                Arc::new(dvs),
                Arc::new(public_file_ids),
                retain_file_id.then_some(file_id_field),
                metrics,
            );
            return Ok(Arc::new(exec) as _);
        }
    }

    get_data_scan_plan(session, scan_plan, replayed, limit, row_ordinal_selections).await
}

/// Materialize deletion vector keep masks for every file in the scan that has one.
///
/// Deletion vectors are loaded as a side-effect of consuming [`ScanFileStream`].  We drain the
/// full stream here (discarding file contexts, stats, and partition values) because the DV
/// loading tasks are spawned lazily during stream poll.  A dedicated DV-only stream that skips
/// stats parsing is possible but not yet warranted — this path is not latency-sensitive and the
/// file-list is typically small.
///
/// [`ReceiverStreamBuilder::build`] returns a merged stream that includes a JoinSet checker;
/// `.try_collect().await` below will not complete until every spawned DV-loading task has
/// finished, so no results are lost.
pub(super) async fn replay_deletion_vectors(
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    config: &DeltaScanConfig,
    stream: ScanMetadataStream,
    file_selection: Option<&ResolvedFileSelection>,
) -> Result<Vec<DeletionVectorSelection>> {
    let mut stream = ScanFileStream::new(
        engine,
        &scan_plan.scan,
        config.clone(),
        file_selection.map(|selection| &selection.active_file_ids),
        stream,
    );
    while stream.try_next().await?.is_some() {}

    let dv_stream = stream.dv_stream.build();
    // Only files with `dv_info.has_vector()` spawn tasks, so every item should carry a DV.
    // Guard with a typed error (instead of panic) in case that invariant drifts.
    let dvs: DashMap<_, _> = dv_stream
        .and_then(|(url, dv, num_records)| {
            ready(match dv {
                Some(keep_mask) => normalize_dv_keep_mask_for_api(keep_mask, num_records, &url)
                    .map(|mask| (url.to_string(), mask))
                    .map_err(DeltaTableError::from),
                None => Err(DeltaTableError::generic(
                    "Invariant violation: DV task spawned for file without deletion vector",
                )),
            })
        })
        .try_collect()
        .await?;

    let mut vectors: Vec<_> = dvs
        .into_iter()
        .map(|(filepath, keep_mask)| DeletionVectorSelection {
            filepath,
            keep_mask,
        })
        .collect();
    vectors.sort_unstable_by(|left, right| left.filepath.cmp(&right.filepath));
    Ok(vectors)
}

pub(super) async fn resolve_file_selection(
    selection: &FileSelection,
    scan_plan: &KernelScanPlan,
    stream: ScanMetadataStream,
) -> Result<ResolvedFileSelection> {
    let requested_file_ids =
        resolve_input_file_ids_on_blocking_pool(selection, scan_plan.scan.table_root()).await?;
    if requested_file_ids.is_empty() {
        return Ok(ResolvedFileSelection::new(
            HashSet::new(),
            Vec::new(),
            selection.missing_file_policy,
        ));
    }

    let mut missing_file_ids = requested_file_ids;
    let selected_active_file_ids = collect_selected_active_file_ids(
        scan_plan.scan.table_root(),
        stream,
        &mut missing_file_ids,
    )
    .await?;
    let mut missing_file_ids: Vec<_> = missing_file_ids.into_iter().collect();
    missing_file_ids.sort_unstable();

    let resolved = ResolvedFileSelection::new(
        selected_active_file_ids,
        missing_file_ids,
        selection.missing_file_policy,
    );
    resolved.validate_missing()?;
    Ok(resolved)
}

async fn resolve_input_file_ids_on_blocking_pool(
    selection: &FileSelection,
    table_root: &Url,
) -> Result<HashSet<String>> {
    let selection = selection.clone();
    let table_root = table_root.clone();
    tokio::task::spawn_blocking(move || selection.resolve_input_file_ids(&table_root))
        .await
        .map_err(|err| DataFusionError::External(Box::new(err)))?
        .map_err(DataFusionError::from)
}

async fn collect_selected_active_file_ids(
    table_root: &Url,
    mut stream: ScanMetadataStream,
    missing_file_ids: &mut HashSet<String>,
) -> Result<HashSet<String>> {
    let mut selected_active_file_ids = HashSet::new();

    while let Some(scan_data) = stream.try_next().await? {
        let (data, mut selection_vector) = scan_data.scan_files.into_parts();
        let batch: RecordBatch = ArrowEngineData::try_from_engine_data(data)
            .map_err(DeltaTableError::from)?
            .into();
        // Delta Kernel may return a short selection vector. Missing entries are selected.
        selection_vector.resize(batch.num_rows(), true);

        for (idx, selected) in selection_vector.into_iter().enumerate() {
            if selected {
                let file_url = replay::parse_path(
                    table_root,
                    LogicalFileView::new(batch.clone(), idx).path_raw(),
                )?;
                let file_id = file_url.to_string();
                if missing_file_ids.remove(&file_id) {
                    selected_active_file_ids.insert(file_id);
                    if missing_file_ids.is_empty() {
                        return Ok(selected_active_file_ids);
                    }
                }
            }
        }
    }

    Ok(selected_active_file_ids)
}

async fn replay_files(
    engine: Arc<dyn Engine>,
    scan_plan: &KernelScanPlan,
    scan_config: DeltaScanConfig,
    stream: ScanMetadataStream,
    file_selection: Option<&ResolvedFileSelection>,
) -> Result<ReplayedScanFiles> {
    let mut stream = ScanFileStream::new(
        engine,
        &scan_plan.scan,
        scan_config,
        file_selection.map(|selection| &selection.active_file_ids),
        stream,
    );
    let mut files = Vec::new();
    while let Some(file) = stream.try_next().await? {
        files.extend(file);
    }

    let mut public_file_ids = PublicFileIdMap::default();
    if scan_plan.contract.retain_file_id {
        for (file_index, file) in files.iter().enumerate() {
            public_file_ids.insert(
                compact_internal_file_id(file_index),
                file.file_url.to_string(),
            );
        }
    }

    let transforms: HashMap<_, _> = files
        .iter_mut()
        .enumerate()
        .flat_map(|(file_index, file)| {
            file.transform
                .take()
                .map(|t| (compact_internal_file_id(file_index), t))
        })
        .collect();

    let dv_stream = stream.dv_stream.build();
    let dvs_by_url: HashMap<_, _> = dv_stream
        .try_filter_map(|(url, dv, _)| ready(Ok(dv.map(|dv| (url.to_string(), dv)))))
        .try_collect()
        .await?;
    let dvs = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url)?;

    let metrics = ExecutionPlanMetricsSet::new();
    MetricBuilder::new(&metrics)
        .global_counter("count_files_scanned")
        .add(stream.metrics.num_scanned);

    Ok(ReplayedScanFiles {
        files,
        transforms,
        dvs,
        public_file_ids,
        metrics,
    })
}

/// Normalize a DV keep mask for `deletion_vectors()`.
///
/// Kernel returns a sparse mask (up to the highest deleted row index). For API output we need one
/// full mask per file, to do this we pad trailing entries with `true` up to `numRecords`. If `numRecords`
/// is missing we fail, because we cannot know the correct full length.
///
/// This is API only. Scan execution does per batch normalization in `exec::consume_dv_mask` and
/// `exec_meta::apply_selection_vector`.
fn normalize_dv_keep_mask_for_api(
    mut mask: Vec<bool>,
    num_records: Option<u64>,
    file_url: &Url,
) -> Result<Vec<bool>> {
    let redacted_url = super::redact_url_for_error(file_url);
    let Some(num_records) = num_records else {
        return plan_err!(
            "Missing numRecords for file with deletion vector: {}",
            redacted_url
        );
    };
    let num_records = usize::try_from(num_records).map_err(|_| {
        DataFusionError::Execution(format!(
            "numRecords does not fit usize for file with deletion vector: {redacted_url}"
        ))
    })?;
    if mask.len() > num_records {
        return plan_err!(
            "Deletion vector mask length {} exceeds numRecords {} for file: {}",
            mask.len(),
            num_records,
            redacted_url
        );
    }
    mask.resize(num_records, true);
    Ok(mask)
}

async fn get_data_scan_plan(
    session: &dyn Session,
    scan_plan: KernelScanPlan,
    replayed: ReplayedScanFiles,
    limit: Option<usize>,
    row_ordinal_selections: Option<&std::collections::HashMap<String, Vec<u64>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let ReplayedScanFiles {
        files,
        transforms,
        dvs,
        public_file_ids,
        metrics,
    } = replayed;
    let mut partition_stats = HashMap::new();

    // Convert files into DataFusion `PartitionedFile`s grouped by object store.
    // Create one `DataSourceExec` plan for each store.
    // Add a compact scan file id as a partition value for file correlation.
    // The exec maps that id back to the public file path only when the file column is projected.
    let to_partitioned_file = |(file_index, f): (usize, ScanFileContext)| {
        if let Some(part_stata) = &f.partitions {
            update_partition_stats(part_stata, &f.stats, &mut partition_stats)?;
        }
        // We create a PartitionedFile from the ObjectMeta to avoid any surprises in path encoding
        // that may arise from using the 'new' method directly. i.e. the 'new' method encodes paths
        // segments again, which may lead to double-encoding in some cases.
        let mut partitioned_file: PartitionedFile = ObjectMeta {
            location: Path::from_url_path(f.file_url.path())?,
            size: f.size,
            last_modified: Utc.timestamp_nanos(0),
            e_tag: None,
            version: None,
        }
        .into();
        let file_value = wrap_file_id_value(compact_internal_file_id(file_index));
        // NOTE: `PartitionedFile::with_statistics` appends exact stats for partition columns based
        // on `partition_values`, so partition values must be set first.
        partitioned_file.partition_values = vec![file_value.clone()];
        partitioned_file = partitioned_file.with_statistics(Arc::new(f.stats));
        Ok::<_, DataFusionError>((
            f.file_url.as_object_store_url(),
            (partitioned_file, None::<Vec<bool>>),
        ))
    };

    // Group the files by their object store url. Since datafusion assumes that all files in a
    // DataSourceExec are stored in the same object store, we need to create one plan per store
    let partitioned_files = files
        .into_iter()
        .enumerate()
        .map(to_partitioned_file)
        .try_collect::<_, Vec<_>, _>()?;

    let files_by_store = partitioned_files.into_iter().into_group_map();

    // TODO(roeap); not sure exactly how row tracking is implemented in kernel right now
    // so leaving predicate as None for now until we are sure this is safe to do.
    //
    // Deletion vectors are applied as per-file keep-masks indexed by ROW POSITION
    // (`exec::consume_dv_mask`). Even when a read scan opts in to pushdown under DV
    // (pushdown_with_deletion_vectors), a file that actually carries a keep-mask
    // must NOT get the predicate pushed: pushdown filters rows before the mask is
    // applied, shifting positions so the mask hides the wrong rows (a deleted row
    // reappears). Drop the predicate when this scan carries any DV keep-mask; the
    // common DV-free scans (e.g. the freshly compacted hot tail) still push down.
    let table_config = scan_plan.table_configuration();
    let predicate =
        if table_config.is_feature_enabled(&TableFeature::RowTracking) || !dvs.is_empty() {
            None
        } else {
            scan_plan.parquet_predicate.as_ref()
        };
    let file_id_field = scan_plan.contract.file_id_field.clone();
    let pq_plan = get_read_plan(
        session,
        files_by_store,
        &scan_plan.parquet_read_schema,
        &scan_plan.parquet_predicate_schema,
        limit,
        &file_id_field,
        predicate,
        row_ordinal_selections,
    )
    .await?;

    let transforms = Arc::new(transforms);
    let dvs = Arc::new(dvs);
    let public_file_ids = Arc::new(public_file_ids);
    let exec = DeltaScanExec::new(
        Arc::new(scan_plan),
        pq_plan,
        Arc::clone(&transforms),
        Arc::clone(&dvs),
        Arc::clone(&public_file_ids),
        partition_stats,
        metrics,
    );

    Ok(Arc::new(exec))
}

fn update_partition_stats(
    data: &StructData,
    stats: &Statistics,
    part_stats: &mut HashMap<String, ColumnStatistics>,
) -> Result<()> {
    for (field, stat) in data.fields().iter().zip(data.values().iter()) {
        let (null_count, value) = if stat.is_null() {
            (stats.num_rows, Precision::Absent)
        } else {
            (
                Precision::Exact(0),
                Precision::Exact(to_datafusion_scalar(stat)?),
            )
        };
        if let Some(part_stat) = part_stats.get_mut(field.name()) {
            part_stat.null_count = part_stat.null_count.add(&null_count);
            part_stat.min_value = part_stat.min_value.min(&value);
            part_stat.max_value = part_stat.max_value.max(&value);
        } else {
            part_stats.insert(
                field.name().clone(),
                ColumnStatistics {
                    null_count,
                    min_value: value.clone(),
                    max_value: value,
                    distinct_count: Precision::Absent,
                    sum_value: Precision::Absent,
                    byte_size: Precision::Absent,
                },
            );
        }
    }

    Ok(())
}

type FilesByStore = (ObjectStoreUrl, Vec<(PartitionedFile, Option<Vec<bool>>)>);

fn compact_internal_file_id(file_index: usize) -> String {
    file_index.to_string()
}

fn remap_deletion_vectors_to_internal_file_ids(
    files: &[ScanFileContext],
    mut dvs_by_url: HashMap<String, Vec<bool>>,
) -> Result<DashMap<String, Vec<bool>>> {
    let dvs = DashMap::new();
    for (file_index, file) in files.iter().enumerate() {
        if dvs_by_url.is_empty() {
            break;
        }
        if let Some(dv) = dvs_by_url.remove(file.file_url.as_str()) {
            dvs.insert(compact_internal_file_id(file_index), dv);
        }
    }
    if let Some(file_url) = dvs_by_url.keys().next() {
        let redacted_url = Url::parse(file_url)
            .map(|url| super::redact_url_for_error(&url))
            .unwrap_or_else(|_| file_url.clone());
        return Err(internal_datafusion_err!(
            "missing internal file id mapping for file with deletion vector: {redacted_url}"
        ));
    }
    Ok(dvs)
}

fn public_file_id<'a>(
    public_file_ids: &'a PublicFileIdMap,
    internal_file_id: &str,
) -> Result<&'a str> {
    public_file_ids
        .get(internal_file_id)
        .map(String::as_str)
        .ok_or_else(|| {
            internal_datafusion_err!(
                "missing public file id mapping for internal file id '{internal_file_id}'"
            )
        })
}

fn file_id_array_for_value(
    file_id_field: &FieldRef,
    file_id: &str,
    row_count: usize,
) -> Result<ArrayRef> {
    let keys = UInt16Array::from(vec![0u16; row_count]);
    let values: ArrayRef = match file_id_field.data_type() {
        DataType::Dictionary(_, value_type) if value_type.as_ref() == &DataType::Utf8View => {
            if row_count == 0 {
                Arc::new(StringViewArray::from_iter_values(std::iter::empty::<&str>()))
            } else {
                Arc::new(StringViewArray::from_iter_values([file_id]))
            }
        }
        _ => {
            if row_count == 0 {
                Arc::new(StringArray::from(Vec::<Option<&str>>::new()))
            } else {
                Arc::new(StringArray::from(vec![Some(file_id)]))
            }
        }
    };

    let file_id_array: DictionaryArray<UInt16Type> = DictionaryArray::try_new(keys, values)?;
    Ok(Arc::new(file_id_array))
}

/// Maximum number of distinct values representable by DataFusion's default partition dictionary
/// encoding (`Dictionary<UInt16, _>`).
const MAX_PARTITION_DICT_CARDINALITY: usize = (u16::MAX as usize) + 1;

fn partitioned_files_to_file_groups(
    files: impl IntoIterator<Item = PartitionedFile>,
) -> Vec<FileGroup> {
    partitioned_files_to_file_groups_with_limit(files, MAX_PARTITION_DICT_CARDINALITY)
}

fn partitioned_files_to_file_groups_with_limit(
    files: impl IntoIterator<Item = PartitionedFile>,
    max_files_per_group: usize,
) -> Vec<FileGroup> {
    let file_groups = files
        .into_iter()
        // Each `PartitionedFile` is assigned to exactly one file group. DeltaScanStream stores
        // row ordinal counters per execution partition. Whole file ownership is required for
        // scan row ordinals.
        // Partition values are dictionary encoded using a UInt16 key (DataFusion's default
        // `wrap_partition_type_in_dict`). Keep file groups small enough that the file-id partition
        // dictionary doesn't exceed the key space (one distinct value per file).
        .chunks(max_files_per_group)
        .into_iter()
        .map(|chunk| chunk.collect::<FileGroup>())
        .collect_vec();

    #[cfg(debug_assertions)]
    {
        let mut owner_by_path = HashMap::new();
        for (partition, group) in file_groups.iter().enumerate() {
            for file in group.iter() {
                let path = file.object_meta.location.to_string();
                if let Some(previous_partition) = owner_by_path.insert(path.clone(), partition) {
                    debug_assert_eq!(
                        previous_partition, partition,
                        "file {path} was assigned to multiple scan partitions; row indexes require whole file ownership"
                    );
                }
            }
        }
    }

    file_groups
}

async fn get_read_plan(
    state: &dyn Session,
    files_by_store: impl IntoIterator<Item = FilesByStore>,
    // Schema of physical file columns to read from Parquet (no Delta partitions, no file-id).
    //
    // This is also the schema used for Parquet pruning/pushdown. It may include view types
    // (e.g. Utf8View/BinaryView) depending on `DeltaScanConfig`.
    parquet_read_schema: &SchemaRef,
    // Predicate binding schema used to bind Parquet predicates, including the synthetic file id
    // column when the provider exposes it.
    parquet_predicate_schema: &SchemaRef,
    limit: Option<usize>,
    file_id_field: &FieldRef,
    predicate: Option<&Expr>,
    // Optional per-file row-ordinal selections keyed by table-relative parquet path. Matching
    // files get a `ParquetAccessPlan` attached so the parquet opener skips non-selected row
    // groups/rows; all other files (and any fallback case) scan fully.
    row_ordinal_selections: Option<&std::collections::HashMap<String, Vec<u64>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut plans = Vec::new();

    let pq_options = TableParquetOptions {
        global: state.config().options().execution.parquet.clone(),
        ..Default::default()
    };

    let mut full_read_schema = SchemaBuilder::from(parquet_read_schema.as_ref().clone());
    full_read_schema.push(file_id_field.as_ref().clone().with_nullable(true));
    let full_read_schema = Arc::new(full_read_schema.finish());
    let parquet_predicate_df_schema = parquet_predicate_schema.clone().to_dfschema()?;
    let adapter_factory = Arc::new(DefaultPhysicalExprAdapterFactory {});

    for (store_url, mut files) in files_by_store.into_iter() {
        let object_store = state.runtime_env().object_store(&store_url)?;
        let metadata_cache: Arc<dyn FileMetadataCache> = Arc::new(
            crate::delta_datafusion::parquet_metrics::InstrumentedFileMetadataCache::new(
                state.runtime_env().cache_manager.get_file_metadata_cache(),
            ),
        );
        let file_count = files.len();
        let planned_bytes: u64 = files.iter().map(|(file, _)| file.object_meta.size).sum();
        crate::delta_datafusion::parquet_metrics::record_scan(file_count, planned_bytes);
        let file_ids = files
            .iter()
            .take(8)
            .map(|(file, _)| file.object_meta.location.as_ref())
            .collect::<Vec<_>>()
            .join(",");
        let _parquet_span = parquet_plan_span(file_count, planned_bytes, &file_ids);
        let reader_factory = Arc::new(
            crate::delta_datafusion::parquet_metrics::InstrumentedParquetFileReaderFactory::new(
                object_store.clone(),
                metadata_cache.clone(),
            ),
        );

        // NOTE: In the "next" provider, DataFusion's Parquet scan partition fields are file-id
        // only. Delta partition columns/values are injected via kernel transforms and handled
        // above Parquet, so they are not part of the Parquet partition schema here.
        let table_schema =
            TableSchema::new(parquet_read_schema.clone(), vec![file_id_field.clone()]);
        let full_table_schema = table_schema.table_schema().clone();
        let mut file_source = ParquetSource::new(table_schema)
            .with_table_parquet_options(pq_options.clone())
            .with_parquet_file_reader_factory(reader_factory);

        // TODO(roeap); we might be able to also push selection vectors into the read plan
        // by creating parquet access plans. However we need to make sure this does not
        // interfere with other delta features like row ids.
        let has_selection_vectors = files.iter().any(|(_, sv)| sv.is_some());
        if !has_selection_vectors && let Some(pred) = predicate {
            match state.create_physical_expr(pred.clone(), &parquet_predicate_df_schema) {
                Ok(physical) => match adapter_factory
                    .create(parquet_predicate_schema.clone(), full_read_schema.clone())
                {
                    Ok(adapter) => match adapter.rewrite(physical) {
                        Ok(rewritten) => {
                            file_source = file_source
                                .with_predicate(rewritten)
                                .with_pushdown_filters(true);
                        }
                        Err(err) => {
                            debug!(
                                predicate = ?pred,
                                schema = ?parquet_predicate_schema,
                                error = %err,
                                "Skipping parquet predicate pushdown because predicate adaptation to the read schema failed"
                            );
                        }
                    },
                    Err(err) => {
                        debug!(
                            predicate = ?pred,
                            schema = ?parquet_predicate_schema,
                            error = %err,
                            "Skipping parquet predicate pushdown because predicate adapter creation failed"
                        );
                    }
                },
                Err(err) => {
                    debug!(
                        predicate = ?pred,
                        schema = ?parquet_predicate_schema,
                        error = %err,
                        "Skipping parquet predicate pushdown because predicate binding failed"
                    );
                }
            }
        }

        // Derive a scan-wide output ordering from the parquet footers. After TimeFusion's
        // "Option A" fix only files actually written in sort order declare `sorting_columns`,
        // so a footer-derived ordering never over-claims. Loading the footers here is cache-
        // backed (same metadata cache the reader uses), so it front-loads reads the scan does
        // anyway rather than adding net IO.
        let derived_ordering = derive_common_ordering(
            object_store.clone(),
            metadata_cache.clone(),
            &files,
            parquet_read_schema.clone(),
        )
        .await;

        // A declared ordering only survives DataFusion's stats-based per-group validation
        // (`FileScanConfig::validated_output_ordering`) if every file carries min/max stats
        // for every sort column, and Delta add-file stats only cover predicate columns. Merge
        // the sort-column stats `derive_common_ordering` extracted from the footers it
        // already fetched (no plan-time IO here), then declare only the longest sort prefix
        // every file has stats for — a prefix claim is still sound (each file is sorted by
        // the full key, hence by any prefix) and is what TopK / bounded dedup need.
        // `regroup` is false when not even the lead column is stats-backed: declare the full
        // ordering over the untouched grouping and let validation decide, exactly as before.
        // Isolate, don't surrender. A file with no (or a different) footer ordering used to
        // void the claim for the WHOLE scan — one unsorted file among thousands cost every
        // other file its ordering, which is what turns off the streaming top-N pushdown and
        // forces merge-on-read dedup into its unbounded `full-set` seen-set. Prod ran with
        // 55% of active files unsorted, so that "one bad file" case was the common case.
        // Now the conforming majority keeps the claim and the rest scan separately, unordered.
        let (mut files, mut unordered_files, output_ordering, regroup) = match derived_ordering {
            Some((ordering, footer_stats, conforms)) => {
                apply_footer_sort_stats(&mut files, footer_stats, parquet_read_schema);
                let (conforming, rest) = split_by_declared_ordering(files, &conforms);
                // Prefix length is a property of the files that will actually carry the
                // claim, so it must be measured on the conforming set alone.
                let (ordering, regroup) = match stats_backed_prefix_len(&conforming, &ordering) {
                    0 => (Some(ordering), false),
                    len => (LexOrdering::new(ordering.iter().take(len).cloned()), true),
                };
                (conforming, rest, ordering, regroup)
            }
            None => (files, Vec::new(), None, false),
        };
        // Nothing conformed (or nothing to claim): one plain scan over everything, exactly
        // as before.
        if output_ordering.is_none() && !unordered_files.is_empty() {
            files.append(&mut unordered_files);
        }

        if let Some(selections) = row_ordinal_selections
            && !selections.is_empty()
        {
            attach_row_ordinal_access_plans(
                object_store.clone(),
                metadata_cache.clone(),
                &mut files,
                selections,
            )
            .await;
        }

        let file_groups = partitioned_files_to_file_groups(files.into_iter().map(|file| file.0));
        // Snapshot-order groups almost never validate under merge-on-read (an UPDATE appends
        // rows with their original timestamps into a new file, so files overlap in the lead
        // key). Repack so the declared ordering survives; without a declared ordering keep
        // the grouping exactly as-is.
        let file_groups = match &output_ordering {
            Some(ordering) if regroup => {
                regroup_for_declared_ordering(file_groups, ordering, &full_table_schema)
            }
            _ => file_groups,
        };
        let (file_groups, statistics) =
            compute_all_files_statistics(file_groups, full_table_schema.clone(), true, false)?;

        let mut config =
            FileScanConfigBuilder::new(store_url.clone(), Arc::new(file_source.clone()))
                .with_file_groups(file_groups)
                .with_statistics(statistics)
                .with_limit(limit)
                .with_expr_adapter(Some(adapter_factory.clone() as _));
        if let Some(ordering) = output_ordering {
            // Auto-enables `preserve_order`: DataFusion keeps each file group's order and
            // merges groups with a SortPreservingMergeExec instead of concatenating.
            config = config.with_output_ordering(vec![ordering]);
        }

        plans.push(DataSourceExec::from_data_source(config.build()) as Arc<dyn ExecutionPlan>);

        // The isolated non-conforming files: same source and predicate, no ordering claim.
        // Unioned as a sibling so the ordered leg above keeps its pushdowns; DataFusion (and
        // TimeFusion's `ordered_union_for_topk`) can then sort just this small leg when a
        // query wants a global order, instead of the whole partition losing the claim.
        if !unordered_files.is_empty() {
            let groups =
                partitioned_files_to_file_groups(unordered_files.into_iter().map(|file| file.0));
            let (groups, statistics) =
                compute_all_files_statistics(groups, full_table_schema, true, false)?;
            let config = FileScanConfigBuilder::new(store_url, Arc::new(file_source))
                .with_file_groups(groups)
                .with_statistics(statistics)
                .with_limit(limit)
                .with_expr_adapter(Some(adapter_factory.clone() as _))
                .build();
            plans.push(DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>);
        }
    }

    Ok(match plans.len() {
        0 => Arc::new(EmptyExec::new(full_read_schema.clone())),
        1 => plans.remove(0),
        _ => UnionExec::try_new(plans)?,
    })
}

/// Per-file sort-column footer stats: `(column index, (min, max, null_count))` for every
/// column of the file's declared footer ordering; `None` when some stat is unavailable.
type FooterSortStats = Option<Vec<(usize, (ScalarValue, ScalarValue, usize))>>;

/// Partition `files` by whether each one declares the scan's common ordering. Conforming
/// files carry the claim; the rest are scanned separately with no claim at all.
fn split_by_declared_ordering(
    files: Vec<(PartitionedFile, Option<Vec<bool>>)>,
    conforms: &[bool],
) -> (
    Vec<(PartitionedFile, Option<Vec<bool>>)>,
    Vec<(PartitionedFile, Option<Vec<bool>>)>,
) {
    let (conforming, rest): (Vec<_>, Vec<_>) = files
        .into_iter()
        .enumerate()
        .partition(|(idx, _)| conforms.get(*idx).copied().unwrap_or(false));
    (
        conforming.into_iter().map(|(_, file)| file).collect(),
        rest.into_iter().map(|(_, file)| file).collect(),
    )
}

/// Returns the [`LexOrdering`] the **largest set** of files agree on, plus which files
/// declare it. Previously this was all-or-nothing — every file had to declare the same
/// non-empty parquet `sorting_columns` footer or the scan lost its ordering entirely — which
/// meant one unsorted file (a compacted/Z-ordered output, or a flush whose sort was skipped)
/// cost every other file its claim. The caller now isolates the non-conforming files instead.
///
/// Alongside the ordering, returns each file's sort-column min/max/null-count extracted from
/// the **same** footers (indexed like `files`) — the ordering only survives DataFusion's
/// stats-based validation when per-file stats back it, and pulling them out here keeps all
/// footer reads in this one bounded, cache-backed pass (no further plan-time IO).
async fn derive_common_ordering(
    store: Arc<dyn ObjectStore>,
    cache: Arc<dyn FileMetadataCache>,
    files: &[(PartitionedFile, Option<Vec<bool>>)],
    read_schema: SchemaRef,
) -> Option<(LexOrdering, Vec<FooterSortStats>, Vec<bool>)> {
    use datafusion::physical_expr::expressions::Column;
    // Fetch footers concurrently (bounded). On a cold metadata cache a scan can span hundreds
    // of files; a serial loop would add that many sequential footer reads to *planning*. The
    // cache is shared with the reader, so these reads also warm execution. The per-file futures
    // own (Arc-clone) everything they touch so the parent scan future stays lifetime-general.
    const FOOTER_FETCH_CONCURRENCY: usize = 16;
    // Detach from `files` (owned ObjectMeta) before building futures so the per-file closure
    // takes an owned value — a closure over a borrowed scan item is not lifetime-general and
    // makes the whole scan future fail HRTB inference.
    let object_metas: Vec<ObjectMeta> = files.iter().map(|(f, _)| f.object_meta.clone()).collect();
    let file_count = object_metas.len();
    let fetches = object_metas
        .into_iter()
        .enumerate()
        .map(|(idx, object_meta)| {
            let (store, cache, read_schema) = (store.clone(), cache.clone(), read_schema.clone());
            async move {
                let meta = DFParquetMetadata::new(store.as_ref(), &object_meta)
                    .with_file_metadata_cache(Some(cache))
                    .fetch_metadata()
                    .await
                    .ok();
                let ordering = meta.as_ref().and_then(|meta| {
                    ordering_from_parquet_metadata(meta, &read_schema)
                        .ok()
                        .flatten()
                });
                let stats: FooterSortStats =
                    ordering
                        .as_ref()
                        .zip(meta.as_ref())
                        .and_then(|(ordering, meta)| {
                            ordering
                                .iter()
                                .map(|expr| {
                                    let column = expr.expr.downcast_ref::<Column>()?.index();
                                    sort_column_footer_stats(meta, &read_schema, column)
                                        .map(|stats| (column, stats))
                                })
                                .collect()
                        });
                (idx, ordering, stats)
            }
        });
    let results: Vec<(usize, Option<LexOrdering>, FooterSortStats)> =
        futures::stream::iter(fetches)
            .buffer_unordered(FOOTER_FETCH_CONCURRENCY)
            .collect()
            .await;

    // Pick the ordering the most files agree on, rather than bailing at the first
    // disagreement. `conforms[i]` then says whether file i can be scanned under that
    // claim; the caller isolates the rest into their own unordered scan instead of
    // losing the claim for everything (see `split_by_declared_ordering`).
    let mut per_file: Vec<Option<LexOrdering>> = vec![None; file_count];
    let mut footer_stats: Vec<FooterSortStats> = vec![None; file_count];
    for (idx, ordering, stats) in results {
        per_file[idx] = ordering;
        footer_stats[idx] = stats;
    }
    let mut tally: Vec<(LexOrdering, usize)> = Vec::new();
    for ordering in per_file.iter().flatten() {
        match tally
            .iter_mut()
            .find(|(candidate, _)| candidate == ordering)
        {
            Some((_, count)) => *count += 1,
            None => tally.push((ordering.clone(), 1)),
        }
    }
    let common = tally
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(ordering, _)| ordering)?;
    let conforms: Vec<bool> = per_file
        .iter()
        .map(|ordering| ordering.as_ref().is_some_and(|o| *o == common))
        .collect();
    Some((common, footer_stats, conforms))
}

/// Upper bound on file groups produced when repacking for a declared ordering. Heavily
/// overlapping files (deep merge-on-read update chains) would otherwise degenerate into one
/// single-file group per file, and a SortPreservingMergeExec over that many concurrent
/// parquet streams costs more than the ordering claim is worth.
const MAX_ORDERED_FILE_GROUPS: usize = 512;

/// Repack `file_groups` so a declared `ordering` survives DataFusion's per-group validation
/// (`FileScanConfig::validated_output_ordering`): a multi-file group only validates when its
/// files are non-overlapping in the sort key min/max stats AND listed in sort order. The
/// default snapshot-order grouping violates this as soon as files overlap in the lead key,
/// silently dropping the claim — and with it TopK streaming and bounded dedup upstream.
///
/// Delegates to DataFusion's own stats-based first-fit packer with the previous group count
/// as the target: disjoint runs stay packed together (parallelism and the dictionary-key
/// cardinality bound are preserved), while mutually overlapping files spill into their own
/// groups, which validate trivially. Falls back to the original grouping when stats are
/// unusable or the result would be degenerate — the declared ordering then dies in
/// validation exactly as before this repacking existed.
fn regroup_for_declared_ordering(
    file_groups: Vec<FileGroup>,
    ordering: &LexOrdering,
    table_schema: &SchemaRef,
) -> Vec<FileGroup> {
    let target_partitions = file_groups.len().max(1);
    match FileScanConfig::split_groups_by_statistics_with_target_partitions(
        table_schema,
        &file_groups,
        ordering,
        target_partitions,
    ) {
        Ok(groups)
            if groups.len() <= MAX_ORDERED_FILE_GROUPS
                && groups
                    .iter()
                    .all(|group| group.len() <= MAX_PARTITION_DICT_CARDINALITY) =>
        {
            groups
        }
        Ok(groups) => {
            debug!(
                groups = groups.len(),
                "keeping snapshot-order file groups; ordered repacking was degenerate"
            );
            file_groups
        }
        Err(err) => {
            debug!(
                error = %err,
                "keeping snapshot-order file groups; ordering claim will not survive validation"
            );
            file_groups
        }
    }
}

/// Merge footer-derived sort-column stats into each file's DataFusion statistics, filling
/// gaps only — Delta add-file stats, when present, stay authoritative. Purely in-memory: the
/// footers were already read by [`derive_common_ordering`], this does no IO.
///
/// DataFusion validates a declared output ordering against per-file min/max statistics, but
/// Delta stats are only materialized for predicate columns, so sort columns (e.g. a tiebreak
/// id) are typically `Absent` and every multi-file group would fail validation. Footer
/// min/max may be truncated bounds (strings), hence `Inexact` — still a valid bound for the
/// non-overlap check.
fn apply_footer_sort_stats(
    files: &mut [(PartitionedFile, Option<Vec<bool>>)],
    footer_stats: Vec<FooterSortStats>,
    read_schema: &SchemaRef,
) {
    for ((file, _), per_column) in files.iter_mut().zip(footer_stats) {
        let Some(per_column) = per_column else {
            continue;
        };
        let stats = file
            .statistics
            .get_or_insert_with(|| Arc::new(Statistics::new_unknown(read_schema)));
        let stats = Arc::make_mut(stats);
        for (column, (min, max, null_count)) in per_column {
            let Some(column_stats) = stats.column_statistics.get_mut(column) else {
                continue;
            };
            if column_stats.min_value.get_value().is_none() {
                column_stats.min_value = Precision::Inexact(min);
            }
            if column_stats.max_value.get_value().is_none() {
                column_stats.max_value = Precision::Inexact(max);
            }
            if !matches!(column_stats.null_count, Precision::Exact(_)) {
                column_stats.null_count = Precision::Exact(null_count);
            }
        }
    }
}

/// Longest prefix of `ordering` for which **every** file carries min/max stats on a plain
/// column — the largest claim DataFusion's stats-based validation can actually confirm.
/// Zero means not even the lead column is stats-backed everywhere.
fn stats_backed_prefix_len(
    files: &[(PartitionedFile, Option<Vec<bool>>)],
    ordering: &LexOrdering,
) -> usize {
    use datafusion::physical_expr::expressions::Column;
    ordering
        .iter()
        .take_while(|expr| {
            expr.expr.downcast_ref::<Column>().is_some_and(|column| {
                files.iter().all(|(file, _)| {
                    file.statistics.as_ref().is_some_and(|stats| {
                        stats
                            .column_statistics
                            .get(column.index())
                            .is_some_and(|cs| {
                                cs.min_value.get_value().is_some()
                                    && cs.max_value.get_value().is_some()
                            })
                    })
                })
            })
        })
        .count()
}

/// (min, max, null_count) of one column across all row groups, from footer statistics.
/// `None` when any row group lacks the stat (a partial bound is not a bound) or the file has
/// no row groups.
fn sort_column_footer_stats(
    meta: &parquet::file::metadata::ParquetMetaData,
    read_schema: &SchemaRef,
    column_index: usize,
) -> Option<(ScalarValue, ScalarValue, usize)> {
    use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
    let converter = StatisticsConverter::try_new(
        read_schema.field(column_index).name(),
        read_schema,
        meta.file_metadata().schema_descr(),
    )
    .ok()?;
    let row_groups = meta.row_groups();
    let mins = converter.row_group_mins(row_groups.iter()).ok()?;
    let maxes = converter.row_group_maxes(row_groups.iter()).ok()?;
    let null_counts = converter.row_group_null_counts(row_groups.iter()).ok()?;
    let min = scalar_extreme(&mins, std::cmp::Ordering::Less)?;
    let max = scalar_extreme(&maxes, std::cmp::Ordering::Greater)?;
    let null_count = null_counts.iter().try_fold(0usize, |acc, count| {
        Some(acc + usize::try_from(count?).ok()?)
    })?;
    Some((min, max, null_count))
}

/// Fold per-row-group stats into a single extreme (`Less` → min, `Greater` → max); `None` on
/// empty input, a missing (null) row-group stat, or incomparable values.
fn scalar_extreme(array: &ArrayRef, keep: std::cmp::Ordering) -> Option<ScalarValue> {
    (0..array.len()).try_fold(None::<ScalarValue>, |best, i| {
        if array.is_null(i) {
            return None;
        }
        let value = ScalarValue::try_from_array(array, i).ok()?;
        Some(Some(match best {
            Some(best) if value.partial_cmp(&best)? != keep => best,
            _ => value,
        }))
    })?
}

/// Attach a [`ParquetAccessPlan`] to every file that has a row-ordinal selection so the parquet
/// opener skips non-selected row groups/rows. Footer fetch failures or out-of-range ordinals
/// leave the file untouched (plain full-file scan) — correctness never depends on the selection
/// being applied; it is purely an optimization.
async fn attach_row_ordinal_access_plans(
    store: Arc<dyn ObjectStore>,
    cache: Arc<dyn FileMetadataCache>,
    files: &mut [(PartitionedFile, Option<Vec<bool>>)],
    selections: &std::collections::HashMap<String, Vec<u64>>,
) {
    const FOOTER_FETCH_CONCURRENCY: usize = 16;
    // Per-file futures own everything they touch (see `derive_common_ordering`): a closure over
    // borrowed data is not lifetime-general and makes the parent scan future fail HRTB inference.
    let matched: Vec<(usize, ObjectMeta, Vec<u64>)> = files
        .iter()
        .enumerate()
        .filter_map(|(idx, (file, dv))| {
            // Deletion-vector keep-masks are consumed POSITIONALLY against the
            // rows the reader emits (see consume_dv_mask); a row-skipping
            // access plan would desynchronize the mask — deleted rows
            // resurface. Same discipline as the predicate-pushdown guard on
            // has_selection_vectors above: never combine the two.
            if dv.is_some() {
                return None;
            }
            let location = file.object_meta.location.as_ref();
            selections
                .iter()
                .find(|(key, _)| path_matches_table_relative(location, key))
                .map(|(_, ordinals)| (idx, file.object_meta.clone(), ordinals.clone()))
        })
        .collect();
    let fetches = matched.into_iter().map(|(idx, object_meta, ordinals)| {
        let (store, cache) = (store.clone(), cache.clone());
        async move {
            let meta = DFParquetMetadata::new(store.as_ref(), &object_meta)
                .with_file_metadata_cache(Some(cache))
                .fetch_metadata()
                .await
                .ok()?;
            let rg_rows: Vec<i64> = meta.row_groups().iter().map(|rg| rg.num_rows()).collect();
            let row_groups = row_groups_for_ordinals(&rg_rows, &ordinals)?;
            access_plan_for_ordinals(&rg_rows, &ordinals)
                .map(|plan| (idx, object_meta.location.to_string(), row_groups, plan))
        }
    });
    let plans: Vec<Option<(usize, String, Vec<usize>, ParquetAccessPlan)>> =
        futures::stream::iter(fetches)
            .buffer_unordered(FOOTER_FETCH_CONCURRENCY)
            .collect()
            .await;
    let mut selected = Vec::new();
    for (idx, file, row_groups, plan) in plans.into_iter().flatten() {
        crate::delta_datafusion::parquet_metrics::record_selected_row_groups(row_groups.len());
        selected.push(format!("{file}:{row_groups:?}"));
        files[idx].0.extensions.insert(plan);
    }
    let _row_selection_span = tracing::info_span!(
        "delta.parquet.row_selection",
        parquet.selected_row_groups = %selected.join(",")
    );
}

fn parquet_plan_span(file_count: usize, planned_bytes: u64, file_ids: &str) -> tracing::Span {
    tracing::info_span!(
        "delta.parquet.plan",
        parquet.files = file_count as i64,
        parquet.bytes = planned_bytes as i64,
        parquet.file_ids = file_ids,
    )
}

fn row_groups_for_ordinals(rg_row_counts: &[i64], ordinals: &[u64]) -> Option<Vec<usize>> {
    let total_rows: u64 = rg_row_counts.iter().map(|&n| n.max(0) as u64).sum();
    let mut ordinals = ordinals.to_vec();
    ordinals.sort_unstable();
    ordinals.dedup();
    if ordinals
        .last()
        .is_some_and(|&ordinal| ordinal >= total_rows)
    {
        return None;
    }
    let (mut offset, mut idx, mut selected) = (0u64, 0usize, Vec::new());
    for (row_group, &rows) in rg_row_counts.iter().enumerate() {
        let end = offset + rows.max(0) as u64;
        let start = idx;
        while idx < ordinals.len() && ordinals[idx] < end {
            idx += 1;
        }
        if idx > start {
            selected.push(row_group);
        }
        offset = end;
    }
    Some(selected)
}

/// Match a store-rooted object path against a table-relative parquet path, requiring the match
/// to start on a full path-segment boundary.
fn path_matches_table_relative(location: &str, key: &str) -> bool {
    !key.is_empty()
        && location.ends_with(key)
        && (location.len() == key.len()
            || location.as_bytes()[location.len() - key.len() - 1] == b'/')
}

/// Build a [`ParquetAccessPlan`] selecting exactly the given global row ordinals, given per
/// row-group row counts. Row groups containing no ordinal are skipped entirely.
///
/// Returns `None` when any ordinal is out of range for the file; the caller then falls back to
/// a plain full-file scan, so a selection can only ever over-select by whole-file fallback,
/// never under-select.
fn access_plan_for_ordinals(rg_row_counts: &[i64], ordinals: &[u64]) -> Option<ParquetAccessPlan> {
    use parquet::arrow::arrow_reader::RowSelector;

    let total_rows: u64 = rg_row_counts.iter().map(|&n| n.max(0) as u64).sum();
    let mut ordinals = ordinals.to_vec();
    ordinals.sort_unstable();
    ordinals.dedup();
    if ordinals
        .last()
        .is_some_and(|&ordinal| ordinal >= total_rows)
    {
        return None;
    }

    let mut plan = ParquetAccessPlan::new_none(rg_row_counts.len());
    let (mut offset, mut idx) = (0u64, 0usize);
    for (rg_idx, &rg_rows) in rg_row_counts.iter().enumerate() {
        let rg_rows = rg_rows.max(0) as u64;
        let rg_end = offset + rg_rows;
        let start = idx;
        while idx < ordinals.len() && ordinals[idx] < rg_end {
            idx += 1;
        }
        if idx > start {
            let mut selectors: Vec<RowSelector> = Vec::new();
            let mut cursor = 0u64;
            for &ordinal in &ordinals[start..idx] {
                let row = ordinal - offset;
                if row > cursor {
                    selectors.push(RowSelector::skip((row - cursor) as usize));
                }
                match selectors.last_mut() {
                    Some(last) if !last.skip => last.row_count += 1,
                    _ => selectors.push(RowSelector::select(1)),
                }
                cursor = row + 1;
            }
            if cursor < rg_rows {
                selectors.push(RowSelector::skip((rg_rows - cursor) as usize));
            }
            // `scan_selection` is a no-op on `Skip` row groups (the `new_none` initial state),
            // so set the selection access directly.
            plan.set(rg_idx, RowGroupAccess::Selection(selectors.into()));
        }
        offset = rg_end;
    }
    Some(plan)
}

// Small helper to reuse some code between exec and exec_meta
fn finalize_transformed_batch(
    batch: RecordBatch,
    scan_plan: &KernelScanPlan,
    file_id_col: Option<(ArrayRef, FieldRef)>,
    schema_adapter: &mut SchemaAdapter,
) -> Result<RecordBatch> {
    let result = if let Some(projection) = scan_plan.contract.result_projection.as_ref() {
        batch.project(projection)?
    } else {
        batch
    };
    // NOTE: most data is read properly typed already, however columns added via
    // literals in the transformations may need to be cast to the physical expected type.
    let result = if result.schema_ref().eq(&scan_plan.contract.result_schema) {
        result
    } else {
        schema_adapter.adapt(result)?
    };
    if let Some((arr, field)) = file_id_col {
        let arr = if arr.data_type() != field.data_type() {
            let options = CastOptions {
                safe: true,
                ..Default::default()
            };
            cast_with_options(arr.as_ref(), field.data_type(), &options)?
        } else {
            arr
        };
        let mut columns = result.columns().to_vec();
        columns.push(arr);
        let mut fields = result.schema().fields().to_vec();
        fields.push(field);
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    } else {
        Ok(result)
    }
}

/// Caches a [`BatchAdapter`] for the most recently seen source schema, avoiding
/// repeated expression-tree construction when consecutive batches share the same
/// physical schema (the common case within a single file).
struct SchemaAdapter {
    factory: BatchAdapterFactory,
    /// Single-entry cache: the source schema for the currently cached adapter.
    cached_source: Option<SchemaRef>,
    cached_adapter: Option<BatchAdapter>,
}

impl SchemaAdapter {
    fn new(target_schema: SchemaRef) -> Self {
        Self {
            factory: BatchAdapterFactory::new(target_schema),
            cached_source: None,
            cached_adapter: None,
        }
    }

    /// Adapt the batch to the target schema, using a cached adapter when the
    /// source schema matches the previous call.
    fn adapt(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let source_schema = batch.schema();
        let can_reuse = matches!(
            (&self.cached_source, &self.cached_adapter),
            (Some(cached_source), Some(_)) if cached_source.eq(&source_schema)
        );
        let needs_rebuild = !can_reuse;
        if needs_rebuild {
            let adapter = self.factory.make_adapter(&source_schema)?;
            self.cached_source = Some(source_schema);
            self.cached_adapter = Some(adapter);
        }
        match self.cached_adapter.as_ref() {
            Some(adapter) => adapter.adapt_batch(&batch),
            None => plan_err!(
                "schema adapter cache entry missing for source schema: {:?}",
                batch.schema()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;
    use arrow_array::Array;
    use arrow_array::{
        BinaryArray, BinaryViewArray, Int32Array, Int64Array, RecordBatch, RecordBatchOptions,
        StringArray, StringViewArray, StructArray,
    };
    use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
    use datafusion::{
        error::DataFusionError,
        physical_plan::collect,
        prelude::{col, lit},
    };
    use object_store::{ObjectStoreExt as _, memory::InMemory};
    use parquet::arrow::ArrowWriter;
    use url::Url;

    use crate::{
        assert_batches_sorted_eq,
        delta_datafusion::{
            DeltaScanConfig, MissingSelectedFilePolicy,
            engine::DataFusionEngine,
            session::create_session,
            table_provider::next::{FILE_ID_COLUMN_DEFAULT, FileSelection},
        },
        kernel::Snapshot,
        test_utils::{TestResult, TestTables},
    };

    use super::{plan::build_parquet_predicate_schema, *};

    #[test]
    fn parquet_plan_span_declares_exported_fields() {
        let span = parquet_plan_span(2, 1024, "first.parquet,second.parquet");
        let fields = span.metadata().unwrap().fields();
        for field in ["parquet.files", "parquet.bytes", "parquet.file_ids"] {
            assert!(fields.field(field).is_some(), "missing {field}");
        }
    }

    #[test]
    fn test_partitioned_files_to_file_groups_respects_dictionary_cardinality_limit() {
        let files = (0..=MAX_PARTITION_DICT_CARDINALITY)
            .map(|i| PartitionedFile::new(format!("memory:///f{i}.parquet"), 0))
            .collect_vec();

        let groups = partitioned_files_to_file_groups(files);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), MAX_PARTITION_DICT_CARDINALITY);
        assert_eq!(groups[1].len(), 1);
    }

    fn ts_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("ts", DataType::Int64, false)]))
    }

    fn ts_ordering(descending: bool) -> LexOrdering {
        use arrow_schema::SortOptions;
        use datafusion::physical_expr::{PhysicalSortExpr, expressions::Column};
        LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new("ts", 0)),
            SortOptions {
                descending,
                nulls_first: false,
            },
        )])
        .unwrap()
    }

    fn ordered_file(path: &str, min: i64, max: i64) -> PartitionedFile {
        let mut file = PartitionedFile::new(format!("memory:///{path}.parquet"), 0);
        file.statistics = Some(Arc::new(Statistics {
            num_rows: Precision::Exact(1),
            total_byte_size: Precision::Exact(0),
            column_statistics: vec![ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Exact(ScalarValue::Int64(Some(min))),
                max_value: Precision::Exact(ScalarValue::Int64(Some(max))),
                ..ColumnStatistics::new_unknown()
            }],
        }));
        file
    }

    fn group_ranges(group: &FileGroup) -> Vec<(i64, i64)> {
        group
            .iter()
            .map(|file| {
                let stats = file.statistics.as_ref().expect("file stats");
                let value = |precision: &Precision<ScalarValue>| match precision.get_value() {
                    Some(ScalarValue::Int64(Some(v))) => *v,
                    other => panic!("unexpected stat {other:?}"),
                };
                (
                    value(&stats.column_statistics[0].min_value),
                    value(&stats.column_statistics[0].max_value),
                )
            })
            .collect()
    }

    #[test]
    fn test_regroup_overlapping_files_yields_validating_groups() {
        // Merge-on-read shape: an UPDATE appended old timestamps into a new file, so the
        // lead-key ranges overlap and snapshot-order grouping can never validate.
        let files = vec![
            ordered_file("a", 0, 100),
            ordered_file("b", 50, 150),
            ordered_file("c", 120, 200),
        ];
        let groups = partitioned_files_to_file_groups(files);
        assert_eq!(groups.len(), 1);

        let regrouped = regroup_for_declared_ordering(groups, &ts_ordering(true), &ts_schema());

        let total: usize = regrouped.iter().map(FileGroup::len).sum();
        assert_eq!(total, 3, "no file may be lost or duplicated");
        assert!(
            regrouped.len() >= 2,
            "overlapping files cannot share one group"
        );
        for group in &regrouped {
            // Every multi-file group must be strictly non-overlapping and listed in the
            // declared (DESC) order — exactly what DataFusion's validator re-checks.
            for window in group_ranges(group).windows(2) {
                let [(prev_min, _), (_, next_max)] = window else {
                    unreachable!()
                };
                assert!(
                    prev_min > next_max,
                    "group files overlap or are misordered: {window:?}"
                );
            }
        }
    }

    #[test]
    fn test_regroup_disjoint_files_stay_packed_in_declared_order() {
        let files = vec![
            ordered_file("mid", 10, 19),
            ordered_file("hi", 20, 29),
            ordered_file("lo", 0, 9),
        ];

        let asc = regroup_for_declared_ordering(
            partitioned_files_to_file_groups(files.clone()),
            &ts_ordering(false),
            &ts_schema(),
        );
        assert_eq!(asc.len(), 1, "disjoint files must not fan out into singles");
        assert_eq!(group_ranges(&asc[0]), vec![(0, 9), (10, 19), (20, 29)]);

        let desc = regroup_for_declared_ordering(
            partitioned_files_to_file_groups(files),
            &ts_ordering(true),
            &ts_schema(),
        );
        assert_eq!(desc.len(), 1);
        assert_eq!(group_ranges(&desc[0]), vec![(20, 29), (10, 19), (0, 9)]);
    }

    #[test]
    fn test_regroup_missing_stats_keeps_original_grouping() {
        let mut no_stats = ordered_file("b", 50, 150);
        no_stats.statistics = None;
        let files = vec![
            ordered_file("a", 0, 100),
            no_stats,
            ordered_file("c", 120, 200),
        ];
        let groups = partitioned_files_to_file_groups(files);

        let regrouped =
            regroup_for_declared_ordering(groups.clone(), &ts_ordering(true), &ts_schema());

        let paths = |groups: &[FileGroup]| {
            groups
                .iter()
                .map(|g| {
                    g.iter()
                        .map(|f| f.object_meta.location.to_string())
                        .collect_vec()
                })
                .collect_vec()
        };
        assert_eq!(paths(&regrouped), paths(&groups));
    }

    #[test]
    fn test_stats_backed_prefix_len_stops_at_first_statless_sort_column() {
        use arrow_schema::SortOptions;
        use datafusion::physical_expr::{PhysicalSortExpr, expressions::Column};

        // Two sort columns; only the lead one (ts) has min/max stats — the MOR reality where
        // the tiebreak id is not a predicate column so Delta never materializes its stats.
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new(
                Arc::new(Column::new("ts", 0)),
                SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            ),
            PhysicalSortExpr::new(Arc::new(Column::new("id", 1)), SortOptions::default()),
        ])
        .unwrap();
        let files = vec![
            (ordered_file("a", 0, 100), None::<Vec<bool>>),
            (ordered_file("b", 50, 150), None),
        ];
        assert_eq!(stats_backed_prefix_len(&files, &ordering), 1);

        // A single file without stats disables even the lead column.
        let mut no_stats = ordered_file("c", 120, 200);
        no_stats.statistics = None;
        let mut files = files;
        files.push((no_stats, None));
        assert_eq!(stats_backed_prefix_len(&files, &ordering), 0);
    }

    #[test]
    fn test_declared_ordering_survives_validation_after_regroup() {
        use datafusion::physical_plan::ExecutionPlanProperties;

        let schema = ts_schema();
        let ordering = ts_ordering(true);
        let files = vec![
            ordered_file("a", 0, 100),
            ordered_file("b", 50, 150),
            ordered_file("c", 120, 200),
        ];
        let build_plan = |groups: Vec<FileGroup>| -> Arc<dyn ExecutionPlan> {
            let source = ParquetSource::new(TableSchema::new(schema.clone(), vec![]));
            let config = FileScanConfigBuilder::new(
                ObjectStoreUrl::parse("memory:///").unwrap(),
                Arc::new(source),
            )
            .with_file_groups(groups)
            .with_output_ordering(vec![ordering.clone()])
            .build();
            DataSourceExec::from_data_source(config)
        };

        // Snapshot-order grouping: DataFusion's stats re-validation silently drops the claim.
        let naive = build_plan(partitioned_files_to_file_groups(files.clone()));
        assert!(
            naive.output_ordering().is_none(),
            "overlapping files in one group must invalidate the declared ordering"
        );

        // Repacked grouping: the same declaration survives validation.
        let repacked = build_plan(regroup_for_declared_ordering(
            partitioned_files_to_file_groups(files),
            &ordering,
            &schema,
        ));
        assert!(
            repacked.output_ordering().is_some(),
            "repacked groups must keep the declared ordering through validation"
        );
    }

    #[test]
    fn test_access_plan_for_ordinals() {
        use parquet::arrow::arrow_reader::RowSelector;

        // Row groups of 3 and 4 rows; ordinals {1, 5, 6} (unsorted input).
        let plan = access_plan_for_ordinals(&[3, 4], &[5, 1, 6]).unwrap();
        assert_eq!(
            plan.inner(),
            &[
                // RG0: row 1 → skip 1, select 1, skip 1
                RowGroupAccess::Selection(
                    vec![
                        RowSelector::skip(1),
                        RowSelector::select(1),
                        RowSelector::skip(1),
                    ]
                    .into()
                ),
                // RG1: rows {2, 3} → skip 2, select 2 (coalesced, no trailing skip)
                RowGroupAccess::Selection(
                    vec![RowSelector::skip(2), RowSelector::select(2)].into()
                ),
            ]
        );

        // Row group with no ordinals stays skipped.
        let plan = access_plan_for_ordinals(&[3, 4], &[0]).unwrap();
        assert_eq!(plan.inner()[1], RowGroupAccess::Skip);

        // Out-of-range ordinal → no plan (whole-file fallback).
        assert!(access_plan_for_ordinals(&[3, 4], &[7]).is_none());

        // Empty ordinals select nothing.
        let plan = access_plan_for_ordinals(&[3, 4], &[]).unwrap();
        assert_eq!(plan.inner(), &[RowGroupAccess::Skip, RowGroupAccess::Skip]);
    }

    #[test]
    fn test_path_matches_table_relative() {
        assert!(path_matches_table_relative(
            "timefusion/default/t/project_id=x/part-0.parquet",
            "project_id=x/part-0.parquet"
        ));
        assert!(path_matches_table_relative(
            "project_id=x/part-0.parquet",
            "project_id=x/part-0.parquet"
        ));
        // Partial segment must not match.
        assert!(!path_matches_table_relative(
            "t/other_project_id=x/part-0.parquet",
            "project_id=x/part-0.parquet"
        ));
        assert!(!path_matches_table_relative("t/part-0.parquet", ""));
    }

    #[tokio::test]
    async fn test_resolve_empty_file_selection_does_not_poll_metadata_stream() -> TestResult {
        let log_store = TestTables::Simple.table_builder()?.build_storage()?;
        let snapshot = Snapshot::try_new(&log_store, Default::default(), None).await?;
        let scan_plan =
            KernelScanPlan::try_new(&snapshot, None, &[], &DeltaScanConfig::default(), None)?;
        let stream: ScanMetadataStream = Box::pin(futures::stream::poll_fn(|_| {
            panic!("unexpected metadata stream poll for empty file selection")
        }));

        let resolved = resolve_file_selection(
            &FileSelection::from_file_paths(Vec::<String>::new()),
            &scan_plan,
            stream,
        )
        .await?;

        assert!(resolved.active_file_ids.is_empty());
        assert!(resolved.missing_file_ids.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_resolved_file_selection_plan_does_not_poll_metadata_stream() -> TestResult {
        let log_store = TestTables::Simple.table_builder()?.build_storage()?;
        let snapshot = Snapshot::try_new(&log_store, Default::default(), None).await?;
        let scan_plan =
            KernelScanPlan::try_new(&snapshot, None, &[], &DeltaScanConfig::default(), None)?;
        let stream: ScanMetadataStream = Box::pin(futures::stream::poll_fn(|_| {
            panic!("unexpected metadata stream poll for empty file selection execution")
        }));
        let session = create_session().into_inner();
        let state = session.state();
        let engine = DataFusionEngine::new_from_session(&state);
        let selection = ResolvedFileSelection::new(
            std::collections::HashSet::new(),
            Vec::new(),
            MissingSelectedFilePolicy::Error,
        );

        let plan = execution_plan(
            &DeltaScanConfig::default(),
            &state,
            scan_plan,
            stream,
            engine,
            None,
            Some(&selection),
            None,
        )
        .await?;

        assert!(plan.is::<EmptyExec>());

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_uses_compact_internal_file_id_partition_values() -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = create_session().into_inner();

        let scan = provider.scan(&session.state(), None, &[], None).await?;
        let exec = scan
            .downcast_ref::<DeltaScanExec>()
            .expect("expected DeltaScanExec");
        let data_source = exec.children()[0]
            .downcast_ref::<DataSourceExec>()
            .expect("expected DataSourceExec child");
        let (file_scan_config, _) = data_source
            .downcast_to_file_source::<ParquetSource>()
            .expect("expected parquet file source");

        let internal_file_ids = file_scan_config
            .file_groups
            .iter()
            .flat_map(|group| group.iter())
            .map(|file| {
                file.partition_values
                    .first()
                    .and_then(|value| value.try_as_str().flatten())
                    .expect("file-id partition value")
            })
            .collect_vec();

        assert!(
            !internal_file_ids.is_empty(),
            "test fixture should plan at least one file"
        );
        for internal_file_id in internal_file_ids {
            assert!(
                internal_file_id.len() <= 20,
                "internal file id should be compact, got {internal_file_id:?}"
            );
            assert!(
                !internal_file_id.contains('/'),
                "internal file id should not carry a full file path, got {internal_file_id:?}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_public_file_id_uses_file_path_with_compact_internal_ids() -> TestResult {
        let table = TestTables::Simple.table_builder()?.load().await?;
        let provider = table.table_provider().with_file_column("file_id").await?;
        let session = create_session().into_inner();

        session.register_table("delta_table", provider)?;

        let file_id_batches = session
            .sql("SELECT CAST(file_id AS STRING) AS file_id FROM delta_table LIMIT 1")
            .await?
            .collect()
            .await?;
        let file_id = file_id_batches[0].column(0).as_string_view().value(0);
        assert!(
            file_id.starts_with("file://") && file_id.ends_with(".parquet"),
            "public file id should remain a file path, got {file_id:?}"
        );

        let escaped_file_id = file_id.replace('\'', "''");
        let df = session
            .sql(&format!(
                "SELECT id FROM delta_table WHERE file_id = '{escaped_file_id}'"
            ))
            .await?;
        let filtered = df.collect().await?;

        assert!(filtered.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0);
        assert!(filtered[0].schema().column_with_name("file_id").is_none());

        Ok(())
    }

    fn scan_file_context(file_url: &str) -> ScanFileContext {
        ScanFileContext {
            file_url: Url::parse(file_url).expect("valid test URL"),
            size: 0,
            transform: None,
            stats: Statistics::new_unknown(&Schema::empty()),
            partitions: None,
        }
    }

    #[test]
    fn test_remap_deletion_vectors_to_internal_file_ids_uses_compact_keys() -> TestResult {
        let files = vec![
            scan_file_context("s3://bucket/very/long/path/first.parquet"),
            scan_file_context("s3://bucket/very/long/path/second.parquet"),
        ];
        let mut dvs_by_url = HashMap::new();
        dvs_by_url.insert(files[1].file_url.to_string(), vec![true, false, true]);

        let dvs = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url)?;

        assert!(dvs.contains_key("1"));
        assert!(!dvs.contains_key(files[1].file_url.as_str()));
        assert_eq!(
            dvs.get("1").expect("compact dv key").as_slice(),
            &[true, false, true]
        );
        Ok(())
    }

    #[test]
    fn test_remap_deletion_vectors_to_internal_file_ids_errors_for_unknown_url() {
        let files = vec![scan_file_context(
            "s3://bucket/very/long/path/first.parquet",
        )];
        let mut dvs_by_url = HashMap::new();
        dvs_by_url.insert(
            "s3://bucket/very/long/path/missing.parquet?X-Amz-Signature=secret-token".to_string(),
            vec![true],
        );

        let err = remap_deletion_vectors_to_internal_file_ids(&files, dvs_by_url).unwrap_err();
        let err = err.to_string();
        assert!(
            err.contains("missing internal file id mapping for file with deletion vector"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("secret-token"),
            "error should redact URL query secrets: {err}"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "row indexes require whole file ownership")]
    fn test_partitioned_files_to_file_groups_rejects_split_file_across_groups_in_debug() {
        let files = vec![
            PartitionedFile::new("memory:///same.parquet", 0),
            PartitionedFile::new("memory:///other.parquet", 0),
            PartitionedFile::new("memory:///same.parquet", 0),
        ];

        let _ = partitioned_files_to_file_groups_with_limit(files, 1);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_short_mask_with_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(vec![true, false], Some(4), &url).unwrap();
        assert_eq!(actual, vec![true, false, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_keeps_equal_length_mask() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let mask = vec![true, false, true];
        let actual = normalize_dv_keep_mask_for_api(mask.clone(), Some(3), &url).unwrap();
        assert_eq!(actual, mask);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_pads_empty_mask_to_all_true() {
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let actual = normalize_dv_keep_mask_for_api(Vec::new(), Some(3), &url).unwrap();
        assert_eq!(actual, vec![true, true, true]);
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_mask_longer_than_num_records() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true, false, true], Some(2), &url)
            .expect_err("longer mask should error");
        let message = err.to_string();
        assert!(message.contains("exceeds numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_missing() {
        let url =
            Url::parse("s3://user:secret@example.com/table/file.parquet?sig=token#frag").unwrap();
        let expected_url = super::super::redact_url_for_error(&url);
        let err = normalize_dv_keep_mask_for_api(vec![true], None, &url)
            .expect_err("missing numRecords should error");
        let message = err.to_string();
        assert!(message.contains("Missing numRecords"));
        assert!(message.contains(&expected_url));
        assert!(!message.contains("sig=token"));
        assert!(!message.contains("secret"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_normalize_dv_keep_mask_for_api_errors_when_num_records_overflow_usize() {
        // This branch is only reachable on 32-bit targets where u64 may exceed usize.
        let url = Url::parse("file:///tmp/table/file.parquet").unwrap();
        let overflow_num_records = (usize::MAX as u64) + 1;
        let err = normalize_dv_keep_mask_for_api(vec![true], Some(overflow_num_records), &url)
            .expect_err("numRecords that does not fit usize should error");
        assert!(err.to_string().contains("does not fit usize"));
    }

    #[test]
    fn test_schema_adapter_synthesizes_nullable_columns() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(2)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 2);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.null_count(), 2);
    }

    #[test]
    fn test_schema_adapter_missing_non_nullable_column_errors() {
        let source_schema = Arc::new(Schema::new(Fields::empty()));
        let source = RecordBatch::try_new_with_options(
            source_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("missing non-nullable columns should error");
        match err {
            DataFusionError::Execution(msg) => {
                assert!(
                    msg.contains("Non-nullable column 'id'"),
                    "expected non-nullable missing-column error, got: {msg}"
                );
                assert!(
                    msg.contains("missing from the physical schema"),
                    "expected missing physical schema detail, got: {msg}"
                );
            }
            other => {
                panic!("expected execution error for missing non-nullable column, got: {other}")
            }
        }
    }

    #[test]
    fn test_schema_adapter_invalid_scalar_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(StringArray::from(vec![Some("not-an-int")]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("invalid value cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for invalid scalar cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_type_widening() {
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let mut adapter = SchemaAdapter::new(target_schema.clone());
        let adapted = adapter.adapt(source).unwrap();

        assert_eq!(adapted.schema().as_ref(), target_schema.as_ref());
        assert_eq!(adapted.num_rows(), 3);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[1i64, 2, 3]);
    }

    #[test]
    fn test_schema_adapter_overflow_cast_errors() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let source = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(Int64Array::from(vec![i64::from(i32::MAX) + 1]))],
        )
        .unwrap();

        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)]));
        let mut adapter = SchemaAdapter::new(target_schema);
        let err = adapter
            .adapt(source)
            .expect_err("overflow cast should fail under DataFusion default cast semantics");
        match err {
            DataFusionError::ArrowError(inner, _) => {
                assert!(
                    matches!(inner.as_ref(), ArrowError::CastError(_)),
                    "expected arrow cast error, got: {inner}"
                );
            }
            other => panic!("expected arrow cast error for overflow cast, got: {other}"),
        }
    }

    #[test]
    fn test_schema_adapter_caches_across_calls() {
        let source_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let target_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let mut adapter = SchemaAdapter::new(target_schema);

        let batch1 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .unwrap();

        let _ = adapter.adapt(batch1).unwrap();
        assert!(adapter.cached_source.is_some());

        // Second call with the same schema should hit the cache (no rebuild).
        let adapted = adapter.adapt(batch2).unwrap();
        assert_eq!(adapted.num_rows(), 1);
        let id = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(id.values(), &[2i64]);
    }

    #[tokio::test]
    async fn test_parquet_plan() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "| 2  | b     | memory:///test_data.parquet |",
            "| 3  | c     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // respect limits
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            Some(1),
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        // extended schema with missing column
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
            Field::new("value2", DataType::Utf8, true),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            Some(1),
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+--------+-----------------------------+",
            "| id | value | value2 | __delta_rs_file_id__        |",
            "+----+-------+--------+-----------------------------+",
            "| 1  | a     |        | memory:///test_data.parquet |",
            "+----+-------+--------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_nested() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let nested_fields: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("nested", DataType::Struct(nested_fields.clone()), true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StructArray::try_new(
                    nested_fields,
                    vec![
                        Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
                        Arc::new(StringArray::from(vec![Some("aa"), Some("bb"), Some("cc")])),
                    ],
                    None,
                )?),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+---------------+-----------------------------+",
            "| id | nested        | __delta_rs_file_id__        |",
            "+----+---------------+-----------------------------+",
            "| 1  | {a: a, b: aa} | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb} | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc} | memory:///test_data.parquet |",
            "+----+---------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        let nested_fields_extended: Fields = vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Utf8, true),
        ]
        .into();
        let arrow_schema_extended = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "nested",
                DataType::Struct(nested_fields_extended.clone()),
                true,
            ),
        ]));
        let parquet_predicate_schema_extended =
            build_parquet_predicate_schema(&arrow_schema_extended, &file_id_field);
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema_extended,
            &parquet_predicate_schema_extended,
            None,
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+--------------------+-----------------------------+",
            "| id | nested             | __delta_rs_file_id__        |",
            "+----+--------------------+-----------------------------+",
            "| 1  | {a: a, b: aa, c: } | memory:///test_data.parquet |",
            "| 2  | {a: b, b: bb, c: } | memory:///test_data.parquet |",
            "| 3  | {a: c, b: cc, c: } | memory:///test_data.parquet |",
            "+----+--------------------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_multiple_stores() -> TestResult {
        let store_1 = Arc::new(InMemory::new());
        let store_url_1 = Url::parse("first:///")?;
        let store_2 = Arc::new(InMemory::new());
        let store_url_2 = Url::parse("second:///")?;

        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url_1, store_1.clone());
        session
            .runtime_env()
            .register_object_store(&store_url_2, store_2.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));

        let data_1 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec![Some("a")])),
            ],
        )?;
        let data_2 = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(StringArray::from(vec![Some("b")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_1)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_1.put(&path, buffer.into()).await?;
        let mut file_1: PartitionedFile = store_1.head(&path).await?.into();
        file_1
            .partition_values
            .push(wrap_file_id_value("first:///test_data.parquet"));

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data_2)?;
        arrow_writer.close()?;
        let path = Path::from("test_data.parquet");
        store_2.put(&path, buffer.into()).await?;
        let mut file_2: PartitionedFile = store_2.head(&path).await?.into();
        file_2
            .partition_values
            .push(wrap_file_id_value("second:///test_data.parquet"));

        let files_by_store = vec![
            (
                store_url_1.as_object_store_url(),
                vec![(file_1, None::<Vec<bool>>)],
            ),
            (
                store_url_2.as_object_store_url(),
                vec![(file_2, None::<Vec<bool>>)],
            ),
        ];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            None,
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 1  | a     | first:///test_data.parquet  |",
            "| 2  | b     | second:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_plan_predicate() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, arrow_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_data.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_data.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&arrow_schema, &file_id_field);

        let predicate = col("id").eq(lit(2i32));
        let plan = get_read_plan(
            &session.state(),
            files_by_store.clone(),
            &arrow_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+-------+-----------------------------+",
            "| id | value | __delta_rs_file_id__        |",
            "+----+-------+-----------------------------+",
            "| 2  | b     | memory:///test_data.parquet |",
            "+----+-------+-----------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_skips_pushdown_when_logical_rewrite_fails() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let parquet_read_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("missing", DataType::Int32, false),
        ]));
        let data = RecordBatch::try_new(
            parquet_read_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer =
            ArrowWriter::try_new(&mut buffer, parquet_read_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_rewrite_failure.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_rewrite_failure.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&logical_schema, &file_id_field);
        let predicate = col("missing").eq(lit(1i32));

        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        let expected = vec![
            "+----+----------------------------------------+",
            "| id | __delta_rs_file_id__                   |",
            "+----+----------------------------------------+",
            "| 1  | memory:///test_rewrite_failure.parquet |",
            "| 2  | memory:///test_rewrite_failure.parquet |",
            "| 3  | memory:///test_rewrite_failure.parquet |",
            "+----+----------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_view_literal_against_base_parquet_file() -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_view_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_view_literal.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit(ScalarValue::Utf8View(Some("bob".to_string()))));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+-------------------------------------+",
            "| id | name | __delta_rs_file_id__                |",
            "+----+------+-------------------------------------+",
            "| 2  | bob  | memory:///test_view_literal.parquet |",
            "+----+------+-------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_sql_literal_against_view_schema() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        // Write a Parquet file with base types, but read it with a view-typed schema.
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_sql_literal.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_sql_literal.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("name").eq(lit("bob"));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        let expected = vec![
            "+----+------+------------------------------------+",
            "| id | name | __delta_rs_file_id__               |",
            "+----+------+------------------------------------+",
            "| 2  | bob  | memory:///test_sql_literal.parquet |",
            "+----+------+------------------------------------+",
        ];
        assert_batches_sorted_eq!(&expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_physical_column_mapping_names() -> TestResult {
        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let physical_name = "col-3877fd94-0973-4941-ac6b-646849a1ff65";
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(physical_name, DataType::Utf8View, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    Some("alice"),
                    Some("bob"),
                    Some("charlie"),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_column_mapping_pushdown.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values.push(wrap_file_id_value(
            "memory:///test_column_mapping_pushdown.parquet",
        ));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col(physical_name).eq(lit("bob"));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 3);

        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let name_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        assert_eq!(name_col.value(0), "bob");

        assert_eq!(batches[0].schema().field(1).name(), physical_name);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }

    #[tokio::test]
    async fn test_predicate_pushdown_allows_binaryview_literal_against_base_parquet_file()
    -> TestResult {
        use datafusion::scalar::ScalarValue;

        let store = Arc::new(InMemory::new());
        let store_url = Url::parse("memory:///")?;
        let session = Arc::new(create_session().into_inner());
        session
            .runtime_env()
            .register_object_store(&store_url, store.clone());

        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Binary, true),
        ]));
        let parquet_read_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::BinaryView, true),
        ]));
        let data = RecordBatch::try_new(
            file_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(BinaryArray::from_opt_vec(vec![
                    Some(b"aaa".as_slice()),
                    Some(b"bbb".as_slice()),
                    Some(b"ccc".as_slice()),
                ])),
            ],
        )?;

        let mut buffer = Vec::new();
        let mut arrow_writer = ArrowWriter::try_new(&mut buffer, file_schema.clone(), None)?;
        arrow_writer.write(&data)?;
        arrow_writer.close()?;

        let path = Path::from("test_binary_view.parquet");
        store.put(&path, buffer.into()).await?;
        let mut file: PartitionedFile = store.head(&path).await?.into();
        file.partition_values
            .push(wrap_file_id_value("memory:///test_binary_view.parquet"));

        let files_by_store = vec![(
            store_url.as_object_store_url(),
            vec![(file, None::<Vec<bool>>)],
        )];

        let file_id_field =
            crate::delta_datafusion::file_id::file_id_field(Some(FILE_ID_COLUMN_DEFAULT));
        let parquet_predicate_schema =
            build_parquet_predicate_schema(&parquet_read_schema, &file_id_field);

        let predicate = col("data").eq(lit(ScalarValue::BinaryView(Some(b"bbb".to_vec()))));
        let plan = get_read_plan(
            &session.state(),
            files_by_store,
            &parquet_read_schema,
            &parquet_predicate_schema,
            None,
            &file_id_field,
            Some(&predicate),
            None,
        )
        .await?;
        let batches = collect(plan, session.task_ctx()).await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let id_col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 2);

        let data_col = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data_col.value(0), b"bbb");

        assert_eq!(batches[0].num_columns(), 3);
        assert_eq!(batches[0].schema().field(2).name(), FILE_ID_COLUMN_DEFAULT);

        Ok(())
    }
}
