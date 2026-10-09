use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering::Relaxed},
};

use bytes::Bytes;
use datafusion::{
    common::{HashMap, TableReference},
    datasource::physical_plan::parquet::{
        CachedParquetFileReaderFactory, ParquetFileReaderFactory,
    },
    execution::cache::{
        Cache, CacheEntryInfo,
        cache_manager::{CachedFileMetadataEntry, FileMetadataCache},
    },
    physical_plan::metrics::ExecutionPlanMetricsSet,
};
use datafusion_datasource::PartitionedFile;
use futures::{FutureExt, future::BoxFuture};
use object_store::{ObjectStore, path::Path};
use parquet::{
    arrow::{arrow_reader::ArrowReaderOptions, async_reader::AsyncFileReader},
    errors::Result as ParquetResult,
};
use std::{
    ops::Range,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ParquetScanMetrics {
    pub metadata_cache_hits: u64,
    pub metadata_cache_misses: u64,
    pub bytes_read: u64,
    pub read_time_us: u64,
    pub scans: u64,
    pub files_planned: u64,
    pub bytes_planned: u64,
    pub selected_row_groups: u64,
    /// Microseconds `DeltaScan::scan` spent per planning phase, see [`PlanPhase`].
    pub plan_phase_us: [u64; PlanPhase::COUNT],
    /// Scans whose file replay could not seed from materialized files and re-read the log.
    pub unseeded_replays: u64,
}

/// Where `DeltaScan::scan` spends its planning time.
#[derive(Debug, Clone, Copy)]
pub enum PlanPhase {
    KernelPlan,
    FileSelection,
    Replay,
    FooterOrdering,
    Build,
}

impl PlanPhase {
    pub const COUNT: usize = 5;
    pub const NAMES: [&str; Self::COUNT] = [
        "kernel_plan",
        "file_selection",
        "replay",
        "footer_ordering",
        "build",
    ];
}

static PLAN_PHASE_US: [AtomicU64; PlanPhase::COUNT] =
    [const { AtomicU64::new(0) }; PlanPhase::COUNT];
static UNSEEDED_REPLAYS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn record_unseeded_replay() {
    UNSEEDED_REPLAYS.fetch_add(1, Relaxed);
}

pub(crate) fn record_plan_phase(phase: PlanPhase, started: Instant) {
    PLAN_PHASE_US[phase as usize].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
}

static METADATA_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static METADATA_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static BYTES_READ: AtomicU64 = AtomicU64::new(0);
static READ_TIME_US: AtomicU64 = AtomicU64::new(0);
static SCANS: AtomicU64 = AtomicU64::new(0);
static FILES_PLANNED: AtomicU64 = AtomicU64::new(0);
static BYTES_PLANNED: AtomicU64 = AtomicU64::new(0);
static SELECTED_ROW_GROUPS: AtomicU64 = AtomicU64::new(0);

pub struct InstrumentedFileMetadataCache {
    inner: Arc<FileMetadataCache>,
}

impl InstrumentedFileMetadataCache {
    pub fn new(inner: Arc<FileMetadataCache>) -> Self {
        Self { inner }
    }
}

impl Cache<Path, CachedFileMetadataEntry> for InstrumentedFileMetadataCache {
    fn get(&self, key: &Path) -> Option<CachedFileMetadataEntry> {
        let value = self.inner.get(key);
        record_metadata_lookup(value.is_some());
        value
    }

    fn put(&self, key: &Path, value: CachedFileMetadataEntry) -> Option<CachedFileMetadataEntry> {
        self.inner.put(key, value)
    }

    fn remove(&self, key: &Path) -> Option<CachedFileMetadataEntry> {
        self.inner.remove(key)
    }

    fn contains_key(&self, key: &Path) -> bool {
        self.inner.contains_key(key)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) {
        self.inner.clear();
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn cache_limit(&self) -> usize {
        self.inner.cache_limit()
    }

    fn update_cache_limit(&self, limit: usize) {
        self.inner.update_cache_limit(limit);
    }

    fn cache_ttl(&self) -> Option<Duration> {
        self.inner.cache_ttl()
    }

    fn update_cache_ttl(&self, ttl: Option<Duration>) {
        self.inner.update_cache_ttl(ttl);
    }

    fn drop_table_entries(&self, table_ref: &TableReference) -> datafusion::common::Result<()> {
        self.inner.drop_table_entries(table_ref)
    }

    fn list_entries(&self) -> HashMap<Path, CacheEntryInfo<CachedFileMetadataEntry>> {
        self.inner.list_entries()
    }
}

#[derive(Debug)]
pub struct InstrumentedParquetFileReaderFactory {
    inner: CachedParquetFileReaderFactory,
}

impl InstrumentedParquetFileReaderFactory {
    pub fn new(store: Arc<dyn ObjectStore>, metadata_cache: Arc<FileMetadataCache>) -> Self {
        Self {
            inner: CachedParquetFileReaderFactory::new(store, metadata_cache),
        }
    }
}

impl ParquetFileReaderFactory for InstrumentedParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion::common::Result<Box<dyn AsyncFileReader + Send>> {
        Ok(Box::new(InstrumentedParquetFileReader::new(
            self.inner.create_reader(
                partition_index,
                partitioned_file,
                metadata_size_hint,
                metrics,
            )?,
        )))
    }
}

struct InstrumentedParquetFileReader {
    inner: Box<dyn AsyncFileReader + Send>,
    ahead: ReadAhead,
}

impl InstrumentedParquetFileReader {
    fn new(inner: Box<dyn AsyncFileReader + Send>) -> Self {
        Self {
            inner,
            ahead: ReadAhead::default(),
        }
    }
}

/// Per-reader budget for column chunks fetched ahead of the row group that asked.
const READ_AHEAD_BYTES: u64 = 8 * 1024 * 1024;

/// Read-ahead is off until the embedding application turns it on (a runtime flag, so it
/// can be measured on and off within one process).
static READ_AHEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_read_ahead(on: bool) {
    READ_AHEAD.store(on, Relaxed);
}

/// Column-chunk read-ahead. The parquet reader fetches one row group's column chunks
/// per `get_byte_ranges` call, one call after another, so a file of N row groups costs
/// N sequential object-store round trips per column set (~0.5 s each on prod's store).
/// When a call reads most of a column's chunk, the same column's chunks in the next row
/// groups ride along in that call's batch; later calls are served from them.
#[derive(Default)]
struct ReadAhead {
    metadata: Option<Arc<parquet::file::metadata::ParquetMetaData>>,
    /// Fetched chunks by byte range, oldest first.
    chunks: Vec<(Range<u64>, Bytes)>,
}

impl ReadAhead {
    fn serve(&self, range: &Range<u64>) -> Option<Bytes> {
        self.chunks
            .iter()
            .find(|(chunk, _)| chunk.start <= range.start && range.end <= chunk.end)
            .map(|(chunk, bytes)| {
                bytes
                    .slice((range.start - chunk.start) as usize..(range.end - chunk.start) as usize)
            })
    }

    /// Chunks to fetch alongside `misses`: for each column a miss reads at least half
    /// of, that column's chunks in the following row groups, nearest first, in budget.
    fn plan(&self, misses: &[Range<u64>]) -> Vec<Range<u64>> {
        let Some(metadata) = self.metadata.as_ref() else {
            return Vec::new();
        };
        let chunk_range = |rg: usize, col: usize| {
            let (start, len) = metadata.row_group(rg).column(col).byte_range();
            start..start + len
        };
        let overlap =
            |a: &Range<u64>, b: &Range<u64>| a.end.min(b.end).saturating_sub(a.start.max(b.start));
        let mut read: HashMap<(usize, usize), u64> = HashMap::new();
        for miss in misses {
            for rg in 0..metadata.num_row_groups() {
                for col in 0..metadata.row_group(rg).num_columns() {
                    let covered = overlap(miss, &chunk_range(rg, col));
                    if covered > 0 {
                        *read.entry((rg, col)).or_default() += covered;
                    }
                }
            }
        }
        let columns: Vec<usize> = read
            .iter()
            .filter(|((rg, col), bytes)| {
                2 * **bytes >= chunk_range(*rg, *col).end - chunk_range(*rg, *col).start
            })
            .map(|(&(_, col), _)| col)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let Some(&last) = read.keys().map(|(rg, _)| rg).max() else {
            return Vec::new();
        };
        let mut budget = READ_AHEAD_BYTES;
        let mut ahead = Vec::new();
        'groups: for rg in last + 1..metadata.num_row_groups() {
            for &col in &columns {
                let chunk = chunk_range(rg, col);
                let len = chunk.end - chunk.start;
                if self.serve(&chunk).is_some() {
                    continue;
                }
                if len > budget {
                    break 'groups;
                }
                budget -= len;
                ahead.push(chunk);
            }
        }
        ahead
    }

    fn keep(&mut self, fetched: impl IntoIterator<Item = (Range<u64>, Bytes)>) {
        self.chunks.extend(fetched);
        // Oldest out first: the reader moves forward through the file.
        let mut held: u64 = self
            .chunks
            .iter()
            .map(|(range, _)| range.end - range.start)
            .sum();
        while held > 2 * READ_AHEAD_BYTES && !self.chunks.is_empty() {
            let (range, _) = self.chunks.remove(0);
            held -= range.end - range.start;
        }
    }
}

impl AsyncFileReader for InstrumentedParquetFileReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let started = Instant::now();
        async move {
            if let Some(bytes) = self.ahead.serve(&range) {
                return Ok(bytes);
            }
            let result = self.inner.get_bytes(range).await;
            if let Ok(bytes) = &result {
                record_read(bytes.len(), started.elapsed().as_micros() as u64);
            }
            result
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>>
    where
        Self: Send,
    {
        let started = Instant::now();
        async move {
            let served: Vec<Option<Bytes>> =
                ranges.iter().map(|range| self.ahead.serve(range)).collect();
            let misses: Vec<Range<u64>> = ranges
                .iter()
                .zip(&served)
                .filter(|(_, hit)| hit.is_none())
                .map(|(range, _)| range.clone())
                .collect();
            if misses.is_empty() {
                return Ok(served.into_iter().flatten().collect());
            }
            let ahead = if READ_AHEAD.load(Relaxed) {
                self.ahead.plan(&misses)
            } else {
                Vec::new()
            };
            let mut fetched = self
                .inner
                .get_byte_ranges(misses.iter().chain(&ahead).cloned().collect())
                .await?;
            record_read(
                fetched.iter().map(Bytes::len).sum(),
                started.elapsed().as_micros() as u64,
            );
            let extra = fetched.split_off(misses.len());
            self.ahead.keep(ahead.into_iter().zip(extra));
            let mut fetched = fetched.into_iter();
            Ok(served
                .into_iter()
                .map(|hit| hit.or_else(|| fetched.next()).unwrap_or_default())
                .collect())
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<parquet::file::metadata::ParquetMetaData>>> {
        async move {
            let metadata = self.inner.get_metadata(options).await?;
            self.ahead.metadata = Some(Arc::clone(&metadata));
            Ok(metadata)
        }
        .boxed()
    }
}

pub fn snapshot() -> ParquetScanMetrics {
    ParquetScanMetrics {
        metadata_cache_hits: METADATA_CACHE_HITS.load(Relaxed),
        metadata_cache_misses: METADATA_CACHE_MISSES.load(Relaxed),
        bytes_read: BYTES_READ.load(Relaxed),
        read_time_us: READ_TIME_US.load(Relaxed),
        scans: SCANS.load(Relaxed),
        files_planned: FILES_PLANNED.load(Relaxed),
        bytes_planned: BYTES_PLANNED.load(Relaxed),
        selected_row_groups: SELECTED_ROW_GROUPS.load(Relaxed),
        plan_phase_us: std::array::from_fn(|i| PLAN_PHASE_US[i].load(Relaxed)),
        unseeded_replays: UNSEEDED_REPLAYS.load(Relaxed),
    }
}

pub(crate) fn record_metadata_lookup(hit: bool) {
    (if hit {
        &METADATA_CACHE_HITS
    } else {
        &METADATA_CACHE_MISSES
    })
    .fetch_add(1, Relaxed);
}

pub(crate) fn record_read(bytes: usize, elapsed_us: u64) {
    BYTES_READ.fetch_add(bytes as u64, Relaxed);
    READ_TIME_US.fetch_add(elapsed_us, Relaxed);
}

pub(crate) fn record_scan(files: usize, bytes: u64) {
    SCANS.fetch_add(1, Relaxed);
    FILES_PLANNED.fetch_add(files as u64, Relaxed);
    BYTES_PLANNED.fetch_add(bytes, Relaxed);
}

pub(crate) fn record_selected_row_groups(count: usize) {
    SELECTED_ROW_GROUPS.fetch_add(count as u64, Relaxed);
}

#[cfg(test)]
mod read_ahead_tests {
    use std::sync::atomic::AtomicUsize;

    use arrow::{
        array::{Int64Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use futures::TryStreamExt;
    use parquet::{
        arrow::{ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask},
        file::{metadata::ParquetMetaDataReader, properties::WriterProperties},
    };

    use super::*;

    /// An in-memory file that counts `get_byte_ranges` calls: one call is one round trip.
    struct Counting(Bytes, Arc<AtomicUsize>);

    impl AsyncFileReader for Counting {
        fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
            self.1.fetch_add(1, Relaxed);
            let bytes = self.0.slice(range.start as usize..range.end as usize);
            async move { Ok(bytes) }.boxed()
        }

        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
            self.1.fetch_add(1, Relaxed);
            let bytes = ranges
                .iter()
                .map(|range| self.0.slice(range.start as usize..range.end as usize))
                .collect();
            async move { Ok(bytes) }.boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, ParquetResult<Arc<parquet::file::metadata::ParquetMetaData>>> {
            let metadata = ParquetMetaDataReader::new()
                .with_page_index_policy(parquet::file::metadata::PageIndexPolicy::Optional)
                .parse_and_finish(&self.0);
            async move { Ok(Arc::new(metadata?)) }.boxed()
        }
    }

    /// Twenty row groups of `(n, label)`.
    fn file() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..2000)),
                Arc::new(StringArray::from_iter_values(
                    (0..2000).map(|i| format!("row-{i}")),
                )),
            ],
        )
        .unwrap();
        let mut out = Vec::new();
        let mut writer = ArrowWriter::try_new(
            &mut out,
            schema,
            Some(
                WriterProperties::builder()
                    .set_max_row_group_row_count(Some(100))
                    .build(),
            ),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(out)
    }

    async fn read(reader: impl AsyncFileReader + Unpin + Send + 'static) -> Vec<RecordBatch> {
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let mask = ProjectionMask::roots(builder.parquet_schema(), [1]);
        builder
            .with_projection(mask)
            .build()
            .unwrap()
            .try_collect()
            .await
            .unwrap()
    }

    /// Same batches, a fraction of the round trips: a column the reader takes whole
    /// rides ahead into the next row groups.
    #[tokio::test]
    async fn read_ahead_returns_the_same_rows_in_fewer_round_trips() {
        set_read_ahead(true);
        let (plain_calls, ahead_calls) =
            (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let plain = read(Counting(file(), Arc::clone(&plain_calls))).await;
        let ahead = read(InstrumentedParquetFileReader::new(Box::new(Counting(
            file(),
            Arc::clone(&ahead_calls),
        ))))
        .await;
        assert_eq!(ahead, plain);
        let (plain_calls, ahead_calls) = (plain_calls.load(Relaxed), ahead_calls.load(Relaxed));
        assert!(
            plain_calls >= 20,
            "one round trip per row group without read-ahead, got {plain_calls}"
        );
        assert!(
            ahead_calls * 4 <= plain_calls,
            "read-ahead took {ahead_calls} round trips against {plain_calls}"
        );
    }
}
