//! Merge-on-read deletion vector writes.
//!
//! Given a set of data files and the physical row indexes that should be logically
//! deleted from each, this writes Delta deletion-vector files (RoaringBitmapArray
//! format, produced by the kernel's [`StreamingDeletionVectorWriter`]) and returns the
//! `Remove(old add)` + `Add(same path, new DV descriptor)` actions needed to commit the
//! change. Any pre-existing DV on a file is read back and unioned so repeated deletes on
//! the same file accumulate rather than clobber.
//!
//! This is the shared primitive behind merge-on-read DELETE and UPDATE: an UPDATE marks
//! the matched rows deleted here and appends the rewritten rows as new Add actions.

use std::collections::HashSet;

use delta_kernel::actions::deletion_vector_writer::{
    KernelDeletionVector, StreamingDeletionVectorWriter,
};
use object_store::ObjectStoreExt as _;
use object_store::path::Path as ObjectStorePath;
use roaring::RoaringTreemap;
use url::Url;
use uuid::Uuid;

use crate::kernel::{Action, Add, DeletionVectorDescriptor, Remove, StorageType};
use crate::logstore::LogStore;
use crate::{DeltaResult, DeltaTableError};

/// One file's worth of newly-deleted physical row indexes plus its current add action.
pub struct FileDeletion {
    /// The current add action for the file (from `LogicalFileView::to_add`).
    pub add: Add,
    /// Physical row indexes (0-based, within the parquet file) to logically delete now.
    pub deleted_indexes: Vec<u64>,
}

/// Frame layout constants for the Delta DV file, matching the kernel writer.
const DV_SIZE_PREFIX: usize = 4; // big-endian u32: size of (magic + data)
const DV_MAGIC_LEN: usize = 4; // little-endian magic

/// Table-root-relative object-store path of the `.bin` file backing a persisted DV, if any.
///
/// Returns `None` for inline DVs (no file) and for descriptors that don't decode. Used by
/// VACUUM to treat DV files referenced by live Adds as valid (never garbage-collect them).
pub fn dv_object_store_relative_path(desc: &DeletionVectorDescriptor) -> Option<String> {
    match desc.storage_type {
        StorageType::UuidRelativePath => dv_relative_path(desc).ok(),
        // Absolute path: the descriptor already holds the object path/URL.
        StorageType::AbsolutePath => Some(desc.path_or_inline_dv.clone()),
        StorageType::Inline => None,
    }
}

/// Reconstruct the DV file's relative path from a persisted-relative descriptor.
///
/// `path_or_inline_dv` is `<prefix><z85(uuid)>`; the uuid is the trailing 20 chars.
fn dv_relative_path(desc: &DeletionVectorDescriptor) -> DeltaResult<String> {
    let s = &desc.path_or_inline_dv;
    if s.len() < 20 {
        return Err(DeltaTableError::generic(format!(
            "invalid DV path_or_inline_dv length {}",
            s.len()
        )));
    }
    let (prefix, encoded_uuid) = s.split_at(s.len() - 20);
    let decoded = z85::decode(encoded_uuid)
        .map_err(|e| DeltaTableError::generic(format!("failed to decode DV uuid: {e}")))?;
    let uuid = Uuid::from_slice(&decoded)
        .map_err(|e| DeltaTableError::generic(format!("invalid DV uuid bytes: {e}")))?;
    Ok(if prefix.is_empty() {
        format!("deletion_vector_{uuid}.bin")
    } else {
        format!("{prefix}/deletion_vector_{uuid}.bin")
    })
}

/// Read an existing persisted-relative DV back into a [`RoaringTreemap`] so new deletions
/// can be unioned onto it. Inline / absolute DVs are not produced by this writer.
async fn read_existing_dv(
    log_store: &dyn LogStore,
    desc: &DeletionVectorDescriptor,
) -> DeltaResult<RoaringTreemap> {
    if desc.storage_type != StorageType::UuidRelativePath {
        return Err(DeltaTableError::generic(format!(
            "cannot merge DV with unsupported storage type {:?}",
            desc.storage_type
        )));
    }
    let rel = dv_relative_path(desc)?;
    let bytes = log_store
        .object_store(None)
        .get(&ObjectStorePath::from(rel.as_str()))
        .await?
        .bytes()
        .await?;

    // Layout at `offset`: [4B size][4B magic][serialized treemap][4B crc].
    // `size_in_bytes` == magic(4) + serialized data, so data length == size_in_bytes - 4.
    let offset = desc.offset.unwrap_or(0) as usize;
    let data_start = offset + DV_SIZE_PREFIX + DV_MAGIC_LEN;
    let data_len = (desc.size_in_bytes as usize)
        .checked_sub(DV_MAGIC_LEN)
        .ok_or_else(|| DeltaTableError::generic("DV size_in_bytes smaller than magic"))?;
    let data_end = data_start + data_len;
    if data_end > bytes.len() {
        return Err(DeltaTableError::generic(format!(
            "DV frame [{data_start}..{data_end}] exceeds file length {}",
            bytes.len()
        )));
    }
    RoaringTreemap::deserialize_from(&bytes[data_start..data_end])
        .map_err(|e| DeltaTableError::generic(format!("failed to deserialize existing DV: {e}")))
}

/// Write deletion vectors for the given files and return the resulting log actions.
///
/// For each file with at least one newly-deleted row this emits `Remove(old add)` and
/// `Add(same data-file path, merged DV descriptor)`. Files whose merged deletion set is
/// empty are skipped. `table_root` is the table's object-store root URL, used only to
/// keep the descriptor path relative.
pub async fn write_deletion_vectors(
    log_store: &dyn LogStore,
    _table_root: &Url,
    deletions: Vec<FileDeletion>,
) -> DeltaResult<Vec<Action>> {
    let mut actions = Vec::new();
    for FileDeletion {
        add,
        deleted_indexes,
    } in deletions
    {
        if deleted_indexes.is_empty() {
            continue;
        }

        // Start from the existing DV (if any) so repeated deletes accumulate.
        let mut bitmap = match &add.deletion_vector {
            Some(desc) => read_existing_dv(log_store, desc).await?,
            None => RoaringTreemap::new(),
        };
        let before = bitmap.len();
        bitmap.extend(deleted_indexes.iter().copied());
        if bitmap.len() == before {
            // Every index was already deleted — nothing changed for this file.
            continue;
        }

        // Serialize via the kernel writer to guarantee protocol-correct framing/CRC.
        let mut dv = KernelDeletionVector::new();
        dv.add_deleted_row_indexes(bitmap.iter().collect::<HashSet<_>>());
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = StreamingDeletionVectorWriter::new(&mut buf);
        let result = writer
            .write_deletion_vector(dv)
            .map_err(|e| DeltaTableError::generic(format!("DV write failed: {e}")))?;
        writer
            .finalize()
            .map_err(|e| DeltaTableError::generic(format!("DV finalize failed: {e}")))?;

        // Persist the DV file at a UUID-derived relative path and build the descriptor.
        let uuid = Uuid::new_v4();
        let rel = format!("deletion_vector_{uuid}.bin");
        log_store
            .object_store(None)
            .put(&ObjectStorePath::from(rel.as_str()), buf.into())
            .await?;
        let descriptor = DeletionVectorDescriptor {
            storage_type: StorageType::UuidRelativePath,
            path_or_inline_dv: z85::encode(uuid.as_bytes()),
            offset: Some(result.offset),
            size_in_bytes: result.size_in_bytes,
            cardinality: result.cardinality,
        };

        actions.push(Action::Remove(remove_for_add(&add)));
        let mut new_add = add;
        new_add.data_change = true;
        new_add.deletion_vector = Some(descriptor);
        actions.push(Action::Add(new_add));
    }
    Ok(actions)
}

#[cfg(all(test, feature = "datafusion"))]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use futures::TryStreamExt as _;

    use crate::kernel::transaction::CommitBuilder;
    use crate::kernel::{DataType, PrimitiveType, StructField, StructType};
    use crate::protocol::{DeltaOperation, SaveMode};
    use crate::writer::test_utils::datafusion::get_data_sorted;
    use crate::{DeltaResult, DeltaTable, TableProperty};

    fn values_batch(vals: impl IntoIterator<Item = i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            ArrowDataType::Int32,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(vals))]).unwrap()
    }

    fn sorted_values(batches: &[RecordBatch]) -> Vec<i32> {
        let mut out = Vec::new();
        for b in batches {
            let col = b
                .column_by_name("value")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            out.extend(col.iter().flatten());
        }
        out.sort_unstable();
        out
    }

    async fn single_file_add(table: &DeltaTable) -> Add {
        let snapshot = table.snapshot().unwrap().snapshot().clone();
        let views: Vec<_> = snapshot
            .file_views(table.log_store().as_ref(), None)
            .try_collect()
            .await
            .unwrap();
        assert_eq!(views.len(), 1, "expected exactly one data file");
        views[0].to_add()
    }

    async fn commit_dv(table: DeltaTable, deleted_indexes: Vec<u64>) -> DeltaResult<DeltaTable> {
        let add = single_file_add(&table).await;
        let log_store = table.log_store();
        let root = log_store.root_url();
        let actions = write_deletion_vectors(
            log_store.as_ref(),
            &root,
            vec![FileDeletion {
                add,
                deleted_indexes,
            }],
        )
        .await?;
        let snapshot = table.snapshot()?.snapshot().clone();
        let commit = CommitBuilder::default()
            .with_actions(actions)
            .build(
                Some(&snapshot),
                table.log_store(),
                DeltaOperation::Delete { predicate: None },
            )
            .await?;
        Ok(DeltaTable::new_with_state(
            table.log_store(),
            commit.snapshot(),
        ))
    }

    async fn make_table() -> DeltaTable {
        let schema = StructType::try_new(vec![StructField::new(
            "value",
            DataType::Primitive(PrimitiveType::Integer),
            true,
        )])
        .unwrap();
        let table = DeltaTable::new_in_memory()
            .create()
            .with_columns(schema.fields().cloned())
            .with_configuration_property(TableProperty::EnableDeletionVectors, Some("true"))
            .await
            .unwrap();
        table
            .write(vec![values_batch(0..10)])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn dv_write_hides_exactly_the_deleted_rows() -> DeltaResult<()> {
        let table = make_table().await;
        let table = commit_dv(table, vec![1, 3, 5]).await?;
        let data = get_data_sorted(&table, "value").await;
        assert_eq!(sorted_values(&data), vec![0, 2, 4, 6, 7, 8, 9]);
        Ok(())
    }

    /// Every live Add in the table (each backs one data file here).
    async fn all_adds(table: &DeltaTable) -> Vec<Add> {
        let snapshot = table.snapshot().unwrap().snapshot().clone();
        snapshot
            .file_views(table.log_store().as_ref(), None)
            .map_ok(|v| v.to_add())
            .try_collect()
            .await
            .unwrap()
    }

    /// Scan the given adds exposing per-row source file + physical row index, and
    /// return `(file_path, row_index)` pairs. Uses the fork's `with_file_column`
    /// + `with_row_index_column` — the exact primitives DV-dedup (option D) relies
    /// on to locate losers across multiple files in one scan.
    async fn scan_file_and_row_index(table: &DeltaTable, adds: Vec<Add>) -> Vec<(String, u64)> {
        use arrow::array::{StringArray, UInt64Array};
        use datafusion::prelude::SessionContext;

        let provider = table
            .table_provider()
            .with_file_column("__fpath")
            .with_row_index_column("__ridx")
            .with_adds(adds)
            .build()
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT __fpath, __ridx FROM t ORDER BY __fpath, __ridx")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        use arrow::compute::cast;
        use arrow::datatypes::DataType as ADataType;
        let mut out = Vec::new();
        for b in &batches {
            let paths_col = cast(b.column(0), &ADataType::Utf8).unwrap();
            let idxs_col = cast(b.column(1), &ADataType::UInt64).unwrap();
            let paths = paths_col.as_any().downcast_ref::<StringArray>().unwrap();
            let idxs = idxs_col.as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((paths.value(i).to_string(), idxs.value(i)));
            }
        }
        out
    }

    /// THE GATE for DV-dedup option D: a single scan over MULTIPLE files must
    /// expose a PER-FILE physical row index (restarting at 1 in each file), not a
    /// global running counter. If it were global, driver-side loser positions
    /// would be off by the sum of preceding files' rows and DV-dedup would delete
    /// the wrong rows. Duplicate-key groups span files, so DV-dedup needs one
    /// multi-file scan; delete.rs only ever scans one file at a time and never
    /// exercised this.
    #[tokio::test]
    async fn row_index_is_per_file_across_a_multi_file_scan() -> DeltaResult<()> {
        // Two separate appends => two data files.
        let schema = StructType::try_new(vec![StructField::new(
            "value",
            DataType::Primitive(PrimitiveType::Integer),
            true,
        )])?;
        let table = DeltaTable::new_in_memory()
            .create()
            .with_columns(schema.fields().cloned())
            .with_configuration_property(TableProperty::EnableDeletionVectors, Some("true"))
            .await?;
        let table = table
            .write(vec![values_batch(0..3)])
            .with_save_mode(SaveMode::Append)
            .await?;
        let table = table
            .write(vec![values_batch(100..105)])
            .with_save_mode(SaveMode::Append)
            .await?;

        let adds = all_adds(&table).await;
        assert_eq!(adds.len(), 2, "expected two data files");
        let pairs = scan_file_and_row_index(&table, adds).await;

        // Group indexes by file.
        use std::collections::BTreeMap;
        let mut by_file: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for (path, idx) in pairs {
            by_file.entry(path).or_default().push(idx);
        }
        assert_eq!(
            by_file.len(),
            2,
            "rows must attribute to two distinct files"
        );
        let mut lens: Vec<usize> = by_file.values().map(Vec::len).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 5], "3-row file and 5-row file");
        for (path, mut idxs) in by_file {
            idxs.sort_unstable();
            let n = idxs.len() as u64;
            // 1-based, contiguous, restarting at 1 per file (delete.rs converts
            // this to 0-based DV indexes via `v - 1`). A global counter would make
            // the second file start at 4 (or 6), which this rejects.
            assert_eq!(
                idxs,
                (1..=n).collect::<Vec<_>>(),
                "file {path} row indexes must be per-file 1..={n}, got {idxs:?}"
            );
        }
        Ok(())
    }

    /// A file that already carries a DV must still report PHYSICAL row indexes for
    /// its surviving rows (masked rows simply absent), so a second DV-dedup pass
    /// unions physical positions onto the existing DV rather than shifting them.
    #[tokio::test]
    async fn row_index_stays_physical_on_a_dv_bearing_file() -> DeltaResult<()> {
        let table = make_table().await; // one file, values 0..10
        let table = commit_dv(table, vec![1, 3, 5]).await?; // mask physical rows 1,3,5
        let adds = all_adds(&table).await;
        assert_eq!(adds.len(), 1);
        let pairs = scan_file_and_row_index(&table, adds).await;
        let mut idxs: Vec<u64> = pairs.iter().map(|(_, i)| *i).collect();
        idxs.sort_unstable();
        // 1-based physical indexes of the SURVIVORS: 0,2,4,6,7,8,9 -> 1,3,5,7,8,9,10.
        assert_eq!(
            idxs,
            vec![1, 3, 5, 7, 8, 9, 10],
            "DV-masked scan must expose physical (not compacted) row indexes of survivors"
        );
        Ok(())
    }

    #[tokio::test]
    async fn dv_delete_all_rows_in_file_reads_empty() -> DeltaResult<()> {
        // Masking every row in a file (cardinality == numRecords) must yield an empty
        // read, not an error or resurrected rows.
        let table = make_table().await;
        let table = commit_dv(table, (0..10).collect()).await?;
        let data = get_data_sorted(&table, "value").await;
        assert!(
            sorted_values(&data).is_empty(),
            "fully-masked file should read empty"
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_vacuum_keeps_live_dv_files_and_preserves_deletes() -> DeltaResult<()> {
        use crate::operations::vacuum::{VacuumBuilder, VacuumMode};
        use object_store::ObjectStore as _;

        let table = make_table().await;
        let table = commit_dv(table, vec![1, 3, 5]).await?;

        // Locate the DV file written by the delete.
        let store = table.log_store().object_store(None);
        let list: Vec<_> = store.list(None).try_collect::<Vec<_>>().await.unwrap();
        let dv_files: Vec<_> = list
            .iter()
            .filter(|m| m.location.as_ref().contains("deletion_vector_"))
            .map(|m| m.location.clone())
            .collect();
        assert_eq!(dv_files.len(), 1, "expected one DV file, got {dv_files:?}");

        // Full vacuum with zero retention — the aggressive case that lists the store.
        let (table, result) = VacuumBuilder::new(
            table.log_store(),
            Some(table.snapshot()?.snapshot().clone()),
        )
        .with_retention_period(chrono::Duration::hours(0))
        .with_mode(VacuumMode::Full)
        .with_enforce_retention_duration(false)
        .await?;
        assert!(
            !result
                .files_deleted
                .iter()
                .any(|f| f.contains("deletion_vector_")),
            "vacuum deleted a live DV file: {:?}",
            result.files_deleted
        );

        // The DV file survives and the logically-deleted rows stay hidden.
        assert!(
            store.head(&dv_files[0]).await.is_ok(),
            "live DV file was garbage-collected by full vacuum"
        );
        let data = get_data_sorted(&table, "value").await;
        assert_eq!(sorted_values(&data), vec![0, 2, 4, 6, 7, 8, 9]);
        Ok(())
    }

    #[tokio::test]
    async fn dv_second_delete_merges_with_existing() -> DeltaResult<()> {
        let table = make_table().await;
        let table = commit_dv(table, vec![1, 3, 5]).await?;
        // Second delete: index 0 is new, index 3 already deleted (must not double-count / clobber).
        let table = commit_dv(table, vec![0, 3]).await?;
        let data = get_data_sorted(&table, "value").await;
        assert_eq!(sorted_values(&data), vec![2, 4, 6, 7, 8, 9]);
        // Cardinality on the surviving Add reflects all four logically-deleted rows.
        let add = single_file_add(&table).await;
        assert_eq!(add.deletion_vector.unwrap().cardinality, 4);
        Ok(())
    }
}

/// Build the Remove tombstone matching an existing Add (carrying its current DV, if any).
fn remove_for_add(add: &Add) -> Remove {
    Remove {
        path: add.path.clone(),
        data_change: true,
        deletion_timestamp: Some(chrono::Utc::now().timestamp_millis()),
        extended_file_metadata: Some(true),
        size: Some(add.size),
        partition_values: Some(add.partition_values.clone()),
        deletion_vector: add.deletion_vector.clone(),
        tags: None,
        base_row_id: add.base_row_id,
        default_row_commit_version: add.default_row_commit_version,
    }
}
