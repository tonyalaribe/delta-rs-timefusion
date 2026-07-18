//! Merge-on-read `UPDATE ... FROM` (MERGE with a single `WHEN MATCHED THEN UPDATE`).
//!
//! This is the deletion-vector analogue of [`crate::operations::merge`] restricted to the
//! enrichment shape TimeFusion uses: join the target against a materialized source, and for
//! every matched target row write the updated copy as a new appended file while masking the
//! original row with a deletion vector. Unlike the copy-on-write `MergeBuilder`, it never
//! rewrites whole matched files — the write footprint is bounded by the matched rows, not the
//! files that contain them (the 2026-07-17 enrichment-MERGE OOM hotspot).
//!
//! Matching is per-file (`with_adds([add])`) so a scanned row's file is unambiguous — the DV
//! index space is exactly that file's physical rows. Multiple source rows matching one target
//! row is rejected, mirroring Delta MERGE semantics (it would otherwise duplicate the row).

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::exec_datafusion_err;
use datafusion::datasource::{provider_as_source, MemTable};
use datafusion::logical_expr::{col, Expr, JoinType, LogicalPlanBuilder};
use datafusion::physical_plan::collect;
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;

use crate::kernel::transaction::CommitBuilder;
use crate::kernel::{Action, Add, EagerSnapshot};
use crate::logstore::LogStoreRef;
use crate::operations::delete::DV_ROW_INDEX_COL;
use crate::operations::deletion_vectors::{write_deletion_vectors, FileDeletion};
use crate::operations::write::execution::write_execution_plan;
use crate::operations::write::WriterStatsConfig;
use crate::protocol::DeltaOperation;
use crate::table::config::TablePropertiesExt as _;
use crate::table::state::DeltaTableState;
use crate::{DeltaResult, DeltaTable, DeltaTableError};

/// Inputs for a merge-on-read `UPDATE ... FROM`.
pub struct MergeDvUpdate {
    /// Materialized source rows (the `FROM` side).
    pub source_batches: Vec<RecordBatch>,
    pub source_schema: SchemaRef,
    /// Target-only predicate used to prune candidate files (partition/stat skipping).
    /// Rows are still matched by `join_predicate`; this only bounds which files are scanned.
    pub target_predicate: Option<Expr>,
    /// Full join condition (equi-join keys AND the user predicate), referencing the
    /// `target_alias` / `source_alias` qualifiers.
    pub join_predicate: Expr,
    /// `(target_column, value_expr)` assignments; `value_expr` references the aliases.
    pub updates: Vec<(String, Expr)>,
    pub target_alias: String,
    pub source_alias: String,
    pub writer_properties: Option<WriterProperties>,
}

/// Execute a merge-on-read `UPDATE ... FROM`, committing DV masks + appended rows atomically.
/// Returns the updated table and the number of target rows updated.
pub async fn merge_update_with_deletion_vectors(
    table: &DeltaTable,
    session: &dyn Session,
    op: MergeDvUpdate,
) -> DeltaResult<(DeltaTable, u64)> {
    let log_store = table.log_store();
    let snapshot = table.snapshot()?.snapshot().clone();
    let operation_id = Uuid::new_v4();

    let (actions, num_updated) =
        collect_merge_dv_actions(&log_store, &snapshot, session, &op, operation_id).await?;
    if actions.is_empty() {
        return Ok((
            DeltaTable::new_with_state(log_store, DeltaTableState { snapshot }),
            0,
        ));
    }

    let predicate = op
        .target_predicate
        .as_ref()
        .map(crate::delta_datafusion::expr::fmt_expr_to_sql)
        .transpose()?;
    let commit = CommitBuilder::default()
        .with_actions(actions)
        .with_operation_id(operation_id)
        .build(Some(&snapshot), log_store.clone(), DeltaOperation::Update { predicate })
        .await?;
    Ok((DeltaTable::new_with_state(log_store, commit.snapshot()), num_updated))
}

async fn collect_merge_dv_actions(
    log_store: &LogStoreRef,
    snapshot: &EagerSnapshot,
    session: &dyn Session,
    op: &MergeDvUpdate,
    operation_id: Uuid,
) -> DeltaResult<(Vec<Action>, u64)> {
    // Prune candidate files with the target-only predicate (partition + stats skipping).
    let matched_adds = candidate_adds(log_store, snapshot, session, op.target_predicate.clone()).await?;
    if matched_adds.is_empty() {
        return Ok((vec![], 0));
    }

    let table = DeltaTable::new_with_state(log_store.clone(), DeltaTableState {
        snapshot: snapshot.clone(),
    });
    let partition_cols = snapshot.metadata().partition_columns().to_vec();
    let target_size = Some(snapshot.table_properties().target_file_size());
    let stats_config = WriterStatsConfig::from_config(snapshot.table_configuration());
    let target_schema = snapshot.arrow_schema();

    let source = Arc::new(MemTable::try_new(op.source_schema.clone(), vec![op.source_batches.clone()])?);

    let mut actions = Vec::new();
    let mut deletions = Vec::new();
    let mut num_updated: u64 = 0;

    for add in matched_adds {
        // Per-file scan exposing physical row indexes, joined to the source.
        let provider = table
            .table_provider()
            .with_row_index_column(DV_ROW_INDEX_COL)
            .with_adds([add.clone()])
            .build()
            .await?;
        let target_plan = LogicalPlanBuilder::scan(
            op.target_alias.as_str(),
            provider_as_source(Arc::new(provider)),
            None,
        )?
        .build()?;
        let source_plan = LogicalPlanBuilder::scan(
            op.source_alias.as_str(),
            provider_as_source(source.clone()),
            None,
        )?
        .build()?;

        // Project the physical row index first, then the updated target columns in table order.
        let mut projection = vec![col(format!("{}.{}", op.target_alias, DV_ROW_INDEX_COL))];
        for field in target_schema.fields() {
            let name = field.name();
            let expr = match op.updates.iter().find(|(c, _)| c == name) {
                Some((_, e)) => e.clone().alias(name),
                None => col(format!("{}.{}", op.target_alias, name)),
            };
            projection.push(expr);
        }
        let joined = LogicalPlanBuilder::from(target_plan)
            .join_on(source_plan, JoinType::Inner, [op.join_predicate.clone()])?
            .project(projection)?
            .build()?;

        let exec = session.create_physical_plan(&joined).await?;
        let batches = collect(exec, session.task_ctx()).await?;

        let (indexes, updated_batches) = split_row_index(batches, &target_schema)?;
        if indexes.is_empty() {
            continue;
        }
        num_updated += indexes.len() as u64;

        // Append the updated rows as new files (no DV; data_change).
        if !updated_batches.is_empty() {
            let mem = Arc::new(MemTable::try_new(target_schema.clone(), vec![updated_batches])?);
            let append_plan = LogicalPlanBuilder::scan("updated", provider_as_source(mem), None)?.build()?;
            let append_exec = session.create_physical_plan(&append_plan).await?;
            let mut appended = write_execution_plan(
                Some(snapshot),
                session,
                append_exec,
                partition_cols.clone(),
                log_store.object_store(Some(operation_id)),
                target_size,
                None,
                op.writer_properties.clone(),
                stats_config.clone(),
            )
            .await?;
            actions.append(&mut appended);
        }

        deletions.push(FileDeletion { add, deleted_indexes: indexes });
    }

    let root = snapshot.table_configuration().table_root().clone();
    let mut dv_actions = write_deletion_vectors(log_store.as_ref(), &root, deletions).await?;
    actions.append(&mut dv_actions);
    Ok((actions, num_updated))
}

/// Candidate files to scan: those whose data may match `target_predicate` (whole table if none).
async fn candidate_adds(
    log_store: &LogStoreRef,
    snapshot: &EagerSnapshot,
    session: &dyn Session,
    target_predicate: Option<Expr>,
) -> DeltaResult<Vec<Add>> {
    use futures::TryStreamExt;

    let predicate = match target_predicate {
        Some(p) => p,
        None => {
            return snapshot
                .file_views(log_store.as_ref(), None)
                .map_ok(|f| f.to_add())
                .try_collect()
                .await;
        }
    };
    let Some(files_scan) =
        crate::delta_datafusion::scan_files_where_matches(session, snapshot, log_store.clone(), predicate).await?
    else {
        return Ok(vec![]);
    };
    let valid = Arc::new(files_scan.files_set());
    let root = Arc::new(snapshot.table_configuration().table_root().clone());
    snapshot
        .snapshot()
        .active_adds(log_store.as_ref(), crate::kernel::ActiveAddOptions {
            predicate: Some(files_scan.delta_predicate.clone()),
            stats: crate::kernel::AddStatsPolicy::RawJson,
        })
        .try_filter_map(|f| {
            let (valid, root) = (Arc::clone(&valid), Arc::clone(&root));
            async move {
                let url = root.join(f.path_raw()).map_err(|e| exec_datafusion_err!("{e}"))?;
                Ok(valid.contains(url.as_ref()).then(|| f.to_add()))
            }
        })
        .try_collect()
        .await
}

/// Split the joined output (row-index column first) into 0-based physical indexes and the
/// target-schema batches. Rejects a target row matched by multiple source rows.
fn split_row_index(
    batches: Vec<RecordBatch>,
    target_schema: &SchemaRef,
) -> DeltaResult<(Vec<u64>, Vec<RecordBatch>)> {
    let mut indexes = Vec::new();
    let mut out = Vec::with_capacity(batches.len());
    let mut seen = std::collections::HashSet::new();
    for batch in batches {
        let idx = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DeltaTableError::Generic("row index column is not UInt64".into()))?;
        for v in idx.iter().flatten() {
            // Scan row index is 1-based; DV physical indexes are 0-based.
            let phys = v - 1;
            if !seen.insert(phys) {
                return Err(DeltaTableError::Generic(
                    "MERGE matched a target row against multiple source rows".into(),
                ));
            }
            indexes.push(phys);
        }
        // Drop the leading row-index column to recover the target-schema batch.
        out.push(batch.project(&(1..batch.num_columns()).collect::<Vec<_>>())?);
    }
    // Ensure the appended batches carry exactly the target schema (names, order, types).
    // The scan may emit view types (Utf8View/BinaryView) where the table declares
    // Utf8/Binary, so cast each column to the target field type.
    let out = out
        .into_iter()
        .map(|b| {
            let cols = target_schema
                .fields()
                .iter()
                .zip(b.columns())
                .map(|(f, c)| arrow_cast::cast(c, f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new(target_schema.clone(), cols).map_err(DeltaTableError::from)
        })
        .collect::<DeltaResult<Vec<_>>>()?;
    Ok((indexes, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use crate::delta_datafusion::create_session;
    use crate::kernel::{DataType as DeltaType, PrimitiveType, StructField, StructType};
    use crate::protocol::SaveMode;
    use crate::writer::test_utils::datafusion::get_data_sorted;
    use crate::{DeltaResult, DeltaTable, TableProperty};

    fn target_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("value", DataType::Int32, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ])
        .unwrap()
    }

    fn source_batch() -> (Vec<RecordBatch>, SchemaRef) {
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("sid", DataType::Utf8, true),
            Field::new("newval", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec!["b", "c"])),
            Arc::new(Int32Array::from(vec![20, 30])),
        ])
        .unwrap();
        (vec![batch], schema)
    }

    #[tokio::test]
    async fn merge_dv_update_masks_and_appends_matched_rows() -> DeltaResult<()> {
        let dtype = |p| DeltaType::Primitive(p);
        let schema = StructType::try_new(vec![
            StructField::new("id", dtype(PrimitiveType::String), true),
            StructField::new("value", dtype(PrimitiveType::Integer), true),
        ])
        .unwrap();
        let table = DeltaTable::new_in_memory()
            .create()
            .with_columns(schema.fields().cloned())
            .with_configuration_property(TableProperty::EnableDeletionVectors, Some("true"))
            .await
            .unwrap();
        let table = table.write(vec![target_batch()]).with_save_mode(SaveMode::Append).await.unwrap();
        assert_eq!(table.snapshot()?.log_data().num_files(), 1);

        let session = create_session().into_inner().state();
        table.update_datafusion_session(&session)?;
        let (source_batches, source_schema) = source_batch();

        let (table, updated) = merge_update_with_deletion_vectors(&table, &session, MergeDvUpdate {
            source_batches,
            source_schema,
            target_predicate: None,
            join_predicate: col("target.id").eq(col("source.sid")),
            updates: vec![("value".to_string(), col("source.newval"))],
            target_alias: "target".to_string(),
            source_alias: "source".to_string(),
            writer_properties: None,
        })
        .await?;

        assert_eq!(updated, 2, "two target rows matched");
        // Original file masked (kept) + one appended file with the updated rows.
        assert_eq!(table.snapshot()?.log_data().num_files(), 2);

        // b,c updated to 20,30; a untouched; count unchanged.
        let data = get_data_sorted(&table, "id,value").await;
        let mut pairs = Vec::new();
        for b in &data {
            let ids = arrow_cast::cast(b.column(0), &DataType::Utf8).unwrap();
            let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
            let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..b.num_rows() {
                pairs.push((ids.value(i).to_string(), vals.value(i)));
            }
        }
        pairs.sort();
        assert_eq!(pairs, vec![
            ("a".to_string(), 1),
            ("b".to_string(), 20),
            ("c".to_string(), 30),
        ]);
        Ok(())
    }
}
