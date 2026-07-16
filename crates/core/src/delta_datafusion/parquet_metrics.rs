use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use bytes::Bytes;
use datafusion::{
    datasource::physical_plan::parquet::{
        CachedParquetFileReaderFactory, ParquetFileReaderFactory,
    },
    execution::cache::{
        CacheAccessor,
        cache_manager::{CachedFileMetadataEntry, FileMetadataCache, FileMetadataCacheEntry},
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
use std::{ops::Range, time::Instant};

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
}

static METADATA_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static METADATA_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static BYTES_READ: AtomicU64 = AtomicU64::new(0);
static READ_TIME_US: AtomicU64 = AtomicU64::new(0);
static SCANS: AtomicU64 = AtomicU64::new(0);
static FILES_PLANNED: AtomicU64 = AtomicU64::new(0);
static BYTES_PLANNED: AtomicU64 = AtomicU64::new(0);
static SELECTED_ROW_GROUPS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct InstrumentedFileMetadataCache {
    inner: Arc<dyn FileMetadataCache>,
}

impl InstrumentedFileMetadataCache {
    pub fn new(inner: Arc<dyn FileMetadataCache>) -> Self {
        Self { inner }
    }
}

impl CacheAccessor<Path, CachedFileMetadataEntry> for InstrumentedFileMetadataCache {
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
}

#[derive(Debug)]
pub struct InstrumentedParquetFileReaderFactory {
    inner: CachedParquetFileReaderFactory,
}

impl InstrumentedParquetFileReaderFactory {
    pub fn new(store: Arc<dyn ObjectStore>, metadata_cache: Arc<dyn FileMetadataCache>) -> Self {
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
        Ok(Box::new(InstrumentedParquetFileReader {
            inner: self.inner.create_reader(
                partition_index,
                partitioned_file,
                metadata_size_hint,
                metrics,
            )?,
        }))
    }
}

struct InstrumentedParquetFileReader {
    inner: Box<dyn AsyncFileReader + Send>,
}

impl AsyncFileReader for InstrumentedParquetFileReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let started = Instant::now();
        async move {
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
            let result = self.inner.get_byte_ranges(ranges).await;
            if let Ok(bytes) = &result {
                record_read(
                    bytes.iter().map(Bytes::len).sum(),
                    started.elapsed().as_micros() as u64,
                );
            }
            result
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<parquet::file::metadata::ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

impl FileMetadataCache for InstrumentedFileMetadataCache {
    fn cache_limit(&self) -> usize {
        self.inner.cache_limit()
    }

    fn update_cache_limit(&self, limit: usize) {
        self.inner.update_cache_limit(limit);
    }

    fn list_entries(&self) -> HashMap<Path, FileMetadataCacheEntry> {
        self.inner.list_entries()
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
