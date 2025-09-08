use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use arrow::array::{StringArray, BinaryArray, ArrayRef, Array};
use arrow::datatypes::DataType::{Binary, Utf8};
use arrow::datatypes::{Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use chrono::{Duration, Utc};
use clap::Parser;
use parquet::arrow::{arrow_reader::ArrowReaderBuilder, arrow_reader::ArrowReaderOptions, async_writer::AsyncArrowWriter, async_writer::ParquetObjectWriter};
use parquet::basic::{Encoding, Compression, ZstdLevel};
use parquet::errors::Result;
use parquet::file::properties::{WriterProperties, EnabledStatistics};
use parquet::file::metadata::KeyValue;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::file::metadata::PageIndexPolicy;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::path::Path;
use std::sync::Arc;

use sysinfo::{Pid, System};

const DATA_COL_SIZE_MD: &str = "data_col_duration_ms";
const MINT_MD: &str        = "minT";
const MAXT_MD: &str        = "maxT";


#[derive(Parser)]
#[command(version)]
/// Writes or reads Parquet files with synthetic series data, while logging timing and memory usage.
struct Args {
    #[arg(long, default_value_t = 1000)]
    /// Number of batches to write
    iterations: u64,

    #[arg(long, default_value_t = 1000000)]
    /// Number of rows in each batch
    batch: u64,

    #[arg(long)]
    /// Read the file instead of writing
    read: bool,

    /// Path to the file to write or read
    path: PathBuf,
}

pub struct BlockWriter {
    writer: AsyncArrowWriter<ParquetObjectWriter>,
    schema: SchemaRef,
    write_batch_size: usize,
    map_col_name_to_idx: HashMap<String, usize>,
    series_count: usize,
    data_column_indexes: Vec<usize>,
    dimension_values: Vec<Vec<Option<String>>>,
    column_indexes_values: Vec<Vec<u8>>,
    data_column_values: Vec<Vec<Option<Vec<u8>>>>,
}

pub struct Label {
    name: String,
    value: String,
}

pub struct Series {
    labels: Vec<Label>,
}

impl BlockWriter {
    pub async fn add_series(&mut self, series: Series) -> Result<()> {
        // Collect column indexes for labels that exist
        let mut col_indexes = Vec::new();
        let mut label_map = HashMap::new();
        
        // Build a map of label names to values for quick lookup
        for label in series.labels {
            label_map.insert(label.name, label.value);
        }
        
        // Iterate over schema fields and add values to each array
        for (field_idx, field) in self.schema.fields().iter().enumerate() {
            let field_name = field.name();
            
            if field_name == "s_col_indexes" {
                // Skip the column indexes field for now, we'll handle it separately
                continue;
            }
            if field_name.starts_with("s_data_") {
                // Skip the data columns for now, we'll handle them separately
                continue;
            }
            
            if let Some(value) = label_map.get(field_name) {
                self.dimension_values[field_idx].push(Some(value.clone()));
                col_indexes.push(field_idx as u64);
            } else {
                self.dimension_values[field_idx].push(None);
            }
        }

        for i in 0..self.data_column_indexes.len() {
            self.data_column_values[i].push(None);
        }
        
        // Handle the column indexes field
        self.column_indexes_values.push(encode_int_slice(&col_indexes));
        self.series_count += 1;
        
        // Check if we need to write a batch
        if self.series_count >= self.write_batch_size {
            self.write_current_batch().await?;
        }
        
        Ok(())
    }
    
    async fn write_current_batch(&mut self) -> Result<()> {
        if self.series_count == 0 {
            return Ok(());
        }

        let mut arrays = Vec::new();
        for (field_idx, field) in self.schema.fields().iter().enumerate() {
            match field.data_type() {
                Utf8 => {
                    arrays.push(Arc::new(StringArray::from(self.dimension_values[field_idx].clone())) as ArrayRef);
                }
                _ => {
                    // For other types, we'd need to handle them appropriately
                    // For now, assuming we only have Utf8 and Binary
                }
            }
        }
        let column_indexes_refs: Vec<&[u8]> = self.column_indexes_values.iter().map(|v| v.as_slice()).collect();
        arrays.push(Arc::new(BinaryArray::from(column_indexes_refs)) as ArrayRef);

        for i in 0..self.data_column_indexes.len() {
            let data_column_refs: Vec<&[u8]> = self.data_column_values[i].iter().map(|v| v.as_ref().map(|v| v.as_slice()).unwrap_or(&[])).collect();
            arrays.push(Arc::new(BinaryArray::from(data_column_refs)) as ArrayRef);
        }
        
        let batch = RecordBatch::try_new(self.schema.clone(), arrays.clone())?;
        self.writer.write(&batch).await?;
        
        // Reset buffers for next batch
        self.reset_buffer();
        self.series_count = 0;
        
        Ok(())
    }
    
    fn reset_buffer(&mut self) {
        // Clear all dimension value vectors
        for dimension_values in &mut self.dimension_values {
            dimension_values.clear();
        }
        
        // Clear column indexes
        self.column_indexes_values.clear();

        for i in 0..self.data_column_indexes.len() {
            self.data_column_values[i].clear();
        }
    }

    pub async fn close(mut self) -> Result<()> {
        // Write any remaining series as a final batch
        self.write_current_batch().await?;
        self.writer.close().await?;
        Ok(())
    }
}

pub struct BlockWriterBuilder {
    dimensions: Vec<String>,
    max_row_group_size: u64,
    write_batch_size: u64,
    key_value_metadata: Vec<KeyValue>,
    sorting_columns: Vec<String>,
    chunk_column_duration: Duration,
    mint: u64,
    maxt: u64,
}

impl BlockWriterBuilder {
    pub fn new(mint: u64, maxt: u64) -> Self {
        Self { 
            dimensions: Vec::new(),
            max_row_group_size: 1024 * 1024,
            write_batch_size: 1024,
            key_value_metadata: Vec::new(),
            sorting_columns: Vec::new(),
            chunk_column_duration: Duration::try_hours(8).unwrap(),
            mint: mint,
            maxt: maxt,
        }
    }

    pub fn with_dimensions(mut self, dimensions: Vec<String>) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_max_row_group_size(mut self, max_row_group_size: u64) -> Self {
        self.max_row_group_size = max_row_group_size;
        self
    }

    pub fn with_write_batch_size(mut self, write_batch_size: u64) -> Self {
        self.write_batch_size = write_batch_size;
        self
    }

    pub fn with_key_value_metadata(mut self, key_value_metadata: Vec<KeyValue>) -> Self {
        self.key_value_metadata = key_value_metadata;
        self
    }

    pub fn with_sorting_columns(mut self, sorting_columns: Vec<String>) -> Self {
        self.sorting_columns = sorting_columns;
        self
    }

    pub fn with_chunk_column_duration(mut self, chunk_column_duration: Duration) -> Self {
        self.chunk_column_duration = chunk_column_duration;
        self
    }

    /// Build a BlockWriter with a local file system store
    pub fn build_local(self, file_path: PathBuf) -> Result<BlockWriter> {
        let store = Arc::new(LocalFileSystem::new());
        let path = Path::from(file_path.to_string_lossy().as_ref());
        self.build(store, path)
    }

    /// Build a BlockWriter with an in-memory store
    pub fn build_inmemory(self, path: &str) -> Result<BlockWriter> {
        let store = Arc::new(InMemory::new());
        let path = Path::from(path);
        self.build(store, path)
    }

    /// Build a BlockWriter with an S3 store
    pub fn build_s3(
        self, 
        bucket: &str, 
        path: &str,
        region: Option<&str>,
    ) -> Result<BlockWriter> {
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(bucket);
        
        if let Some(region) = region {
            builder = builder.with_region(region);
        }
        
        let store = Arc::new(builder.build()?);
        let path = Path::from(path);
        self.build(store, path)
    }

    /// Build a BlockWriter with a custom store and path
    pub fn build(self, store: Arc<dyn object_store::ObjectStore>, path: Path) -> Result<BlockWriter> {
        let mut properties_builder = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_max_row_group_size(self.max_row_group_size as usize)
            .set_write_batch_size(self.write_batch_size as usize)
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_write_page_header_statistics(true);

        let mut key_value_metadata = vec![
            KeyValue::new(MINT_MD.to_string(), self.mint.to_string()), 
            KeyValue::new(MAXT_MD.to_string(), self.maxt.to_string()),
            KeyValue::new(DATA_COL_SIZE_MD.to_string(), self.chunk_column_duration.num_seconds().to_string()),
        ];
        key_value_metadata.extend(self.key_value_metadata);
        properties_builder = properties_builder.set_key_value_metadata(Some(key_value_metadata));

        let mut fields = Vec::new();
        // Generate data for all dimensions
        let mut dimension_values: Vec<Vec<Option<String>>> = Vec::new();

        // Initialize vectors for each dimension
        for _ in 0..self.dimensions.len() {
            dimension_values.push(Vec::new());
        }

        for dimension in &self.dimensions {
            fields.push(Field::new(dimension.clone(), Utf8, true));

            properties_builder = properties_builder
                .set_column_dictionary_enabled(dimension.clone().into(), true)
        }

        // Add the column indexes field
        fields.push(Field::new("s_col_indexes", Binary, false));
        properties_builder = properties_builder
            .set_column_encoding("s_col_indexes".into(), Encoding::DELTA_BYTE_ARRAY)
            .set_column_statistics_enabled("s_col_indexes".into(), EnabledStatistics::None);

        let mut col_idx: usize = self.dimensions.len() + 1;
        let mut data_column_indexes = Vec::new();
        let mut data_column_idx = 0;
        for i in (self.mint..=self.maxt).step_by(self.chunk_column_duration.num_milliseconds() as usize) {
            let col_name = format!("s_data_{}", data_column_idx);
            fields.push(Field::new(col_name.clone(), Binary, true));
            properties_builder = properties_builder
                .set_column_encoding(col_name.clone().into(), Encoding::DELTA_BYTE_ARRAY)
                .set_column_statistics_enabled(col_name.into(), EnabledStatistics::None);
            data_column_indexes.push(col_idx);
            col_idx += 1;
            data_column_idx += 1;
        }
        let mut data_column_values: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
        for i in 0..data_column_indexes.len() {
            data_column_values.push(Vec::new());
        }
        
        
        let schema = Arc::new(Schema::new(fields));
        
        if !self.sorting_columns.is_empty() {
            let mut sorting_columns = Vec::new();
            for sorting_column_name in &self.sorting_columns {
                if let Some((col_idx, _)) = schema.fields().iter().enumerate().find(|(_, field)| field.name() == sorting_column_name) {
                    sorting_columns.push(
                        parquet::format::SortingColumn::new(col_idx as i32, false, true)
                    );
                }
            }
            properties_builder = properties_builder.set_sorting_columns(Some(sorting_columns));
        }

        let mut map_col_name_to_idx = HashMap::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            map_col_name_to_idx.insert(field.name().to_string(), idx);
        }

        let properties = properties_builder.build();
        
        // Create ParquetObjectWriter with the provided store and path
        let object_writer = ParquetObjectWriter::new(store, path);
        let writer = AsyncArrowWriter::try_new(object_writer, schema.clone(), Some(properties))?;
        
        Ok(BlockWriter { 
            writer: writer,
            schema: schema,
            write_batch_size: self.write_batch_size as usize,
            data_column_indexes: data_column_indexes,
            map_col_name_to_idx: map_col_name_to_idx,
            series_count: 0,
            dimension_values: dimension_values,
            data_column_values: data_column_values,
            column_indexes_values: Vec::new(),
        })
    }
}

enum MatcherType {
    Equal,
    NotEqual,
    Regex,
    NotRegex,
}

struct Matcher {
    matcher_type: MatcherType,
    name: String,
    value: String,
}

impl Matcher {
    fn new(matcher_type: MatcherType, name: String, value: String) -> Self {
        Self { matcher_type, name, value }
    }

    fn matches(&self, value: &String) -> bool {
        match self.matcher_type {
            MatcherType::Equal => value == &self.value,
            MatcherType::NotEqual => value != &self.value,
            MatcherType::Regex => {
                // Simple regex matching - for now just do string contains
                value.contains(&self.value)
            }
            MatcherType::NotRegex => {
                // Simple regex matching - for now just do string not contains
                !value.contains(&self.value)
            }
        }
    }
}

fn encode_int_slice(slice: &[u64]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut sorted = slice.to_vec();
    sorted.sort();
    let mut len = sorted.len() as u64;
    while len >= 0x80 {
        result.push((len & 0x7F) as u8 | 0x80);
        len >>= 7;
    }
    result.push(len as u8);
    for &value in &sorted {
        let mut val = value;
        while val >= 0x80 {
            result.push((val & 0x7F) as u8 | 0x80);
            val >>= 7;
        }
        result.push(val as u8);
    }
    result
}

fn decode_int_slice(data: &[u8]) -> Result<Vec<u64>> {
    let mut result = Vec::new();
    let mut pos = 0;
    if pos >= data.len() {
        return Ok(result);
    }
    let mut len = 0u64;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            return Err(parquet::errors::ParquetError::General("Unexpected end of data while decoding length".to_string()));
        }
        let byte = data[pos];
        pos += 1;
        len |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
    }
    for _ in 0..len {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            if pos >= data.len() {
                return Err(parquet::errors::ParquetError::General("Unexpected end of data while decoding value".to_string()));
            }
            let byte = data[pos];
            pos += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
        }
        result.push(value);
    }
    Ok(result)
}

fn search(_file_path: &PathBuf) -> Result<()> {
    // TODO: Implement search functionality
    Ok(())
}

fn read_parquet_file(file_path: &PathBuf) -> Result<()> {
    eprintln!("Reading Parquet file: {}", file_path.display());
    
    // Open the file for reading
    let file = File::open(file_path)?;
    
    // First, read only the column indexes column to get the schema and column indexes
    let builder = ArrowReaderBuilder::try_new(file)?;
    let parquet_schema = builder.parquet_schema();
    
    // Create projection mask for only the column indexes column
    let col_indexes_mask = ProjectionMask::columns(&parquet_schema, ["s_col_indexes"]);
    let reader_options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional);
    let col_indexes_reader = {
        let builder_clone = ArrowReaderBuilder::try_new_with_options(File::open(file_path)?, reader_options.clone())?;
        builder_clone
        .with_projection(col_indexes_mask)
        .build()?
    };
    
    eprintln!("Reading column indexes first...");
    let mut all_column_indexes = Vec::new();
    let mut total_rows = 0;
    
    // Read only the column indexes column
    for maybe_batch in col_indexes_reader {
        let batch = maybe_batch?;
        total_rows += batch.num_rows();
        
        if let Some(col_indexes_col) = batch.column_by_name("s_col_indexes") {
            if let Some(binary_array) = col_indexes_col.as_any().downcast_ref::<BinaryArray>() {
                for row_idx in 0..batch.num_rows() {
                    if binary_array.is_valid(row_idx) {
                        let binary_data = binary_array.value(row_idx);
                        match decode_int_slice(binary_data) {
                            Ok(indexes) => {
                                all_column_indexes.push(indexes);
                            }
                            Err(e) => {
                                eprintln!("Error decoding column indexes for row {}: {}", total_rows - batch.num_rows() + row_idx, e);
                                all_column_indexes.push(Vec::new());
                            }
                        }
                    } else {
                        all_column_indexes.push(Vec::new());
                    }
                }
            }
        }
    }
    
    eprintln!("Total rows: {}", total_rows);
    eprintln!("Column indexes decoded: {}", all_column_indexes.len());
    
    // Build column mappings: which columns are needed for which row selections
    let row_selection = RowSelection::from(vec![parquet::arrow::arrow_reader::RowSelector::select(total_rows)]);
    let column_to_row_selection = build_column_mappings(row_selection, &all_column_indexes);
    
    eprintln!("Column mappings built:");
    for (col_idx, row_selection) in &column_to_row_selection {
        eprintln!("  Column {}: needed for {} rows", col_idx, row_selection.row_count());
    }
    
    // Get the full schema to know what columns are available
    let file_for_schema = File::open(file_path)?;
    let schema_builder = ArrowReaderBuilder::try_new(file_for_schema)?;
    let full_schema = schema_builder.schema();
    
    eprintln!("Full schema: {}", full_schema);
    
    // Now read each column with its associated row selections
    for (col_idx, row_selection) in &column_to_row_selection {
        if *col_idx >= full_schema.fields().len() {
            eprintln!("Skipping invalid column index: {}", col_idx);
            continue;
        }
        
        let field = &full_schema.fields()[*col_idx];
        let col_name = field.name();
        eprintln!("Reading column '{}' (index {}) for {} selected rows", col_name, col_idx, row_selection.row_count());
        
        // Create projection mask for this specific column
        let col_mask = ProjectionMask::columns(&parquet_schema, [col_name.as_str()]);
        
        // Open a new file handle for reading this column
        let file_for_col = File::open(file_path)?;
        let col_builder = ArrowReaderBuilder::try_new_with_options(file_for_col, reader_options.clone())?;
        
        // Apply row selection to only read the rows we need
        let col_reader = col_builder
            .with_projection(col_mask)
            .with_row_selection(row_selection.clone())
            .build()?;
        
        // Read the column data
        let mut current_row = 0;
        for (batch_idx, maybe_batch) in col_reader.enumerate() {
            let batch = maybe_batch?;
            eprintln!("  Batch {}: {} rows", batch_idx, batch.num_rows());
            
            // Process rows in this batch
            for row_idx in 0..batch.num_rows() {
                let global_row_idx = current_row + row_idx;
                
                if global_row_idx < all_column_indexes.len() {
                    let column_indexes = &all_column_indexes[global_row_idx];
                    
                    // Check if this row actually needs this column
                    if column_indexes.contains(&(*col_idx as u64)) {
                        if let Some(array) = batch.column_by_name(col_name) {
                            if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                                if string_array.is_valid(row_idx) {
                                    eprintln!("    Row {}: {} = {}", global_row_idx, col_name, string_array.value(row_idx));
                                } else {
                                    eprintln!("    Row {}: {} = <null>", global_row_idx, col_name);
                                }
                            }
                        }
                    }
                }
            }
            current_row += batch.num_rows();
        }
    }
    
    Ok(())
}

// build_column_mappings creates mappings between columns and row selections based on column indexes
fn build_column_mappings(_rs: RowSelection, column_indexes: &[Vec<u64>]) -> HashMap<usize, RowSelection> {
    let mut column_to_row_selection: HashMap<usize, RowSelection> = HashMap::new();
    
    // For each column, collect all row indices that need it
    for (_row_idx, column_indexes) in column_indexes.iter().enumerate() {
        for &col_idx in column_indexes {
            // Create a RowSelection for this specific row
            let row_selector = parquet::arrow::arrow_reader::RowSelector::select(1);
            let row_selection = RowSelection::from(vec![row_selector]);
            
            column_to_row_selection
                .entry(col_idx as usize)
                .or_insert_with(|| RowSelection::from(vec![]))
                .union(&row_selection);
        }
    }
    
    column_to_row_selection
}

fn get_label_value<'a>(series: &'a Series, label_name: &str) -> &'a str {
    series.labels.iter()
        .find(|label| label.name == label_name)
        .map(|label| label.value.as_str())
        .unwrap_or("")
}

fn sort_series_by_columns(series: &mut [Series], sorting_columns: &[&str]) {
    series.sort_by(|a, b| {
        for &column in sorting_columns {
            let a_value = get_label_value(a, column);
            let b_value = get_label_value(b, column);
            let cmp_result = a_value.cmp(b_value);
            if cmp_result != std::cmp::Ordering::Equal {
                return cmp_result;
            }
        }
        std::cmp::Ordering::Equal
    });
}

fn create_series(dimensions: &[&str], row: u64) -> Series {
    let mut labels = Vec::new();
    
    for dimension in dimensions {
        let value = match *dimension {
            "__name__" => Some(format!("http_requests_total_{}", row % 10)),
            "job" => Some(format!("node-exporter-{}", row % 5)),
            "instance" => Some(format!("localhost:{}", 9090 + (row % 10))),
            "cluster" => Some(format!("prod-cluster-{}", row % 3)),
            "namespace" => {
                if row % 3 == 0 { Some(format!("kube-system-{}", row % 2)) } else { None }
            }
            "pod" => {
                if row % 4 == 0 { Some(format!("pod-{}", row % 100)) } else { None }
            }
            "container" => {
                if row % 5 == 0 { Some(format!("container-{}", row % 50)) } else { None }
            }
            _ => Some(format!("value-{}", row)),
        };
        
        if let Some(val) = value {
            labels.push(Label {
                name: dimension.to_string(),
                value: val,
            });
        }
    }
    
    Series { labels }
}

async fn write_parquet(args: &Args) -> Result<()> {
    let start = Instant::now();
    let mut system = System::new();
    
    // Define dimensions dynamically
    let dimensions = vec!["__name__", "job", "instance", "cluster", "namespace", "pod", "container"];
    let sorting_columns = vec!["__name__"];
    
    // Get UTC time for beginning of current day and next day
    let now = Utc::now();
    let beginning_of_today = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
    let beginning_of_tomorrow = beginning_of_today + Duration::days(1);
    
    // Convert to milliseconds since Unix epoch
    let mint = beginning_of_today.and_utc().timestamp_millis() as u64;
    let maxt = beginning_of_tomorrow.and_utc().timestamp_millis() as u64 - 1;

    // Create BlockWriter using the builder with local file system
    let mut block_writer = BlockWriterBuilder::new(mint, maxt)
        .with_dimensions(dimensions.iter().map(|s| s.to_string()).collect())
        .with_sorting_columns(sorting_columns.iter().map(|s| s.to_string()).collect())
        .build_local(args.path.clone())?;
    
            // Generate all series for this iteration
    let mut all_series = Vec::new();
    for iteration in 0..args.iterations {
        if iteration % 100 == 0 {
            system.refresh_all();
            if let Some(process) = system.process(Pid::from_u32(std::process::id())) {
                let memory_usage = process.memory();
                eprintln!("Iteration {}: RSS: {} KB", iteration, memory_usage / 1024);
            }
        }
        

        for row in 0..args.batch {
            let series = create_series(&dimensions, row);
            all_series.push(series);
        }
    }

            
        // Sort series by the specified sorting columns
        sort_series_by_columns(&mut all_series, &sorting_columns);
        
        // Add sorted series to the writer
        for series in all_series {
            block_writer.add_series(series).await?;
        }
    
    block_writer.close().await?;
    
    let duration = start.elapsed();
    eprintln!("Writing completed in {:?}", duration);
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    if args.read {
        return read_parquet_file(&args.path);
    }
    
    // Write mode
    write_parquet(&args).await
}
