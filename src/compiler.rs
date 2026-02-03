use crate::schema::DocumentChunk;
use anyhow::{Context, Result};
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;

pub struct Compiler {
    output_path: String,
}

impl Compiler {
    pub fn new(output_path: &str) -> Self {
        Self {
            output_path: output_path.to_string(),
        }
    }

    pub async fn compile(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        if chunks.is_empty() {
            println!("No chunks to compile.");
            return Ok(());
        }

        // infer dimension from the first chunk
        let dim = chunks[0].vector.len();

        // Convert to Arrow RecordBatch
        // Note: In a real large-scale scenario, we should stream batches instead of loading all into memory.
        let batch = DocumentChunk::to_arrow_batch(&chunks, dim)
            .context("Failed to convert chunks to Arrow RecordBatch")?;

        let reader = arrow_array::RecordBatchIterator::new(
            vec![Ok(batch)],
            DocumentChunk::arrow_schema(dim),
        );

        let params = WriteParams {
            mode: WriteMode::Overwrite, // or Append, configurable in real app
            ..Default::default()
        };

        Dataset::write(reader, &self.output_path, Some(params))
            .await
            .context("Failed to write Lance dataset")?;

        println!(
            "Successfully compiled {} chunks to {}",
            chunks.len(),
            self.output_path
        );

        Ok(())
    }
}
