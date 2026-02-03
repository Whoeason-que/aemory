use crate::embedder::Embedder;
use anyhow::{Context, Result};
use futures::TryStreamExt;
use lance::Dataset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub content: String,
    pub source_path: String,
    pub metadata: String,
    pub score: f32, // Distance (lower is better for L2/Cosine on Lance, usually)
}

pub struct Retriever {
    dataset_uri: String,
    embedder: Embedder,
}

impl Retriever {
    pub fn new(dataset_uri: &str, model_name: &str) -> Result<Self> {
        let embedder = Embedder::new(model_name)?;
        Ok(Self {
            dataset_uri: dataset_uri.to_string(),
            embedder,
        })
    }

    pub async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // 1. Embed Query
        // Note: We use the same embedder logic, assuming single query
        let query_embedding = self.embedder.generate_embeddings(&[query.to_string()])?;
        let query_vec = query_embedding
            .first()
            .context("Failed to generate embedding for query")?;

        // 2. Open Dataset
        let dataset = Dataset::open(&self.dataset_uri).await?;

        // 3. Perform Vector Search
        // Lance search returns a RecordBatch stream
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", &Float32Array::from(query_vec.clone()), limit)?
            .project(&["content", "source_path", "metadata", "_distance"])?;

        let stream = scanner.try_into_stream().await?;
        let results: Vec<arrow_array::RecordBatch> = stream.try_collect().await?;

        // 4. Parse Results
        let mut search_results = Vec::new();
        for batch in results {
            let content_col = batch
                .column_by_name("content")
                .context("Missing content column")?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("Content column is not StringArray")?;

            let source_path_col = batch
                .column_by_name("source_path")
                .context("Missing source_path column")?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("Source path column is not StringArray")?;

            let metadata_col = batch
                .column_by_name("metadata")
                .context("Missing metadata column")?
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .context("Metadata column is not StringArray")?;

            let distance_col = batch
                .column_by_name("_distance")
                .context("Missing _distance column")?
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .context("Distance column is not Float32Array")?;

            for i in 0..batch.num_rows() {
                search_results.push(SearchResult {
                    content: content_col.value(i).to_string(),
                    source_path: source_path_col.value(i).to_string(),
                    metadata: metadata_col.value(i).to_string(),
                    score: distance_col.value(i),
                });
            }
        }

        Ok(search_results)
    }
}

use arrow_array::Float32Array;
