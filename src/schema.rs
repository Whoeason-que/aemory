use anyhow::Result;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub vector: Vec<f32>,
    pub content: String,
    pub source_path: String,
    pub parent_header: Option<String>,
    pub metadata: String, // JSON string
    pub created_at: i64,
}

impl DocumentChunk {
    pub fn new(
        content: String,
        source_path: String,
        parent_header: Option<String>,
        metadata: serde_json::Value,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            vector,
            content,
            source_path,
            parent_header,
            metadata: metadata.to_string(),
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    pub fn arrow_schema(dim: usize) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("source_path", DataType::Utf8, false),
            Field::new("parent_header", DataType::Utf8, true),
            Field::new("metadata", DataType::Utf8, false), // JSON stored as string
            Field::new("created_at", DataType::Int64, false),
        ]))
    }

    pub fn to_arrow_batch(chunks: &[DocumentChunk], dim: usize) -> Result<RecordBatch> {
        if chunks.is_empty() {
            return Ok(RecordBatch::new_empty(Self::arrow_schema(dim)));
        }

        let ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();
        let contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let source_paths: Vec<String> = chunks.iter().map(|c| c.source_path.clone()).collect();
        let parent_headers: Vec<Option<String>> =
            chunks.iter().map(|c| c.parent_header.clone()).collect();
        let metadatas: Vec<String> = chunks.iter().map(|c| c.metadata.clone()).collect();
        let created_ats: Vec<i64> = chunks.iter().map(|c| c.created_at).collect();

        // Flatten vectors for FixedSizeList
        let vectors_flat: Vec<f32> = chunks.iter().flat_map(|c| c.vector.clone()).collect();

        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;

        // Construct FixedSizeListArray for vectors
        let vector_values = Float32Array::from(vectors_flat);
        let vector_array = Arc::new(FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(vector_values),
            None,
        )?) as ArrayRef;

        let content_array = Arc::new(StringArray::from(contents)) as ArrayRef;
        let source_path_array = Arc::new(StringArray::from(source_paths)) as ArrayRef;
        let parent_header_array = Arc::new(StringArray::from(parent_headers)) as ArrayRef;
        let metadata_array = Arc::new(StringArray::from(metadatas)) as ArrayRef;
        let created_at_array = Arc::new(Int64Array::from(created_ats)) as ArrayRef;

        let batch = RecordBatch::try_new(
            Self::arrow_schema(dim),
            vec![
                id_array,
                vector_array,
                content_array,
                source_path_array,
                parent_header_array,
                metadata_array,
                created_at_array,
            ],
        )?;

        Ok(batch)
    }
}
