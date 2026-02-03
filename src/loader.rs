use crate::schema::DocumentChunk;
use anyhow::{Context, Result};
use gray_matter::{engine::YAML, Matter};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const CHUNK_SIZE_LIMIT: usize = 1000; // Approximate char limit (proxy for tokens)

pub struct Loader {
    root_dir: PathBuf,
}

impl Loader {
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let matter = Matter::<YAML>::new();

        for entry in WalkDir::new(&self.root_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().extension().map_or(false, |ext| ext == "md") {
                let path = entry.path();
                let content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read file: {:?}", path))?;

                let result = matter.parse(&content);
                let metadata = result
                    .as_ref()
                    .ok()
                    .and_then(|parsed| parsed.data.as_ref())
                    .map(|pod: &gray_matter::Pod| {
                        let json_val: serde_json::Value =
                            pod.deserialize().unwrap_or(serde_json::Value::Null);
                        json_val
                    })
                    .unwrap_or(serde_json::Value::Null);

                if let Ok(parsed) = result {
                    let file_chunks = self.split_markdown(&parsed.content, path, &metadata);
                    chunks.extend(file_chunks);
                }
            }
        }

        Ok(chunks)
    }

    fn split_markdown(
        &self,
        content: &str,
        path: &Path,
        metadata: &serde_json::Value,
    ) -> Vec<DocumentChunk> {
        // Simple MVP splitting strategy: Split by double newlines + headers
        let mut chunks = Vec::new();
        let source_path = path.to_string_lossy().to_string();

        // Very basic splitter for MVP: split by double newlines (paragraphs)
        // In a real implementation, we would want a more sophisticated recursive Markdown parser
        let raw_splits: Vec<&str> = content.split("\n\n").collect();

        let mut current_chunk = String::new();
        let mut current_header: Option<String> = None;

        for split in raw_splits {
            let split_trim = split.trim();
            if split_trim.is_empty() {
                continue;
            }

            // Simple header detection
            if split_trim.starts_with("#") {
                // If we have accumulated content, save it
                if !current_chunk.is_empty() {
                    chunks.push(DocumentChunk::new(
                        current_chunk.clone(),
                        source_path.clone(),
                        current_header.clone(),
                        metadata.clone(),
                        vec![], // Embeddings added later
                    ));
                    current_chunk.clear();
                }
                current_header = Some(split_trim.to_string());
                current_chunk.push_str(split_trim); // Include header in the next chunk context
            } else {
                if current_chunk.len() + split_trim.len() > CHUNK_SIZE_LIMIT {
                    // Chunk is getting full, push it
                    chunks.push(DocumentChunk::new(
                        current_chunk.clone(),
                        source_path.clone(),
                        current_header.clone(),
                        metadata.clone(),
                        vec![],
                    ));
                    current_chunk.clear();
                }

                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(split_trim);
            }
        }

        // Push remaining content
        if !current_chunk.is_empty() {
            chunks.push(DocumentChunk::new(
                current_chunk,
                source_path,
                current_header,
                metadata.clone(),
                vec![],
            ));
        }

        chunks
    }
}
