pub mod compiler;
pub mod embedder;
pub mod loader;
pub mod retriever;
pub mod schema;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::{prelude::*, types::PyDict, Py, types::PyAny};
use std::path::PathBuf;

/// Build the dataset (Python Wrapper)
#[pyfunction]
fn build(input: String, output: String, model: String) -> PyResult<()> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        build_async(input, output, model).await.unwrap();
    });
    Ok(())
}

/// Core async build logic (Shared with CLI)
pub async fn build_async(input: String, output: String, model: String) -> Result<()> {
    println!("🚀 Starting Aemory Build Pipeline (Python Bindings)...");

    // 1. Load Documents
    let input_path = PathBuf::from(input);
    let loader = loader::Loader::new(&input_path);
    let mut chunks = loader.load()?;
    println!("✅ Loaded {} chunks", chunks.len());

    if chunks.is_empty() {
        println!("No markdown files found or empty content.");
        return Ok(());
    }

    // 2. Embed Documents
    println!("🧠 Initializing Embedding Model: {}", model);
    let mut embedder = embedder::Embedder::new(&model)?;

    // TODO: Improve progress bar for Python context if needed, otherwise standard stdout
    let pb = ProgressBar::new(chunks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("Generating Embeddings...");

    // Batch processing for embeddings
    let batch_size = 64;
    let mut chunk_index = 0;

    while chunk_index < chunks.len() {
        let end = std::cmp::min(chunk_index + batch_size, chunks.len());
        let batch_slice = &chunks[chunk_index..end];
        let texts: Vec<String> = batch_slice.iter().map(|c| c.content.clone()).collect();

        // Handle error inside loop
        let embeddings = embedder.generate_embeddings(&texts)?;

        for (i, embedding) in embeddings.into_iter().enumerate() {
            chunks[chunk_index + i].vector = embedding;
        }

        chunk_index = end;
        pb.inc((end - chunk_index + batch_size) as u64);
    }
    pb.finish_with_message("✅ Embeddings generation complete");

    // 3. Compass / Write to Lance
    println!("💾 Writing to Lance dataset at {}", output);
    let compiler = compiler::Compiler::new(&output);
    compiler.compile(chunks).await?;

    println!("✨ Pipeline finished successfully!");
    Ok(())
}

/// Search the dataset (Python Wrapper)
#[pyfunction]
fn search(
    py: Python,
    uri: String,
    query: String,
    limit: usize,
    model: String,
) -> PyResult<Vec<Py<PyAny>>> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let results = rt.block_on(async {
        let mut retriever = retriever::Retriever::new(&uri, &model)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        retriever
            .search(&query, limit)
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })?;

    // Convert Vec<SearchResult> to Python-friendly Vec<Dict>
    let py_results = results
        .into_iter()
        .map(|r| {
            let dict = PyDict::new(py);
            dict.set_item("content", r.content).unwrap();
            dict.set_item("source_path", r.source_path).unwrap();
            dict.set_item("metadata", r.metadata).unwrap();
            dict.set_item("score", r.score).unwrap();
            dict.unbind().into_any()
        })
        .collect();

    Ok(py_results)
}

/// A Python module implemented in Rust.
#[pymodule]
fn aemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    Ok(())
}
