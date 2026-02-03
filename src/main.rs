// Modules are now defined in lib.rs
// To use them here, we refer to the library crate `aemory`
use aemory::{compiler, embedder, loader};

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a Lance dataset from a directory of Markdown files
    Build {
        /// Input directory containing Markdown files
        #[arg(long)]
        input: PathBuf,

        /// Output path for the Lance dataset
        #[arg(short, long)]
        output: String,

        /// Embedding model to use (default: BAAI/bge-small-en-v1.5)
        #[arg(short, long, default_value = "BAAI/bge-small-en-v1.5")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Build {
            input,
            output,
            model,
        } => {
            println!("🚀 Starting Aemory Build Pipeline...");

            // 1. Load Documents
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap(),
            );
            spinner.set_message(format!("Loading markdown from {:?}", input));

            let loader = loader::Loader::new(input);
            let mut chunks = loader.load()?;
            spinner.finish_with_message(format!("✅ Loaded {} chunks", chunks.len()));

            if chunks.is_empty() {
                println!("No markdown files found or empty content. Exiting.");
                return Ok(());
            }

            // 2. Embed Documents
            println!("🧠 Initializing Embedding Model: {}", model);
            let mut embedder = embedder::Embedder::new(model)?;

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
            let compiler = compiler::Compiler::new(output);
            compiler.compile(chunks).await?;

            println!("✨ Pipeline finished successfully!");
        }
    }

    Ok(())
}
