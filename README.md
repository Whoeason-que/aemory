# Aemory

[![CI](https://github.com/Whoeason-que/aemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Whoeason-que/aemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Aemory** is a high-performance **Data Compiler for RAG** (Retrieval-Augmented Generation), written in Rust with Python bindings. It efficiently compiles Markdown knowledge bases into vector-searchable [Lance](https://lancedb.com/) datasets.

## Features

- 🚀 **High Performance**: Built on Rust and LanceDB for blazing fast ingestion and retrieval.
- 🧠 **Smart Chunking**: Recursively chunks Markdown by headers to preserve semantic context.
- 🐍 **Python Bindings**: Native integration via PyO3. Use it just like any Python library.
- 🔋 **Zero-Copy**: Leverages Arrow format for efficient data handling.

## Installation

### Prerequisites
- Rust Toolchain (stable)
- Python 3.8+

### Development Build
```bash
# Install maturin for building the python extension
pip install maturin

# Build and install in current environment
maturin develop --release
```

## Quick Start

### 1. Build a Dataset
Compile your markdown files into a vector index.

```python
import aemory

# Build the dataset
aemory.build(
    input="./knowledge_base", 
    output="./data.lance", 
    model="BAAI/bge-small-en-v1.5"
)
```

### 2. Search
Retrieve relevant context for your LLM.

```python
results = aemory.search(
    uri="./data.lance", 
    query="How does the compiler work?", 
    limit=3,
    model="BAAI/bge-small-en-v1.5"
)

for r in results:
    print(f"[{r['score']:.4f}] {r['content']}")
```

## Architecture

Aemory operates as a compiler:
1.  **Loader**: Traverses markdown files and parses frontmatter.
2.  **Chunker**: Splits content hierarchically based on headers.
3.  **Embedder**: Generates vector embeddings using FastEmbed (ONNX).
4.  **Compiler**: Writes structured data (Content + Vectors + Metadata) to LanceDB.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
