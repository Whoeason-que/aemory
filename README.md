# Aemory

[![CI](https://github.com/Whoeason-que/aemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Whoeason-que/aemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Aemory** is a high-performance **Data Compiler for RAG** (Retrieval-Augmented Generation), written in Rust with Python bindings. It efficiently compiles Markdown knowledge bases into vector-searchable [Lance](https://lancedb.com/) datasets.

## Features

- 🚀 **High Performance**: Built on Rust and LanceDB for blazing fast ingestion and retrieval.
- 🧠 **Smart Chunking**: Recursively chunks Markdown by headers to preserve semantic context.
- 🐍 **Python Bindings**: Native integration via PyO3. Use it just like any Python library.
- 🔋 **Zero-Copy**: Leverages Arrow format for efficient data handling.
- 🦀 **Pure Rust**: Uses Candle for embeddings and rustls for TLS - no system dependencies required.
- 🌍 **Universal Platform Support**: Pre-built wheels for 7 platforms (Linux glibc/musl, Windows x64, macOS Intel/Apple Silicon).

## Installation

### From PyPI (Recommended)

```bash
pip install aemory
```

Pre-built wheels are available for:

- **Linux**: x86_64, aarch64 (glibc and musl)
- **Windows**: x64 only
- **macOS**: Intel (x86_64) and Apple Silicon (aarch64)

> **Note**: x86/ARMv7 (Linux) and ARM64 (Windows) are temporarily disabled due to:
> - x86: Upstream compilation issue in lance-core 2.0.1 on i686
> - ARMv7: Upstream compilation issue in lance-core 2.0.1
> - Windows ARM64: GitHub Actions Python installation instability

### Development Build

```bash
# Install maturin for building the python extension
pip install maturin

# Build and install in current environment
maturin develop --release
```

**Note**: Building from source requires the Rust toolchain. The protoc (Protocol Buffers compiler) is automatically downloaded by the build script.

### Release Optimization Profile

This project uses a more aggressive Cargo release profile to maximize runtime performance and reduce wheel size:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
debug = false
incremental = false
```

These flags trade longer compile time for faster runtime and smaller release artifacts.

## Quick Start

### 1. Build a Dataset

Compile your markdown files into a vector index.

```python
import aemory

# Build the dataset
aemory.build(
    input="./knowledge_base", 
    output="./data.lance", 
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 2. Search

Retrieve relevant context for your LLM.

```python
results = aemory.search(
    uri="./data.lance", 
    query="How does the compiler work?", 
    limit=3,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

for r in results:
    print(f"[{r['score']:.4f}] {r['content']}")
```

### 3. Full Example

See [example/test_search.py](example/test_search.py) for a complete working example.

## Supported Models

Aemory uses BERT-based models compatible with Candle. Recommended models:

- `sentence-transformers/all-MiniLM-L6-v2` (lightweight, 80MB)
- `BAAI/bge-small-en-v1.5` (English, 130MB)
- `BAAI/bge-base-en-v1.5` (English, better quality, 440MB)

Models are automatically downloaded from Hugging Face Hub on first use.

## Architecture

Aemory operates as a compiler pipeline:

1. **Loader**: Traverses markdown files and parses frontmatter.
2. **Chunker**: Splits content hierarchically based on headers.
3. **Embedder**: Generates vector embeddings using Candle (pure Rust BERT implementation).
4. **Compiler**: Writes structured data (Content + Vectors + Metadata) to Lance format.

### Technical Stack

- **Embedding**: Candle with BERT models from Hugging Face Hub
- **Vector Store**: Lance (Apache Arrow columnar format)
- **TLS**: rustls (pure Rust, no OpenSSL dependency)
- **Python Bindings**: PyO3

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
