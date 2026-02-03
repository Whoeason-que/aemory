# Project Specification: Aemory

## 1. 项目愿景 (Project Vision)

**Aemory** (AI + Memory) 是一个基于 **Rust** 的高性能 CLI 工具，被定义为 **"The Data Compiler for RAG"**。
它的核心任务是：读取包含 Markdown 文件的本地目录，对其进行智能语义切片（Semantic Chunking），生成向量嵌入（Embeddings），并将结果编译为可被 AI 高效索引和读取的 **Lance** 数据集。

## 2. 技术栈约束 (Tech Stack Constraints)

* **Language:** Rust (Latest Stable)
* **Core Storage:** `lance` (Rust native crate)
* **Data Format:** Apache Arrow (via `arrow_array`, `arrow_schema`)
* **Embedding Model:** `fastembed` (Rust crate, ONNX runtime) - 使用 `BAAI/bge-small-en-v1.5` 或 `multilingual` 作为默认模型，实现完全本地化、零外部 API 依赖。
* **CLI Framework:** `clap` (derive feature)
* **Async Runtime:** `tokio`
* **Markdown Processing:** 自定义解析逻辑，结合 `gray_matter` (用于解析 YAML Frontmatter)。

## 3. 核心数据结构 (Schema Definition)

生成的 Lance 数据集必须包含以下 Arrow Schema：

| Field Name | Data Type | Description |
| --- | --- | --- |
| `id` | `Utf8` | UUID v4，唯一标识符 |
| `vector` | `FixedSizeList<Float32>` | 文本的 Embedding 向量 (维度取决于模型，如 384) |
| `content` | `Utf8` | 切片后的实际文本内容 |
| `source_path` | `Utf8` | 原始文件路径 |
| `parent_header` | `Utf8` (Nullable) | 该切片所属的 Markdown 标题上下文 (例如: "# Intro > ## Setup") |
| `metadata` | `Json` (Utf8) | 从 Markdown Frontmatter 提取的完整元数据 |
| `created_at` | `Int64` | 编译时间戳 |

## 4. 构建步骤 (Step-by-Step Implementation Plan)

请 AI Agent 严格按照以下步骤生成代码，每一步完成后进行编译检查。

### Step 1: 项目初始化与依赖配置

* 执行 `cargo new aemory`。
* 配置 `Cargo.toml`，引入核心库：
* `lance`, `arrow`, `arrow-array`, `arrow-schema`
* `fastembed` (用于本地向量化)
* `clap` (features = ["derive"])
* `tokio` (features = ["full"])
* `gray_matter` (解析 Markdown metadata)
* `uuid`, `serde`, `serde_json`, `walkdir` (遍历目录)



### Step 2: 定义数据模型 (The Record Struct)

* 创建一个 `schema.rs` 模块。
* 定义 Rust 结构体 `DocumentChunk`，并实现将其转换为 Arrow `RecordBatch` 的 trait 或方法。
* **关键点：** 确保 Vector 列的构建符合 Lance 的 `FixedSizeList` 要求。

### Step 3: 实现 Markdown 智能加载器 (Markdown Loader)

* 创建一个 `loader.rs` 模块。
* **功能 A：** 遍历指定目录下的所有 `.md` 文件。
* **功能 B (核心)：** 解析 Markdown。
1. 提取 YAML Frontmatter (作为 metadata)。
2. **智能切分逻辑：** 不要只按字符数切分。实现一个基于 Header 的递归切分器。
* 规则：如果一个 Section (H1/H2) 下的内容超过 `chunk_size` (如 512 tokens)，则按段落切分；如果未超过，则保持完整。
* Context 注入：每个 Chunk 必须保留其父级标题路径 (例如 `Graph Theory > Algorithms > Dijkstra`)，拼接到内容前或存入 `parent_header` 字段，这对 RAG 至关重要。





### Step 4: 集成嵌入引擎 (Embedding Engine)

* 创建一个 `embedder.rs` 模块。
* 初始化 `fastembed::TextEmbedding`。
* 实现一个 `generate_embeddings` 函数，接收字符串列表，返回 `Vec<Vec<f32>>`。
* **优化：** 使用 `rayon` 进行数据预处理，但 Embedding 推理通常受限于 CPU/GPU，使用 Batch 处理 (Batch Size 设为 64 或 128)。

### Step 5: 实现 Lance 编译器 (The Compiler)

* 创建一个 `compiler.rs` 模块。
* 功能：接收带有 Embedding 的 Chunks，将其写入 Lance Dataset。
* 如果 Dataset 已存在，使用 `Overwrite` 或 `Append` 模式（CLI 参数控制）。
* 开启 Lance 的自动索引构建 (IVF-PQ) 以加速查询（可选，建议数据量 > 10k 时开启）。

### Step 6: CLI 入口点 (Main Integration)

* 在 `main.rs` 中使用 Clap 定义命令：
```bash
aemory build --input ./knowledge_base --output ./aemory.lance --model bge-small-en

```


* 串联 Pipeline： `Load -> Chunk -> Embed -> Write`。
* 添加进度条 (`indicatif` crate) 显示编译进度。

---

### 给 AI 的额外提示 (Pro Tips for the Agent)

1. **关于 Error Handling:** 使用 `anyhow` 进行错误传递，不要到处 `unwrap()`。
2. **关于 Markdown 切片:** 这是一个难点。如果写不出完美的递归切分器，可以先使用 "按双换行符 (`\n\n`) 分割段落 + 附加 Frontmatter" 的简单策略作为 MVP。
3. **关于 Lance:** 确保写入时 Batch Size 足够大（例如 1024 行一个 Batch），频繁的小写入会降低 Lance 的性能并产生大量碎片文件。
4. 根据步骤一步一步来,你有任何疑问，可以向我提问,有大方向的决策需要向我确认。
