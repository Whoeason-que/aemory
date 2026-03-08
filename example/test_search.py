"""
Example: Building and searching a knowledge base with Aemory

This script demonstrates:
1. Building a Lance dataset from markdown files
2. Performing semantic search on the compiled dataset
"""

import os
import aemory

# Recommended models (automatically downloaded on first use):
# - "sentence-transformers/all-MiniLM-L6-v2" (lightweight, 80MB)
# - "BAAI/bge-small-en-v1.5" (English, 130MB)
# - "BAAI/bge-base-en-v1.5" (English, better quality, 440MB)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    # Get directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, "knowledge_base")
    output_path = os.path.join(base_dir, "test_output.lance")

    # Step 1: Compile markdown files into Lance dataset
    # Always rebuild to ensure model consistency
    if os.path.exists(output_path):
        import shutil

        print("⚠️  Removing existing dataset to ensure model consistency...")
        shutil.rmtree(output_path)

    print("Building dataset from markdown files...")
    print(f"  Input:  {kb_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {MODEL_NAME}")
    aemory.build(kb_path, output_path, MODEL_NAME)
    print("✓ Dataset built successfully\n")

    # Step 2: Search for relevant chunks
    print(f"--- Searching in {output_path} ---")
    query = "compiler"
    limit = 3

    print(f"Query: '{query}'")
    print(f"Top {limit} results:\n")

    results = aemory.search(output_path, query, limit, MODEL_NAME)

    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Score: {r['score']:.4f}")
        print(f"    Source: {r['source_path']}")
        print(f"    Content: {r['content'][:100]}...")
        print(f"    Metadata: {r['metadata']}")
        print("-" * 60)
