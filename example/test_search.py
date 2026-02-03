import aemory
import os

# Get directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(base_dir, "knowledge_base")
output_path = os.path.join(base_dir, "test_output.lance")

# Compile first
if not os.path.exists(output_path):
    print("Building dataset first...")
    aemory.build(kb_path, output_path, "BAAI/bge-small-en-v1.5")

print(f"\n--- Searching in {output_path} ---")
results = aemory.search(output_path, "compiler", 3, "BAAI/bge-small-en-v1.5")

for r in results:
    print(f"Score: {r['score']}")
    print(f"Content: {r['content'][:50]}...")
    print("-" * 20)
