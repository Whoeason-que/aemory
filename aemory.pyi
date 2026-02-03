from typing import List, Dict

def build(input: str, output: str, model: str) -> None:
    """
    Build the Lance dataset from a directory of Markdown files.

    This function reads markdown files from the input directory, chunks them based on headers,
    generates embeddings using the specified model, and writes the result to a Lance dataset.

    Args:
        input (str): Path to the directory containing markdown files.
        output (str): Destination path for the Lance dataset.
        model (str): Name of the embedding model to use (e.g., "BAAI/bge-small-en-v1.5").

    Raises:
        ValueError: If input or output paths are invalid.
        RuntimeError: If the embedding model fails to load or inference fails.
    """
    ...

def search(uri: str, query: str, limit: int, model: str) -> List[Dict[str, str]]:
    """
    Search the Lance dataset for chunks similar to the query.

    Args:
        uri (str): Path to the Lance dataset.
        query (str): The search query string.
        limit (int): Maximum number of results to return.
        model (str): Name of the embedding model to use.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing:
            - 'content': The text content of the chunk.
            - 'source_path': Path to the original file.
            - 'metadata': JSON string of metadata.
            - 'score': Similarity score (distance).
    """
    ...
