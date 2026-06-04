# src/bm25_index.py
import os
import pickle
import string
from rank_bm25 import BM25Okapi
from src.helper import load_raw_text_by_role, chunk_text_for_bm25

# Path where the BM25 index is saved to disk
BM25_INDEX_PATH = "bm25_index.pkl"


def tokenize(text: str) -> list:
    """
    Simple tokenizer for BM25.
    Lowercases, removes punctuation, splits on whitespace.

    BM25 works on token lists — the quality of tokenization
    directly affects retrieval quality.

    Args:
        text (str): Input text

    Returns:
        list: List of lowercase tokens
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return tokens


def build_bm25_index(data_dir: str = "data") -> tuple:
    """
    Builds a BM25 index from all documents in all role folders.

    Steps:
    1. Load raw text from all .md files
    2. Chunk them the same way as Pinecone ingestion
    3. Tokenize each chunk
    4. Build BM25Okapi index

    Returns:
        tuple: (BM25Okapi index, list of chunk dicts)
    """
    print("\n[BM25] Building index...")

    # Load and chunk all documents
    role_texts = load_raw_text_by_role(data_dir)
    all_chunks = chunk_text_for_bm25(role_texts)

    # Tokenize each chunk for BM25
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in all_chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    print(f"[BM25] Index built with {len(all_chunks)} chunks.")
    return bm25, all_chunks


def save_bm25_index(bm25, chunks: list,
                    path: str = BM25_INDEX_PATH) -> None:
    """
    Saves the BM25 index and chunk metadata to disk using pickle.

    Args:
        bm25: BM25Okapi index object
        chunks (list): List of chunk dicts with metadata
        path (str): File path to save to
    """
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"[BM25] Index saved to {path}")


def load_bm25_index(path: str = BM25_INDEX_PATH) -> tuple:
    """
    Loads the BM25 index and chunks from disk.

    Args:
        path (str): Path to the saved pickle file

    Returns:
        tuple: (BM25Okapi index, list of chunk dicts)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BM25 index not found at {path}. "
            f"Run python setup.py to build it first."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"[BM25] Index loaded from {path} "
          f"({len(data['chunks'])} chunks)")
    return data["bm25"], data["chunks"]


def bm25_search(query: str, role: str, bm25,
                chunks: list, top_k: int = 4) -> list:
    """
    Searches the BM25 index for a query, filtered by allowed roles.

    This is the sparse retrieval component of hybrid search.
    It only returns chunks that the user's role is allowed to see.

    Args:
        query (str): The user's question
        role (str): The user's role
        bm25: BM25Okapi index
        chunks (list): All chunk dicts with metadata
        top_k (int): Number of results to return

    Returns:
        list: Top-k chunk dicts ranked by BM25 score
    """
    from src.rbac import get_allowed_namespaces

    allowed_namespaces = get_allowed_namespaces(role)

    # Tokenize the query the same way we tokenized documents
    tokenized_query = tokenize(query)

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score and filter by allowed roles
    scored_chunks = []
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        if chunk["role"] in allowed_namespaces:
            scored_chunks.append({
                "chunk": chunk,
                "score": score,
                "index": i
            })

    # Sort by score descending and take top_k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k]

    print(f"[BM25] Query: '{query[:50]}...' "
          f"| Role: {role} "
          f"| Top result: {top_chunks[0]['chunk']['source'] if top_chunks else 'none'}")

    return top_chunks