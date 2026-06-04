# src/hybrid_retriever.py
from langchain.schema import Document
from src.retriever import retrieve_documents
from src.bm25_index import load_bm25_index, bm25_search

# Load BM25 index once at module level
_bm25 = None
_bm25_chunks = None


def get_bm25():
    """Returns BM25 index and chunks, loading from disk if needed."""
    global _bm25, _bm25_chunks
    if _bm25 is None:
        _bm25, _bm25_chunks = load_bm25_index()
    return _bm25, _bm25_chunks


def reciprocal_rank_fusion(dense_docs: list,
                            sparse_results: list,
                            k: int = 60) -> list:
    """
    Merges dense and sparse retrieval results using
    Reciprocal Rank Fusion (RRF).

    RRF formula for each document:
        score = sum of 1/(rank + k) across all result lists

    The constant k=60 prevents rank-1 results from dominating.
    A document appearing in both lists scores higher than one
    appearing in only one list.

    Args:
        dense_docs (list): LangChain Documents from Pinecone
        sparse_results (list): Scored chunk dicts from BM25
        k (int): RRF constant (default 60 is standard)

    Returns:
        list: Merged and re-ranked LangChain Documents
    """
    # Dictionary to accumulate RRF scores by content hash
    rrf_scores = {}
    doc_map = {}

    # Score dense results
    for rank, doc in enumerate(dense_docs):
        content_key = doc.page_content[:100]
        if content_key not in rrf_scores:
            rrf_scores[content_key] = 0.0
            doc_map[content_key] = doc
        rrf_scores[content_key] += 1.0 / (rank + k)

    # Score sparse results — convert to LangChain Document format
    for rank, result in enumerate(sparse_results):
        chunk = result["chunk"]
        content_key = chunk["text"][:100]

        if content_key not in rrf_scores:
            rrf_scores[content_key] = 0.0
            # Convert BM25 chunk dict to LangChain Document
            doc_map[content_key] = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "role": chunk["role"],
                    "chunk_index": chunk["chunk_index"],
                    "retrieval_method": "bm25"
                }
            )

        rrf_scores[content_key] += 1.0 / (rank + k)

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(),
                         key=lambda x: rrf_scores[x],
                         reverse=True)

    merged = [doc_map[key] for key in sorted_keys]

    print(f"[Hybrid] Dense: {len(dense_docs)} chunks | "
          f"Sparse: {len(sparse_results)} chunks | "
          f"Merged: {len(merged)} unique chunks")

    return merged


def hybrid_retrieve(query: str, role: str,
                    top_k: int = 4) -> list:
    """
    Main hybrid retrieval function.
    Combines Pinecone dense search + BM25 sparse search
    using Reciprocal Rank Fusion.

    This replaces the simple retrieve_documents() call in app.py.

    Args:
        query (str): The user's question
        role (str): The user's role
        top_k (int): Final number of chunks to return

    Returns:
        list: Top-k merged and re-ranked LangChain Documents
    """
    print(f"\n[Hybrid] Query: '{query[:60]}'")
    print(f"[Hybrid] Role: {role}")

    # Dense retrieval from Pinecone
    dense_docs = retrieve_documents(query=query, role=role, top_k=top_k)

    # Sparse retrieval from BM25
    bm25, chunks = get_bm25()
    sparse_results = bm25_search(
        query=query,
        role=role,
        bm25=bm25,
        chunks=chunks,
        top_k=top_k
    )

    # Merge with RRF
    merged = reciprocal_rank_fusion(dense_docs, sparse_results)

    # Return top_k after fusion
    final_results = merged[:top_k]

    print(f"[Hybrid] Final chunks returned: {len(final_results)}")
    for doc in final_results:
        method = doc.metadata.get("retrieval_method", "dense")
        print(f"  - {doc.metadata['source']} "
              f"({doc.metadata['role']}) [{method}]")

    return final_results