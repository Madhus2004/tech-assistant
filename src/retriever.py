# src/retriever.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.rbac import get_allowed_namespaces
from src.embeddings import get_embeddings

load_dotenv()

_pinecone_client = None

def get_pinecone_client() -> Pinecone:
    """Returns the Pinecone client, initializing it only once."""
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return _pinecone_client


def retrieve_documents(query: str, role: str, top_k: int = 4) -> list:
    """
    Retrieves relevant document chunks for a query based on user role.

    This is the core RBAC retrieval function. It:
    1. Determines which namespaces the role can access
    2. Queries each allowed namespace separately
    3. Merges and deduplicates results
    4. Returns top_k most relevant chunks

    Args:
        query (str): The user's question
        role (str): The user's role (controls which namespaces are searched)
        top_k (int): Number of chunks to return per namespace

    Returns:
        list: Retrieved Document objects with content and metadata
    """
    allowed_namespaces = get_allowed_namespaces(role)
    embeddings = get_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME",
                           "enterprise-knowledge-assistant")

    all_results = []

    # Query each allowed namespace separately
    for namespace in allowed_namespaces:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )

        # similarity_search returns chunks ranked by cosine similarity
        results = vectorstore.similarity_search(
            query=query,
            k=top_k
        )

        all_results.extend(results)

    # Deduplicate by content in case same chunk appears in multiple namespaces
    seen = set()
    unique_results = []
    for doc in all_results:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen:
            seen.add(content_hash)
            unique_results.append(doc)

    print(f"\n[Retriever] Role: {role}")
    print(f"[Retriever] Searched namespaces: {allowed_namespaces}")
    print(f"[Retriever] Retrieved {len(unique_results)} unique chunks")

    return unique_results


def format_context(documents: list) -> str:
    """
    Formats retrieved document chunks into a single context string
    for the LLM prompt.

    Each chunk is labeled with its source document so the LLM
    can cite sources in its response.

    Args:
        documents (list): Retrieved Document objects

    Returns:
        str: Formatted context string
    """
    if not documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        role = doc.metadata.get("role", "unknown")
        context_parts.append(
            f"[Source {i}: {source} ({role} document)]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)