# src/ingest.py
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.helper import load_all_documents


load_dotenv()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes the local HuggingFace embedding model.
    Uses all-MiniLM-L6-v2 which produces 384-dimensional vectors.
    """
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded.")
    return embeddings


def initialize_pinecone_index(index_name: str) -> None:
    """
    Creates Pinecone index if it doesn't already exist.
    Dimension 384 matches all-MiniLM-L6-v2 output.
    Cosine metric is standard for semantic similarity.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        print("Waiting for index to initialize...")
        time.sleep(10)
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

def clear_pinecone_index(index_name: str) -> None:
    """
    Deletes all vectors from all namespaces in the index.
    Used before re-ingestion to prevent duplicate chunks.
    """
    pc = get_pinecone_client() if False else Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
    index = pc.Index(index_name)

    roles = ["intern", "engineer", "manager", "executive"]
    for namespace in roles:
        print(f"Clearing namespace: {namespace}...")
        index.delete(delete_all=True, namespace=namespace)

    print("All namespaces cleared.\n")

def ingest_role_chunks(chunks: list, role: str,
                       embeddings: HuggingFaceEmbeddings,
                       index_name: str) -> None:
    """
    Embeds and upserts chunks for a single role into Pinecone.

    Args:
        chunks (list): Chunked LangChain Documents
        role (str): The role these chunks belong to
        embeddings: The embedding model
        index_name (str): Target Pinecone index
    """
    print(f"[{role}] Ingesting {len(chunks)} chunks into Pinecone...")

    # PineconeVectorStore handles embedding + upserting in batches
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=role   # separate namespace per role for clean organization
    )

    print(f"[{role}] Done.\n")


def run_ingestion() -> None:
    """
    Full ingestion pipeline:
    1. Load all documents from data/ folder
    2. Chunk them
    3. Initialize Pinecone index
    4. Embed and upsert each role's chunks
    """
    index_name = os.getenv("PINECONE_INDEX_NAME", "enterprise-knowledge-assistant")

    # Step 1: Load and chunk all documents
    print("=" * 50)
    print("STEP 1: Loading and chunking documents")
    print("=" * 50)
    all_chunks = load_all_documents()

    # Step 2: Load embedding model
    print("=" * 50)
    print("STEP 2: Loading embedding model")
    print("=" * 50)
    embeddings = get_embedding_model()

    # Step 3: Initialize Pinecone
    print("=" * 50)
    print("STEP 3: Initializing Pinecone index")
    print("=" * 50)
    initialize_pinecone_index(index_name)

    # Step 4: Ingest each role
    print("=" * 50)
    print("STEP 4: Ingesting chunks into Pinecone")
    print("=" * 50)
    for role, chunks in all_chunks.items():
        ingest_role_chunks(chunks, role, embeddings, index_name)

    print("=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"\nTotal chunks ingested: "
          f"{sum(len(c) for c in all_chunks.values())}")


if __name__ == "__main__":
    run_ingestion()