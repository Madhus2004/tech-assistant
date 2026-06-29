# src/ingest.py
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.helper import load_all_documents
from src.embeddings import get_embeddings

load_dotenv()




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
    Clears all vectors from all namespaces.
    Uses list-then-delete approach for Pinecone serverless compatibility.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Index '{index_name}' does not exist yet. "
              f"Skipping clear — will be created during ingestion.\n")
        return

    # For serverless indexes, delete the index entirely and recreate
    # This is more reliable than namespace-level deletion on serverless
    print(f"Deleting index '{index_name}' for clean re-ingestion...")
    pc.delete_index(index_name)

    # Wait for deletion to complete
    import time
    time.sleep(5)
    print("Index deleted. Will be recreated during ingestion.\n")

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
    embeddings = get_embeddings()

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