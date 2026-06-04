# setup.py
from src.ingest import run_ingestion, clear_pinecone_index
from src.bm25_index import build_bm25_index, save_bm25_index
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Clear existing vectors first to prevent duplicates
    print("=" * 50)
    print("STEP 0: Clearing existing Pinecone data")
    print("=" * 50)
    index_name = os.getenv("PINECONE_INDEX_NAME",
                           "enterprise-knowledge-assistant")
    clear_pinecone_index(index_name)

    # Pinecone ingestion
    run_ingestion()

    # Build and save BM25 index
    print("\n" + "=" * 50)
    print("STEP 5: Building BM25 index")
    print("=" * 50)
    bm25, chunks = build_bm25_index()
    save_bm25_index(bm25, chunks)
    print("\nFULL SETUP COMPLETE")