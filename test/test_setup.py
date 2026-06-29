# test_setup.py
import sys
import os

# Add project root to Python path so src/ is findable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve_documents, format_context


from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os

load_dotenv()

# Local model - already verified working
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

test_vector = embeddings.embed_query("Hello world")
print(f"Embedding model loaded. Vector size: {len(test_vector)}")

# Test Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes = pc.list_indexes()
print(f"Pinecone connected. Indexes: {[i.name for i in indexes]}")

print("\nAll systems ready.")