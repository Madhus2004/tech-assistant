# test_ingestion.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

stats = index.describe_index_stats()
print("\nPinecone Index Stats:")
print(f"Total vectors: {stats.total_vector_count}")
print(f"\nVectors per namespace:")
for namespace, ns_stats in stats.namespaces.items():
    print(f"  {namespace}: {ns_stats.vector_count} vectors")