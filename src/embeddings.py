# src/embeddings.py
import os
from dotenv import load_dotenv

load_dotenv()

_embeddings = None


def get_embeddings():
    """
    Returns the embedding model based on EMBEDDING_MODE env variable.

    EMBEDDING_MODE=local  → HuggingFaceEmbeddings (local model, no API needed)
    EMBEDDING_MODE=api    → HuggingFaceEndpointEmbeddings (HF Inference API)

    Local is used for development on networks that block api-inference.huggingface.co
    API is used in Docker/Render where torch is not installed but network is open.
    """
    global _embeddings
    if _embeddings is None:
        mode = os.getenv("EMBEDDING_MODE", "local")

        if mode == "api":
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            print("Initializing HuggingFace API embeddings...")
            _embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("Initializing local HuggingFace embeddings...")
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    return _embeddings