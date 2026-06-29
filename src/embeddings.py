# src/embeddings.py
import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_embeddings = None

class HFDirectEmbeddings:
    """
    Custom embedding class using direct HTTP calls to HuggingFace router.
    This approach works on Render's free tier where the standard
    HuggingFace client fails due to DNS resolution issues.

    Used and verified working in production on Render.
    """

    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.url = (
            "https://router.huggingface.co/hf-inference/models/"
            "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: list) -> list:
        """
        Embeds a list of texts. Called during retrieval for multiple chunks.

        Args:
            texts (list): List of strings to embed

        Returns:
            list: List of 384-dimensional vectors
        """
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts}
        )
        response.raise_for_status()
        embeddings = response.json()

        # Handle nested list response format from HF router
        if isinstance(embeddings[0][0], list):
            embeddings = [e[0] for e in embeddings]

        return embeddings

    def embed_query(self, text: str) -> list:
        """
        Embeds a single query string. Called during retrieval for the question.

        Args:
            text (str): The query string

        Returns:
            list: 384-dimensional vector
        """
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": [text]}
        )
        response.raise_for_status()
        result = response.json()

        # Handle nested list response
        if isinstance(result[0][0], list):
            return result[0][0]
        return result[0]


def get_embeddings():
    """
    Returns the embedding model based on EMBEDDING_MODE env variable.

    EMBEDDING_MODE=local  → HuggingFaceEmbeddings (local model)
    EMBEDDING_MODE=api    → HFDirectEmbeddings (direct HTTP to HF router)
    """
    global _embeddings
    if _embeddings is None:
        mode = os.getenv("EMBEDDING_MODE", "local")

        if mode == "api":
            print("Initializing HF Direct embeddings (router.huggingface.co)...")
            _embeddings = HFDirectEmbeddings()
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("Initializing local HuggingFace embeddings...")
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    return _embeddings