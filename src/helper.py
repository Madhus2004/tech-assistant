# src/helper.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
import pickle


load_dotenv()


def load_documents_from_role(role: str, data_dir: str = "data") -> list:
    """
    Loads all .md files from a role's folder.

    Args:
        role (str): One of intern, engineer, manager, executive
        data_dir (str): Root data directory

    Returns:
        list: LangChain Document objects with content and metadata
    """
    role_path = os.path.join(data_dir, role)

    loader = DirectoryLoader(
        role_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()

    # Attach role metadata to every document before chunking
    for doc in documents:
        doc.metadata["role"] = role
        # Extract clean filename without extension as source name
        filename = os.path.basename(doc.metadata["source"])
        doc.metadata["source"] = os.path.splitext(filename)[0]

    print(f"[{role}] Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list, chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list:
    """
    Splits documents into smaller chunks for embedding.

    Args:
        documents (list): LangChain Document objects
        chunk_size (int): Max characters per chunk
        chunk_overlap (int): Overlapping characters between chunks

    Returns:
        list: Chunked Document objects with preserved metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order
        # It tries to split on paragraphs first, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata so we know position in original doc
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"Split into {len(chunks)} chunks "
          f"(avg {len(''.join(c.page_content for c in chunks)) // len(chunks)}"
          f" chars/chunk)")
    return chunks


def load_all_documents(data_dir: str = "data") -> dict:
    """
    Loads and chunks documents for all 4 roles.

    Returns:
        dict: Keys are role names, values are lists of chunked Documents
    """
    roles = ["intern", "engineer", "manager", "executive"]
    all_chunks = {}

    for role in roles:
        docs = load_documents_from_role(role, data_dir)
        chunks = chunk_documents(docs)
        all_chunks[role] = chunks
        print(f"[{role}] Ready: {len(chunks)} chunks\n")

    return all_chunks

def load_raw_text_by_role(data_dir: str = "data") -> dict:
    """
    Loads raw text content from all markdown files grouped by role.
    Used for building the BM25 index.

    Returns:
        dict: {role: [{"text": str, "source": str, "role": str}]}
    """
    roles = ["intern", "engineer", "manager", "executive"]
    role_texts = {}

    for role in roles:
        role_path = os.path.join(data_dir, role)
        texts = []

        for filename in os.listdir(role_path):
            if filename.endswith(".md"):
                filepath = os.path.join(role_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                source = os.path.splitext(filename)[0]
                texts.append({
                    "text": content,
                    "source": source,
                    "role": role
                })

        role_texts[role] = texts
        print(f"[BM25] Loaded {len(texts)} raw docs for role: {role}")

    return role_texts


def chunk_text_for_bm25(role_texts: dict,
                         chunk_size: int = 500,
                         chunk_overlap: int = 50) -> list:
    """
    Chunks raw text documents for BM25 indexing.
    Returns a flat list of chunk dicts with metadata.

    Returns:
        list: [{"text": str, "source": str, "role": str, "chunk_index": int}]
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_index = 0

    for role, docs in role_texts.items():
        for doc in docs:
            # Wrap in LangChain Document for splitting
            lc_doc = Document(
                page_content=doc["text"],
                metadata={"source": doc["source"], "role": doc["role"]}
            )
            split_docs = splitter.split_documents([lc_doc])

            for split in split_docs:
                all_chunks.append({
                    "text": split.page_content,
                    "source": split.metadata["source"],
                    "role": split.metadata["role"],
                    "chunk_index": chunk_index
                })
                chunk_index += 1

    print(f"[BM25] Total chunks for BM25 index: {len(all_chunks)}")
    return all_chunks