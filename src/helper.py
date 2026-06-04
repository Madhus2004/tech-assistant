# src/helper.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

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
