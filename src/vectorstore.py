"""Chroma index management.

Each chunking strategy gets its own persist directory and collection, so the
two indexes never contaminate each other. Distance is forced to cosine so the
scores are on a comparable 0–2 scale across strategies and runs.
"""

import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import get_embeddings, index_path

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def build_index(strategy: str, chunks: List[Document], reset: bool = True) -> Chroma:
    path = index_path(strategy)
    if reset and Path(path).exists():
        shutil.rmtree(path)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=path,
        collection_name=f"handbook_{strategy}",
        collection_metadata=_COLLECTION_METADATA,
    )
    print(f"Indexed {len(chunks)} chunks for '{strategy}' at {path}")
    return store


def get_vectorstore(strategy: str) -> Chroma:
    path = index_path(strategy)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No index for '{strategy}'. Run: python scripts/build_indexes.py"
        )
    return Chroma(
        persist_directory=path,
        embedding_function=get_embeddings(),
        collection_name=f"handbook_{strategy}",
        collection_metadata=_COLLECTION_METADATA,
    )


def count(strategy: str) -> int:
    return get_vectorstore(strategy)._collection.count()