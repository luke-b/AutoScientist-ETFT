"""
agents/literature/vector_store.py — Persistent vector store for literature chunks.

Provides semantic similarity search over paper chunks using ChromaDB
(if installed) or falls back to a simple in-memory list-based store that
returns chunks in insertion order.

Architecture
------------
  LiteratureRetriever   → VectorStore.upsert(chunks)   (persist new chunks)
  LiteratureSynthesizer → VectorStore.query(text, n)   (retrieve relevant chunks)

Usage
-----
    from agents.literature.vector_store import build_vector_store
    store = build_vector_store(cfg)
    store.upsert(chunks)
    relevant = store.query("batch normalisation", n_results=10)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.literature.retriever import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract interface for a chunk vector store."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Persist *chunks*, skipping any that are already stored."""

    @abstractmethod
    def query(self, text: str, n_results: int = 20) -> list[Chunk]:
        """Return up to *n_results* chunks most relevant to *text*."""

    @abstractmethod
    def has_paper(self, paper_id: str) -> bool:
        """Return True if any chunk for *paper_id* is already stored."""


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory store that returns chunks in insertion order.

    No semantic similarity — acts as a ordered buffer.  This is the
    fallback when ChromaDB is not installed.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._paper_ids: set[str] = set()

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            doc_id = f"{chunk.paper_id}::{chunk.chunk_index}"
            if doc_id not in {f"{c.paper_id}::{c.chunk_index}" for c in self._chunks}:
                self._chunks.append(chunk)
                self._paper_ids.add(chunk.paper_id)

    def query(self, text: str, n_results: int = 20) -> list[Chunk]:  # noqa: ARG002
        return self._chunks[:n_results]

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._paper_ids


# ---------------------------------------------------------------------------
# ChromaDB-backed persistent store
# ---------------------------------------------------------------------------


class ChromaVectorStore(VectorStore):
    """
    Persistent vector store backed by ChromaDB.

    ChromaDB uses its built-in embedding function (sentence-transformers or
    a lightweight default) to embed chunk texts and enables cosine-similarity
    queries.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB persists its data across runs.
    collection_name:
        Name of the ChromaDB collection to use.
    """

    def __init__(self, persist_dir: str, collection_name: str = "literature") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore: collection=%r persist_dir=%r (count=%d)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [f"{c.paper_id}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"paper_id": c.paper_id, "title": c.title, "chunk_index": c.chunk_index} for c in chunks]
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.debug("Upserted %d chunks into ChromaDB.", len(chunks))

    def query(self, text: str, n_results: int = 20) -> list[Chunk]:
        from agents.literature.retriever import Chunk as ChunkType

        count = self._collection.count()
        if count == 0:
            return []

        effective_n = min(n_results, count)
        results = self._collection.query(
            query_texts=[text],
            n_results=effective_n,
            include=["documents", "metadatas"],
        )

        chunks: list[ChunkType] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            chunks.append(
                ChunkType(
                    paper_id=meta.get("paper_id", ""),
                    title=meta.get("title", ""),
                    text=doc,
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return chunks

    def has_paper(self, paper_id: str) -> bool:
        results = self._collection.get(where={"paper_id": paper_id}, limit=1)
        return bool(results and results.get("ids"))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_vector_store(cfg: dict | None = None) -> VectorStore:
    """
    Build and return a :class:`VectorStore` based on the config.

    Config keys (under ``vector_store``):
        backend      : "chroma" | "memory"  (default "memory")
        persist_dir  : path for ChromaDB persistence  (default "./data/vector_store")
        collection   : collection name  (default "literature")

    Falls back to :class:`InMemoryVectorStore` if ``backend`` is ``"chroma"``
    but ChromaDB is not installed.
    """
    vs_cfg = (cfg or {}).get("vector_store", {})
    backend: str = vs_cfg.get("backend", "memory")

    if backend == "chroma":
        try:
            persist_dir = vs_cfg.get("persist_dir", "./data/vector_store")
            collection = vs_cfg.get("collection", "literature")
            return ChromaVectorStore(persist_dir=persist_dir, collection_name=collection)
        except ImportError:
            logger.warning(
                "chromadb not installed — falling back to InMemoryVectorStore. "
                "Install with: pip install 'autoscientist-etft[rag]'"
            )

    return InMemoryVectorStore()
