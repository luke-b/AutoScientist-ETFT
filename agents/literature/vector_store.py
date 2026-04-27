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

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
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
    Simple in-memory store with token-overlap relevance ranking.

    Uses token-overlap scoring between the query and each chunk as a
    lightweight substitute for vector similarity.  This is the fallback
    when ChromaDB is not installed and is suitable for unit tests and
    short-lived single-run evaluations.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._paper_ids: set[str] = set()
        self._doc_ids: set[str] = set()

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            doc_id = f"{chunk.paper_id}::{chunk.chunk_index}"
            if doc_id not in self._doc_ids:
                self._chunks.append(chunk)
                self._paper_ids.add(chunk.paper_id)
                self._doc_ids.add(doc_id)

    def query(self, text: str, n_results: int = 20) -> list[Chunk]:
        if not self._chunks:
            return []
        query_tokens = set(text.lower().split())

        def _overlap(chunk: Chunk) -> int:
            return len(query_tokens & set(chunk.text.lower().split()))

        ranked = sorted(self._chunks, key=_overlap, reverse=True)
        return ranked[:n_results]

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self._paper_ids


# ---------------------------------------------------------------------------
# ChromaDB-backed persistent store
# ---------------------------------------------------------------------------


class _OfflineEmbeddingFunction:
    """
    Lightweight hash-based embedding for offline / CI use.

    Uses word-level MD5 hashing to produce a 128-dimensional unit vector.
    Requires no network access or external model downloads, making it
    suitable for air-gapped environments and CI pipelines.

    Pass an instance to :class:`ChromaVectorStore` when semantic quality
    is not critical::

        store = ChromaVectorStore(
            persist_dir="/tmp/vs",
            embedding_function=_OfflineEmbeddingFunction(),
        )

    In production, omit *embedding_function* to use ChromaDB's default
    ONNX-based embedding function for proper semantic similarity.
    """

    _DIM = 128

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        embeddings: list[list[float]] = []
        for text in input:
            vec = [0.0] * self._DIM
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._DIM
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0.0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings

    # -- ChromaDB duck-typed interface --

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Delegate to __call__ (query and document embeddings are identical)."""
        return self(input)

    @staticmethod
    def name() -> str:
        return "offline_hash"

    @staticmethod
    def build_from_config(embedding_config: dict) -> _OfflineEmbeddingFunction:  # noqa: ARG004
        return _OfflineEmbeddingFunction()

    def get_config(self) -> dict:
        return {}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]


class ChromaVectorStore(VectorStore):
    """
    Persistent vector store backed by ChromaDB.

    By default ChromaDB uses its built-in ONNX embedding function which
    requires a one-time model download.  Pass an explicit
    *embedding_function* (e.g. :class:`_OfflineEmbeddingFunction`) to skip
    any network access — useful in CI and air-gapped environments.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB persists its data across runs.
    collection_name:
        Name of the ChromaDB collection to use.
    embedding_function:
        Optional callable conforming to the ChromaDB ``EmbeddingFunction``
        protocol.  Defaults to ``None`` (uses ChromaDB's built-in default).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "literature",
        embedding_function=None,
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
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


def build_vector_store(
    cfg: dict | None = None,
    persist_dir: str | None = None,
) -> VectorStore:
    """
    Build and return a :class:`VectorStore` based on the config.

    Config keys (under ``vector_store``):
        backend      : "chroma" | "memory"  (default "memory")
        persist_dir  : path for ChromaDB persistence  (default "./data/vector_store")
        collection   : collection name  (default "literature")

    The persistence directory can also be set via the ``VECTOR_STORE_PERSIST_DIR``
    environment variable, which takes precedence over ``config.yaml``.

    Parameters
    ----------
    cfg:
        Configuration dict (from config.yaml).
    persist_dir:
        Explicit override for the ChromaDB persistence directory.
        Takes highest precedence over the env var and config.yaml.

    Raises
    ------
    ImportError
        When ``backend`` is ``"chroma"`` but chromadb is not installed.
    """
    vs_cfg = (cfg or {}).get("vector_store", {})
    backend: str = vs_cfg.get("backend", "memory")

    if backend == "chroma":
        try:
            import chromadb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for the 'chroma' vector store backend. "
                "Install it with: pip install 'autoscientist-etft[rag]'"
            ) from exc

        resolved_dir = (
            persist_dir
            or os.getenv("VECTOR_STORE_PERSIST_DIR")
            or vs_cfg.get("persist_dir", "./data/vector_store")
        )
        collection = vs_cfg.get("collection", "literature")

        # Ensure the directory exists before ChromaDB opens it
        Path(resolved_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Vector store backend: chroma (persist_dir=%r, collection=%r)",
            resolved_dir,
            collection,
        )
        return ChromaVectorStore(persist_dir=resolved_dir, collection_name=collection)

    logger.info("Vector store backend: memory (transient)")
    return InMemoryVectorStore()
