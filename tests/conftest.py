"""
Shared pytest fixtures for the AutoScientist-ETFT test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Temporary data directories
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Provide a temporary data root with the standard sub-directory layout."""
    for sub in ("d_gen", "d_rationale", "d_perf", "trajectories"):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """A mock LLM client that returns a canned completion string."""
    client = MagicMock()
    response = MagicMock()
    response.content = "mocked LLM response"
    client.complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# Vector store — parametrized over both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "chroma"])
def any_vector_store(request, tmp_path: Path):
    """
    A VectorStore instance parametrized over both the in-memory and ChromaDB
    backends.  The chroma variant is automatically skipped when chromadb is
    not installed.
    """
    backend = request.param
    if backend == "chroma":
        pytest.importorskip("chromadb")
        from agents.literature.vector_store import ChromaVectorStore, _OfflineEmbeddingFunction

        return ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test",
            embedding_function=_OfflineEmbeddingFunction(),
        )

    from agents.literature.vector_store import InMemoryVectorStore

    return InMemoryVectorStore()
