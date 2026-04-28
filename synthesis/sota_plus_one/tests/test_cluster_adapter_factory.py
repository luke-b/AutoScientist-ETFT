"""
tests for synthesis/sota_plus_one/cluster_adapter.py — get_adapter() factory
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Test 1: local adapter
# ---------------------------------------------------------------------------


def test_get_adapter_local():
    """get_adapter returns LocalSubprocessAdapter for 'local'."""
    from synthesis.sota_plus_one.cluster_adapter import (
        LocalSubprocessAdapter,
        get_adapter,
    )

    adapter = get_adapter({"cluster": {"adapter": "local"}})
    assert isinstance(adapter, LocalSubprocessAdapter)


# ---------------------------------------------------------------------------
# Test 2: slurm adapter
# ---------------------------------------------------------------------------


def test_get_adapter_slurm():
    """get_adapter returns SlurmAdapter for 'slurm'."""
    from synthesis.sota_plus_one.cluster_adapter import SlurmAdapter, get_adapter

    adapter = get_adapter({"cluster": {"adapter": "slurm"}})
    assert isinstance(adapter, SlurmAdapter)


# ---------------------------------------------------------------------------
# Test 3: kubernetes adapter
# ---------------------------------------------------------------------------


def test_get_adapter_kubernetes():
    """get_adapter returns KubernetesAdapter for 'kubernetes'."""
    from synthesis.sota_plus_one.cluster_adapter import KubernetesAdapter, get_adapter

    adapter = get_adapter({"cluster": {"adapter": "kubernetes"}})
    assert isinstance(adapter, KubernetesAdapter)


# ---------------------------------------------------------------------------
# Test 4: unknown adapter raises ValueError
# ---------------------------------------------------------------------------


def test_get_adapter_unknown():
    """get_adapter raises ValueError for an unknown adapter name."""
    from synthesis.sota_plus_one.cluster_adapter import get_adapter

    with pytest.raises(ValueError, match="Unknown cluster adapter"):
        get_adapter({"cluster": {"adapter": "ray"}})

    # Error message lists valid options
    with pytest.raises(ValueError, match="local"):
        get_adapter({"cluster": {"adapter": "bogus"}})


# ---------------------------------------------------------------------------
# Test 5: default to local when cluster section is absent
# ---------------------------------------------------------------------------


def test_get_adapter_default_local():
    """get_adapter defaults to LocalSubprocessAdapter when cfg is empty."""
    from synthesis.sota_plus_one.cluster_adapter import (
        LocalSubprocessAdapter,
        get_adapter,
    )

    adapter = get_adapter({})
    assert isinstance(adapter, LocalSubprocessAdapter)


# ---------------------------------------------------------------------------
# Test 6: default to local when cfg is None
# ---------------------------------------------------------------------------


def test_get_adapter_none_cfg():
    """get_adapter defaults to LocalSubprocessAdapter when cfg is None."""
    from synthesis.sota_plus_one.cluster_adapter import (
        LocalSubprocessAdapter,
        get_adapter,
    )

    adapter = get_adapter(None)
    assert isinstance(adapter, LocalSubprocessAdapter)


# ---------------------------------------------------------------------------
# Test 7: generate.py uses get_adapter instead of hardcoded LocalSubprocessAdapter
# ---------------------------------------------------------------------------


def test_generate_uses_get_adapter(tmp_path):
    """_submit_candidate uses get_adapter() — verify the import path changed."""
    import inspect

    from synthesis.sota_plus_one import generate

    source = inspect.getsource(generate._submit_candidate)
    assert "get_adapter" in source
    assert "LocalSubprocessAdapter" not in source
