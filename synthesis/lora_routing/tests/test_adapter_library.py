"""
synthesis/lora_routing/tests/test_adapter_library.py — Unit tests for
AdapterLibrary (CRUD operations using temporary directories).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from synthesis.lora_routing.adapter_library import AdapterLibrary, AdapterRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lib(tmp_path) -> AdapterLibrary:
    """Return an AdapterLibrary backed by a fresh tmp directory."""
    index_path = tmp_path / "adapter_index.json"
    return AdapterLibrary(index_path=index_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_creates_record(lib, tmp_path):
    adapter_dir = tmp_path / "gen_1"
    adapter_dir.mkdir()
    record = lib.register(generation_index=1, adapter_path=adapter_dir)

    assert record.generation_index == 1
    assert record.adapter_path == str(adapter_dir)
    assert record.validated is False


def test_register_persists_to_disk(lib, tmp_path):
    adapter_dir = tmp_path / "gen_2"
    adapter_dir.mkdir()
    lib.register(generation_index=2, adapter_path=adapter_dir)

    # Reload from the same index file
    lib2 = AdapterLibrary(index_path=lib._index_path)
    record = lib2.get(2)
    assert record.adapter_path == str(adapter_dir)


def test_get_returns_registered_record(lib, tmp_path):
    adapter_dir = tmp_path / "gen_5"
    adapter_dir.mkdir()
    lib.register(generation_index=5, adapter_path=adapter_dir, metadata={"test": True})

    record = lib.get(5)
    assert record.generation_index == 5
    assert record.metadata["test"] is True


def test_get_raises_key_error_for_missing(lib):
    with pytest.raises(KeyError, match="generation 99"):
        lib.get(99)


def test_list_adapters_sorted(lib, tmp_path):
    for i in [3, 1, 2]:
        d = tmp_path / f"gen_{i}"
        d.mkdir()
        lib.register(generation_index=i, adapter_path=d)

    adapters = lib.list_adapters()
    indices = [r.generation_index for r in adapters]
    assert indices == [1, 2, 3]


def test_register_overwrites_existing(lib, tmp_path):
    d1 = tmp_path / "gen_1_v1"
    d1.mkdir()
    lib.register(generation_index=1, adapter_path=d1)

    d2 = tmp_path / "gen_1_v2"
    d2.mkdir()
    lib.register(generation_index=1, adapter_path=d2, metadata={"version": 2})

    record = lib.get(1)
    assert record.adapter_path == str(d2)
    assert record.metadata["version"] == 2


def test_mark_validated_updates_flag(lib, tmp_path):
    d = tmp_path / "gen_3"
    d.mkdir()
    lib.register(generation_index=3, adapter_path=d)
    assert lib.get(3).validated is False

    lib.mark_validated(3)
    assert lib.get(3).validated is True


def test_mark_validated_raises_for_missing(lib):
    with pytest.raises(KeyError):
        lib.mark_validated(404)


def test_index_file_is_valid_json(lib, tmp_path):
    d = tmp_path / "gen_7"
    d.mkdir()
    lib.register(generation_index=7, adapter_path=d)

    raw = json.loads(lib._index_path.read_text())
    assert "adapters" in raw
    assert len(raw["adapters"]) == 1
    assert raw["adapters"][0]["generation_index"] == 7


def test_corrupted_index_starts_fresh(tmp_path):
    index_path = tmp_path / "adapter_index.json"
    index_path.write_text("not valid json{{{{")

    lib = AdapterLibrary(index_path=index_path)
    assert lib.list_adapters() == []


def test_save_new_adapter_no_torch(lib, tmp_path):
    """save_new_adapter registers the path even when torch is unavailable."""
    adapter_dir = tmp_path / "gen_9"
    adapter_dir.mkdir()

    # Pass lora_state_dict=None to skip torch.save
    record = lib.save_new_adapter(
        generation_index=9,
        lora_state_dict=None,
        adapter_dir=adapter_dir,
    )
    assert record.generation_index == 9
    assert lib.get(9).adapter_path == str(adapter_dir)


def test_metadata_preserved_across_reload(lib, tmp_path):
    d = tmp_path / "gen_4"
    d.mkdir()
    lib.register(
        generation_index=4,
        adapter_path=d,
        metadata={"base_model": "llama3", "variant_count": 4},
    )

    lib2 = AdapterLibrary(index_path=lib._index_path)
    record = lib2.get(4)
    assert record.metadata["base_model"] == "llama3"
    assert record.metadata["variant_count"] == 4
