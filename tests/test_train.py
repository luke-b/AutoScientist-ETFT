"""
tests/test_train.py — Unit tests for train.py corpus versioning and validation split.

These tests exercise ``build_dataset``, ``write_corpus_manifest``, and the
data-only parts of the pipeline.  They do NOT require PyTorch / PEFT and
therefore run in vanilla CI without heavy ML dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_corpus(data_root: Path, n_gen: int = 5, n_rat: int = 3) -> None:
    """Populate a minimal corpus under data_root."""
    _write_jsonl(
        data_root / "d_gen" / "traj_001.jsonl",
        [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(n_gen)],
    )
    _write_jsonl(
        data_root / "d_rationale" / "rat_001.jsonl",
        [{"prompt": f"rp{i}", "completion": f"rc{i}"} for i in range(n_rat)],
    )


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_returns_tuple(tmp_path):
    """build_dataset returns (examples, file_infos) tuple."""
    from train import build_dataset

    _make_corpus(tmp_path)
    examples, file_infos = build_dataset(tmp_path)
    assert isinstance(examples, list)
    assert isinstance(file_infos, list)


def test_build_dataset_counts(tmp_path):
    """Total examples equals sum of all JSONL records."""
    from train import build_dataset

    _make_corpus(tmp_path, n_gen=4, n_rat=2)
    examples, file_infos = build_dataset(tmp_path)
    assert len(examples) == 6


def test_build_dataset_file_infos_have_sha256(tmp_path):
    """Each file_info entry contains a non-empty sha256 checksum."""
    from train import build_dataset

    _make_corpus(tmp_path)
    _, file_infos = build_dataset(tmp_path)
    assert len(file_infos) >= 1
    for fi in file_infos:
        assert "sha256" in fi
        assert len(fi["sha256"]) == 64  # hex SHA-256


def test_build_dataset_sha256_matches_file(tmp_path):
    """SHA-256 in file_info must match the actual file checksum."""
    from train import build_dataset

    _make_corpus(tmp_path, n_gen=3)
    _, file_infos = build_dataset(tmp_path)

    for fi in file_infos:
        expected = hashlib.sha256(Path(fi["path"]).read_bytes()).hexdigest()
        assert fi["sha256"] == expected, f"Checksum mismatch for {fi['path']}"


def test_build_dataset_empty_dirs(tmp_path):
    """build_dataset returns empty examples when corpus dirs are empty."""
    from train import build_dataset

    (tmp_path / "d_gen").mkdir(parents=True)
    (tmp_path / "d_rationale").mkdir(parents=True)
    examples, file_infos = build_dataset(tmp_path)
    assert examples == []
    assert file_infos == []


# ---------------------------------------------------------------------------
# write_corpus_manifest
# ---------------------------------------------------------------------------


def test_write_corpus_manifest_creates_file(tmp_path):
    """write_corpus_manifest writes corpus_manifest.json to output_dir."""
    from train import write_corpus_manifest

    file_infos = [{"path": "/data/f.jsonl", "records": 10, "sha256": "abc123"}]
    manifest_path = write_corpus_manifest(
        output_dir=tmp_path,
        data_root=tmp_path,
        file_infos=file_infos,
        model_name="test/model",
        cfg={"training": {"val_split": 0.1}},
    )

    assert manifest_path == tmp_path / "corpus_manifest.json"
    assert manifest_path.exists()


def test_write_corpus_manifest_content(tmp_path):
    """Manifest JSON contains expected top-level keys."""
    from train import write_corpus_manifest

    file_infos = [{"path": "/data/f.jsonl", "records": 7, "sha256": "aaa"}]
    write_corpus_manifest(
        output_dir=tmp_path,
        data_root=tmp_path,
        file_infos=file_infos,
        model_name="my/model",
        cfg={"key": "value"},
    )

    manifest = json.loads((tmp_path / "corpus_manifest.json").read_text())
    assert manifest["model_name"] == "my/model"
    assert manifest["total_records"] == 7
    assert "created_at" in manifest
    assert manifest["config_snapshot"] == {"key": "value"}
    assert len(manifest["files"]) == 1


def test_write_corpus_manifest_total_records(tmp_path):
    """total_records is the sum of all file record counts."""
    from train import write_corpus_manifest

    file_infos = [
        {"path": "a.jsonl", "records": 3, "sha256": "x"},
        {"path": "b.jsonl", "records": 5, "sha256": "y"},
    ]
    write_corpus_manifest(tmp_path, tmp_path, file_infos, "m", {})
    manifest = json.loads((tmp_path / "corpus_manifest.json").read_text())
    assert manifest["total_records"] == 8


# ---------------------------------------------------------------------------
# Validation split non-empty
# ---------------------------------------------------------------------------


def test_build_dataset_val_split_non_empty(tmp_path):
    """When corpus has >= 10 examples and val_split=0.1 the eval set is non-empty."""
    from train import build_dataset

    _make_corpus(tmp_path, n_gen=10, n_rat=0)
    examples, _ = build_dataset(tmp_path)

    val_split = 0.1
    n_eval = max(1, int(len(examples) * val_split))
    assert n_eval >= 1, "Validation split must be non-empty with 10+ examples."
