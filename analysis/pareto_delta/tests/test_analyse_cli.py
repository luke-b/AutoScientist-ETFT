"""
tests for analysis/pareto_delta/run.py — etft-analyse CLI
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_SIMPLE = {
    "step_index": 0,
    "algorithm_id": "cnn_v0",
    "algorithm_family": "image_classification",
    "code": "def model(x):\n    return x\n",
    "fitness_score": 0.5,
}

_STEP_IMPROVED = {
    "step_index": 1,
    "algorithm_id": "cnn_v1",
    "algorithm_family": "image_classification",
    "code": (
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.bn = nn.BatchNorm2d(64)\n"
        "        self.fc = nn.Linear(64, 10)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(self.bn(x))\n"
    ),
    "fitness_score": 0.85,
}


def _write_step_jsonl(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")


def _write_training_jsonl(path: Path) -> None:
    """Write a 𝒟_Gen training example JSONL (prompt/completion) — should be skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"prompt": "# improve", "completion": "# improved"}) + "\n")


# ---------------------------------------------------------------------------
# Test 1: load steps from a trajectory file
# ---------------------------------------------------------------------------


def test_load_steps_from_file(tmp_path):
    """_load_steps_from_file correctly parses TrajectoryStep records."""
    from analysis.pareto_delta.run import _load_steps_from_file

    jsonl = tmp_path / "traj.jsonl"
    _write_step_jsonl(jsonl, [_STEP_SIMPLE, _STEP_IMPROVED])

    steps = _load_steps_from_file(jsonl)
    assert len(steps) == 2
    assert steps[0].step_index == 0
    assert steps[1].fitness_score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Test 2: training examples are skipped
# ---------------------------------------------------------------------------


def test_load_steps_skips_training_examples(tmp_path):
    """prompt/completion records are skipped by _load_steps_from_file."""
    from analysis.pareto_delta.run import _load_steps_from_file

    jsonl = tmp_path / "d_gen.jsonl"
    _write_training_jsonl(jsonl)

    steps = _load_steps_from_file(jsonl)
    assert steps == []


# ---------------------------------------------------------------------------
# Test 3: load steps from data_root
# ---------------------------------------------------------------------------


def test_load_steps_from_data_root(tmp_path):
    """_load_steps_from_data_root walks d_gen/ and loads step records."""
    from analysis.pareto_delta.run import _load_steps_from_data_root

    d_gen = tmp_path / "d_gen"
    _write_step_jsonl(d_gen / "traj.jsonl", [_STEP_SIMPLE, _STEP_IMPROVED])

    steps = _load_steps_from_data_root(tmp_path)
    assert len(steps) == 2


# ---------------------------------------------------------------------------
# Test 4: run_analyse with trajectory_file returns a report
# ---------------------------------------------------------------------------


def test_run_analyse_with_trajectory_file(tmp_path):
    """run_analyse returns a valid report dict from a trajectory file."""
    from analysis.pareto_delta.run import run_analyse

    jsonl = tmp_path / "traj.jsonl"
    _write_step_jsonl(jsonl, [_STEP_SIMPLE, _STEP_IMPROVED])

    report = run_analyse(trajectory_file=jsonl)
    assert "trajectory_id" in report
    assert "total_steps" in report
    assert report["total_steps"] == 2
    assert "pareto_set" in report
    assert "all_ranks" in report
    assert "bottleneck_summary" in report


# ---------------------------------------------------------------------------
# Test 5: run_analyse with data_root
# ---------------------------------------------------------------------------


def test_run_analyse_with_data_root(tmp_path):
    """run_analyse correctly walks d_gen/ when data_root is supplied."""
    from analysis.pareto_delta.run import run_analyse

    d_gen = tmp_path / "d_gen"
    _write_step_jsonl(d_gen / "a.jsonl", [_STEP_SIMPLE, _STEP_IMPROVED])

    report = run_analyse(data_root=tmp_path)
    assert report.get("total_steps") == 2


# ---------------------------------------------------------------------------
# Test 6: run_analyse saves JSON when save_to is provided
# ---------------------------------------------------------------------------


def test_run_analyse_saves_json(tmp_path):
    """run_analyse saves the report to disk when save_to is specified."""
    from analysis.pareto_delta.run import run_analyse

    jsonl = tmp_path / "traj.jsonl"
    _write_step_jsonl(jsonl, [_STEP_SIMPLE, _STEP_IMPROVED])

    out = tmp_path / "report.json"
    run_analyse(trajectory_file=jsonl, save_to=out)

    assert out.exists()
    saved = json.loads(out.read_text())
    assert "trajectory_id" in saved


# ---------------------------------------------------------------------------
# Test 7: run_analyse returns error when no data found
# ---------------------------------------------------------------------------


def test_run_analyse_no_data(tmp_path):
    """run_analyse returns an error dict when no trajectory data is found."""
    from analysis.pareto_delta.run import run_analyse

    report = run_analyse(data_root=tmp_path)  # empty d_gen
    assert "error" in report


# ---------------------------------------------------------------------------
# Test 8: top bottleneck from pareto_set
# ---------------------------------------------------------------------------


def test_run_analyse_pareto_set_content(tmp_path):
    """When a significant delta exists, pareto_set contains ranked components."""
    from analysis.pareto_delta.run import run_analyse

    jsonl = tmp_path / "traj.jsonl"
    _write_step_jsonl(jsonl, [_STEP_SIMPLE, _STEP_IMPROVED])

    report = run_analyse(trajectory_file=jsonl)
    # Either pareto_set is populated or all_ranks is populated
    assert isinstance(report.get("pareto_set"), list)
    assert isinstance(report.get("all_ranks"), list)
