"""
tests/test_performance_estimator.py — Unit tests for corpus/performance_estimator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from corpus.performance_estimator.dataset_builder import PerfDatasetBuilder
from corpus.performance_estimator.feature_extractor import extract_features, feature_names
from corpus.performance_estimator.filter_model import ProbabilisticFilter
from corpus.regression_pipeline.schemas import ValidationResult, ValidationStatus

# ---------------------------------------------------------------------------
# feature_extractor
# ---------------------------------------------------------------------------


def test_feature_names_stable():
    names = feature_names()
    assert len(names) > 0
    assert "num_functions" in names
    assert "max_nesting_depth" in names


def test_extract_features_valid_code():
    code = """
def foo(x):
    for i in range(10):
        x += i
    return x

class Bar:
    pass
"""
    feats = extract_features(code)
    assert feats["num_functions"] == 1.0
    assert feats["num_classes"] == 1.0
    assert feats["num_loops"] >= 1.0
    assert feats["max_nesting_depth"] >= 1.0


def test_extract_features_syntax_error_returns_zeros():
    feats = extract_features("def foo(\n")
    assert all(v == 0.0 for v in feats.values())


def test_feature_vector_length_matches_names():
    code = "x = 1\n"
    feats = extract_features(code)
    assert set(feats.keys()) == set(feature_names())


# ---------------------------------------------------------------------------
# PerfDatasetBuilder
# ---------------------------------------------------------------------------


def test_perf_dataset_builder_writes_jsonl(tmp_path):
    out = tmp_path / "d_perf.jsonl"
    builder = PerfDatasetBuilder(out)

    pass_result = ValidationResult(status=ValidationStatus.PASS, duration_seconds=0.1)
    fail_result = ValidationResult(status=ValidationStatus.FAIL_OOM, stderr="CUDA out of memory")

    s1 = builder.add("algo_1", "x = 1\n", pass_result)
    s2 = builder.add("algo_2", "raise MemoryError()\n", fail_result)

    assert s1.label == 0
    assert s2.label == 1

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2


def test_perf_dataset_builder_load_roundtrip(tmp_path):
    out = tmp_path / "d_perf.jsonl"
    builder = PerfDatasetBuilder(out)
    result = ValidationResult(status=ValidationStatus.PASS)
    builder.add("a1", "pass\n", result)

    loaded = PerfDatasetBuilder.load(out)
    assert len(loaded) == 1
    assert loaded[0].algorithm_id == "a1"


# ---------------------------------------------------------------------------
# ProbabilisticFilter
# ---------------------------------------------------------------------------


def test_filter_no_model_returns_zero(tmp_path):
    filt = ProbabilisticFilter(model_path=tmp_path / "nonexistent.joblib")
    p = filt.predict_failure_probability("x = 1\n")
    assert p == 0.0


def test_filter_should_not_reject_below_threshold(tmp_path):
    filt = ProbabilisticFilter(model_path=tmp_path / "nonexistent.joblib")
    assert not filt.should_reject("x = 1\n", threshold=0.5)


def test_filter_with_mock_model():
    mock_model = MagicMock()
    mock_model.classes_ = [0, 1]
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    filt = ProbabilisticFilter.__new__(ProbabilisticFilter)
    filt._model = mock_model
    filt._model_path = Path("mock.joblib")

    p = filt.predict_failure_probability("x = 1\n")
    assert abs(p - 0.8) < 1e-6
    assert filt.should_reject("x = 1\n", threshold=0.7)
    assert not filt.should_reject("x = 1\n", threshold=0.9)
