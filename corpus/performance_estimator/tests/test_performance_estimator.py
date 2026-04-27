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


# ---------------------------------------------------------------------------
# TF-IDF semantic features
# ---------------------------------------------------------------------------


def test_tfidf_features_present_in_names():
    """feature_names() must now include TF-IDF feature names for all ML vocab terms."""
    from corpus.performance_estimator.feature_extractor import _ML_VOCAB

    names = feature_names()
    for term in _ML_VOCAB:
        assert f"tfidf_{term}" in names, f"Expected 'tfidf_{term}' in feature_names()"


def test_tfidf_feature_count():
    """Total features = 16 AST + len(_ML_VOCAB) TF-IDF."""
    from corpus.performance_estimator.feature_extractor import _ML_VOCAB

    expected = 16 + len(_ML_VOCAB)
    assert len(feature_names()) == expected


def test_tfidf_nonzero_for_matching_token():
    """Code containing a vocab token should produce a non-zero TF-IDF feature."""
    code = "optimizer = 'adam'\nprint(optimizer)\n"
    feats = extract_features(code)
    assert feats["tfidf_adam"] > 0.0 or feats["tfidf_optimizer"] > 0.0


def test_tfidf_zero_for_absent_token():
    """Code that doesn't mention a vocab term should produce 0 for that feature."""
    code = "x = 1\n"
    feats = extract_features(code)
    assert feats["tfidf_cuda"] == 0.0


def test_tfidf_sublinear_tf():
    """TF-IDF value should be 1+log(count) for count>0 (sublinear scaling)."""
    import math

    # Valid Python code that mentions 'adam' exactly 3 times as a bare token
    code = "adam = 1\nprint(adam)\nx = adam + 1\n"
    feats = extract_features(code)
    expected = 1.0 + math.log(3)
    assert abs(feats["tfidf_adam"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Filter version / schema compatibility
# ---------------------------------------------------------------------------


def test_filter_rejects_mismatched_version(tmp_path):
    """A model saved with a different feature version must be rejected (pass-through)."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model_path = tmp_path / "wrong_version.joblib"
    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    clf.fit([[0] * 16], [0])  # trained on 16 old AST features
    payload = {
        "model": clf,
        "feature_names": [f"feat_{i}" for i in range(16)],
        "version": 1,  # old version
    }
    joblib.dump(payload, model_path)

    filt = ProbabilisticFilter(model_path=model_path)
    # Must fall back to pass-through (P=0.0) since version mismatches
    assert filt._model is None
    assert filt.predict_failure_probability("x = 1\n") == 0.0


def test_filter_train_and_predict_with_versioned_payload(tmp_path):
    """Train, save, and reload a filter; prediction must return a valid probability."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from corpus.performance_estimator.filter_model import CURRENT_FEATURE_VERSION

    names = feature_names()
    # Two synthetic samples — one pass (label=0), one fail (label=1)
    pass_code = "def f(): return 1\n"
    fail_code = "\n".join(["import torch"] * 5 + ["x = torch.cuda.alloc(1 << 30)\n"])
    X = np.array([  # noqa: N806
        [extract_features(pass_code).get(n, 0.0) for n in names],
        [extract_features(fail_code).get(n, 0.0) for n in names],
    ])
    y = np.array([0, 1])

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipeline.fit(X, y)

    model_path = tmp_path / "filter_v2.joblib"
    payload = {"model": pipeline, "feature_names": names, "version": CURRENT_FEATURE_VERSION}
    joblib.dump(payload, model_path)

    filt = ProbabilisticFilter(model_path=model_path)
    assert filt._model is not None
    p = filt.predict_failure_probability(pass_code)
    assert 0.0 <= p <= 1.0
