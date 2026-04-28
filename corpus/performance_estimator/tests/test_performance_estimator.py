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


# ---------------------------------------------------------------------------
# Bootstrap data generator
# ---------------------------------------------------------------------------


def test_bootstrap_generates_expected_counts(tmp_path):
    """generate_bootstrap_data writes exactly 2 × n_samples records."""
    from corpus.performance_estimator.bootstrap import generate_bootstrap_data

    out = tmp_path / "bootstrap.jsonl"
    total = generate_bootstrap_data(out, n_samples=10)

    assert total == 20  # 10 safe + 10 risky
    assert out.exists()
    lines = [ln for ln in out.read_text().strip().splitlines() if ln]
    assert len(lines) == 20


def test_bootstrap_labels_balanced(tmp_path):
    """Bootstrap data contains equal counts of label=0 and label=1."""
    import json

    from corpus.performance_estimator.bootstrap import generate_bootstrap_data

    out = tmp_path / "bootstrap.jsonl"
    generate_bootstrap_data(out, n_samples=15)

    lines = [json.loads(ln) for ln in out.read_text().strip().splitlines() if ln]
    labels = [r["label"] for r in lines]
    assert labels.count(0) == 15
    assert labels.count(1) == 15


def test_bootstrap_features_non_empty(tmp_path):
    """Each bootstrap record must have a non-empty features dict."""
    import json

    from corpus.performance_estimator.bootstrap import generate_bootstrap_data

    out = tmp_path / "bootstrap.jsonl"
    generate_bootstrap_data(out, n_samples=5)

    for line in out.read_text().strip().splitlines():
        record = json.loads(line)
        assert record["features"], f"Empty features in: {record['algorithm_id']}"


def test_bootstrap_can_train_filter(tmp_path):
    """A filter trained on bootstrap data achieves > 60 % accuracy on held-out data."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    from corpus.performance_estimator.bootstrap import generate_bootstrap_data
    from corpus.performance_estimator.dataset_builder import PerfDatasetBuilder
    from corpus.performance_estimator.feature_extractor import feature_names
    from corpus.performance_estimator.train_filter import train

    d_perf_dir = tmp_path / "d_perf"
    d_perf_dir.mkdir()
    generate_bootstrap_data(d_perf_dir / "bootstrap.jsonl", n_samples=50)

    output_path = tmp_path / "perf_filter.joblib"
    train(d_perf_dir, output_path)

    assert output_path.exists()

    # Evaluate
    import joblib
    payload = joblib.load(output_path)
    model = payload["model"]
    names = feature_names()

    samples = PerfDatasetBuilder.load(d_perf_dir / "bootstrap.jsonl")
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in samples])  # noqa: N806
    y = np.array([s.label for s in samples])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)  # noqa: N806

    accuracy = (model.predict(X_test) == y_test).mean()
    assert accuracy > 0.60, f"Filter accuracy {accuracy:.2%} below 60% threshold."


def test_train_filter_bootstrap_flag_generates_data(tmp_path):
    """train(..., bootstrap=True) auto-generates data when dir is empty."""
    from corpus.performance_estimator.train_filter import train

    empty_dir = tmp_path / "d_perf_empty"
    empty_dir.mkdir()
    output_path = tmp_path / "model.joblib"

    train(empty_dir, output_path, bootstrap=True)

    # Bootstrap data should have been created
    assert (empty_dir / "bootstrap.jsonl").exists()
    assert output_path.exists()
