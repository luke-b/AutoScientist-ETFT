"""
corpus/performance_estimator/filter_model.py — Probabilistic Heuristic Filter.

Uses a scikit-learn RandomForestClassifier to estimate P(failure) for a
candidate algorithm before committing GPU resources.

The model is trained on 𝒟_Perf (see train_filter.py) and persisted to disk
as a joblib file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from corpus.performance_estimator.feature_extractor import extract_features, feature_names

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("checkpoints/perf_filter.joblib")


class ProbabilisticFilter:
    """
    Wraps a trained sklearn classifier that predicts P(failure) for a
    candidate algorithm given its structural features.

    Parameters
    ----------
    model_path:
        Path to a joblib-serialised sklearn pipeline/estimator.
        If the file does not exist, the filter falls back to a safe default
        (P(failure) = 0.0 — always pass) so the pipeline can still run
        without a trained model.
    """

    def __init__(self, model_path: Path | str = _DEFAULT_MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._model: Any = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "No trained filter found at %s — using pass-through (P=0.0).",
                self._model_path,
            )
            return
        try:
            import joblib

            self._model = joblib.load(self._model_path)
            logger.info("Loaded probabilistic filter from %s", self._model_path)
        except Exception as exc:
            logger.error("Failed to load filter model: %s", exc)

    # ------------------------------------------------------------------
    def predict_failure_probability(self, code: str) -> float:
        """
        Return P(failure) ∈ [0, 1] for *code*.

        0.0 = almost certainly safe, 1.0 = almost certainly fails.
        Falls back to 0.0 if no model is available.
        """
        if self._model is None:
            return 0.0

        features = extract_features(code)
        names = feature_names()
        X = np.array([[features.get(n, 0.0) for n in names]])  # noqa: N806

        try:
            proba = self._model.predict_proba(X)
            # Class 1 = failure
            classes = list(self._model.classes_)
            fail_idx = classes.index(1) if 1 in classes else -1
            return float(proba[0][fail_idx]) if fail_idx >= 0 else 0.0
        except Exception as exc:
            logger.error("Filter prediction error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    def should_reject(self, code: str, threshold: float = 0.7) -> bool:
        """Return True if the candidate should be rejected (P(fail) > threshold)."""
        return self.predict_failure_probability(code) > threshold
