"""
corpus/performance_estimator/filter_model.py — Probabilistic Heuristic Filter.

Uses a scikit-learn RandomForestClassifier to estimate P(failure) for a
candidate algorithm before committing GPU resources.

The model is trained on 𝒟_Perf (see train_filter.py) and persisted to disk
as a joblib file containing a dict with the trained pipeline and schema metadata.

Schema versioning
-----------------
``train_filter.py`` saves ``{"model": pipeline, "feature_names": [...], "version": int}``
so that loading can detect a feature-schema mismatch (e.g. after adding new
TF-IDF features) and fall back to pass-through rather than silently producing
wrong predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from corpus.performance_estimator.feature_extractor import extract_features, feature_names

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("checkpoints/perf_filter.joblib")

# Increment this constant whenever the feature schema changes.
# Models saved with a different version will be rejected on load.
CURRENT_FEATURE_VERSION = 2  # v1 = 16 AST features; v2 = 66 (AST + TF-IDF)


class ProbabilisticFilter:
    """
    Wraps a trained sklearn classifier that predicts P(failure) for a
    candidate algorithm given its structural features.

    Parameters
    ----------
    model_path:
        Path to a joblib-serialised payload (see module docstring).
        If the file does not exist, or if it was trained with a different
        feature schema version, the filter falls back to a safe default
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

            payload = joblib.load(self._model_path)
        except Exception as exc:
            logger.error("Failed to load filter model: %s", exc)
            return

        # Support both legacy (bare pipeline) and versioned (dict) payloads
        if isinstance(payload, dict):
            saved_version = payload.get("version", 1)
            saved_names = payload.get("feature_names", [])
            current_names = feature_names()
            if saved_version != CURRENT_FEATURE_VERSION or saved_names != current_names:
                logger.warning(
                    "Filter model at %s was trained with a different feature schema "
                    "(saved version=%s, current=%s; saved features=%d, current=%d). "
                    "Falling back to pass-through (P=0.0). Retrain the filter with "
                    "'python -m corpus.performance_estimator.train_filter'.",
                    self._model_path,
                    saved_version,
                    CURRENT_FEATURE_VERSION,
                    len(saved_names),
                    len(current_names),
                )
                return
            self._model = payload["model"]
        else:
            # Legacy bare-pipeline payload — check feature count heuristically
            current_names = feature_names()
            logger.warning(
                "Filter model at %s uses legacy format (no version metadata). "
                "Checking feature count compatibility.",
                self._model_path,
            )
            try:
                n_features = payload.n_features_in_
            except AttributeError:
                try:
                    n_features = payload.steps[-1][1].n_features_in_
                except Exception:
                    n_features = None

            if n_features is not None and n_features != len(current_names):
                logger.warning(
                    "Legacy filter expects %d features but current schema has %d. "
                    "Falling back to pass-through (P=0.0).",
                    n_features, len(current_names),
                )
                return
            self._model = payload

        logger.info("Loaded probabilistic filter from %s", self._model_path)

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
