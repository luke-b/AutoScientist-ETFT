"""
corpus/performance_estimator/train_filter.py — Train the Probabilistic
Heuristic Filter (RandomForest) on 𝒟_Perf.

Usage:
    python -m corpus.performance_estimator.train_filter \\
        --data ./data/d_perf --output ./checkpoints/perf_filter.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from corpus.performance_estimator.dataset_builder import PerfDatasetBuilder
from corpus.performance_estimator.feature_extractor import feature_names

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train(data_dir: Path, output_path: Path, cfg: dict | None = None) -> None:
    """Load all 𝒟_Perf JSONL files and train a RandomForestClassifier."""
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("scikit-learn / joblib not installed. Run: pip install scikit-learn joblib") from exc

    estimator_cfg = (cfg or {}).get("corpus", {}).get("performance_estimator", {})
    test_size = float(estimator_cfg.get("test_size", 0.2))
    random_state = int(estimator_cfg.get("random_state", 42))
    n_estimators = int(estimator_cfg.get("n_estimators", 200))

    # Load samples
    samples = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        samples.extend(PerfDatasetBuilder.load(jsonl_file))

    if len(samples) < 10:
        raise SystemExit(
            f"Only {len(samples)} samples found in {data_dir}. "
            "Need at least 10 to train the filter."
        )

    names = feature_names()
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in samples])  # noqa: N806
    y = np.array([s.label for s in samples])

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])

    logger.info("Training RandomForest on %d samples …", len(X_train))
    pipeline.fit(X_train, y_train)

    report = classification_report(y_test, pipeline.predict(X_test), target_names=["pass", "fail"])
    logger.info("Evaluation on %d test samples:\n%s", len(X_test), report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Model saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ETFT Probabilistic Heuristic Filter.")
    parser.add_argument("--data", default="./data/d_perf", help="Directory with 𝒟_Perf JSONL files.")
    parser.add_argument("--output", default="./checkpoints/perf_filter.joblib", help="Output model path.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    from etft.config import load_config
    args = parse_args()
    cfg = load_config(args.config)
    train(Path(args.data), Path(args.output), cfg)


if __name__ == "__main__":
    main()
