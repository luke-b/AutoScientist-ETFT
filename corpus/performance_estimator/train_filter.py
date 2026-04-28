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
from corpus.performance_estimator.filter_model import CURRENT_FEATURE_VERSION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train(data_dir: Path, output_path: Path, cfg: dict | None = None, bootstrap: bool = False) -> None:
    """
    Load all 𝒟_Perf JSONL files and train a RandomForestClassifier.

    Parameters
    ----------
    data_dir:
        Directory containing ``*.jsonl`` 𝒟_Perf files.
    output_path:
        Destination path for the serialised model payload.
    cfg:
        Optional runtime configuration dict.
    bootstrap:
        When *True* and *data_dir* contains fewer than 10 samples,
        synthetic bootstrap data is generated automatically before training.
    """
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

    # Auto-bootstrap if requested and data is sparse
    if bootstrap:
        existing = []
        if data_dir.exists():
            for jf in sorted(data_dir.glob("*.jsonl")):
                existing.extend(PerfDatasetBuilder.load(jf))
        if len(existing) < 10:
            from corpus.performance_estimator.bootstrap import generate_bootstrap_data
            bootstrap_path = data_dir / "bootstrap.jsonl"
            logger.info(
                "Bootstrap flag set and only %d existing samples found — "
                "generating synthetic bootstrap data at %s",
                len(existing), bootstrap_path,
            )
            generate_bootstrap_data(bootstrap_path, n_samples=100)

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
    payload = {
        "model": pipeline,
        "feature_names": names,
        "version": CURRENT_FEATURE_VERSION,
    }
    joblib.dump(payload, output_path)
    logger.info("Model saved to %s (feature_version=%d)", output_path, CURRENT_FEATURE_VERSION)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ETFT Probabilistic Heuristic Filter.")
    parser.add_argument("--data", default="./data/d_perf", help="Directory with 𝒟_Perf JSONL files.")
    parser.add_argument("--output", default="./checkpoints/perf_filter.joblib", help="Output model path.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help=(
            "Auto-generate synthetic 𝒟_Perf bootstrap data when the data directory "
            "contains fewer than 10 samples before training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    from etft.config import load_config
    args = parse_args()
    cfg = load_config(args.config)
    train(Path(args.data), Path(args.output), cfg, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
