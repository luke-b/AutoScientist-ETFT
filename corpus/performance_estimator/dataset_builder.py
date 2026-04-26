"""
corpus/performance_estimator/dataset_builder.py — Builds 𝒟_Perf from
CI/CD validation outcomes.

Each ValidationResult produced by the regression pipeline is converted into
a PerfSample (features + label) and appended to a JSONL file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corpus.performance_estimator.feature_extractor import extract_features
from corpus.regression_pipeline.schemas import PerfSample, ValidationResult

logger = logging.getLogger(__name__)


class PerfDatasetBuilder:
    """Accumulates PerfSample records into a JSONL 𝒟_Perf file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def add(self, algorithm_id: str, code: str, result: ValidationResult) -> PerfSample:
        """
        Extract features from *code*, assign a label from *result*, and
        append to the JSONL file.

        Returns the created PerfSample.
        """
        features = extract_features(code)
        label = 0 if result.passed else 1
        failure_reason = None if result.passed else result.status.value

        sample = PerfSample(
            algorithm_id=algorithm_id,
            features=features,
            label=label,
            failure_reason=failure_reason,
            metadata={
                "duration_seconds": result.duration_seconds,
                "peak_memory_mb": result.peak_memory_mb,
            },
        )
        self._append(sample)
        return sample

    # ------------------------------------------------------------------
    def _append(self, sample: PerfSample) -> None:
        with open(self.output_path, "a") as f:
            f.write(sample.model_dump_json() + "\n")
        logger.debug("PerfSample written: %s label=%d", sample.algorithm_id, sample.label)

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: Path) -> list[PerfSample]:
        samples: list[PerfSample] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(PerfSample.model_validate_json(line))
        return samples
