"""
dataset_builder.py — Assembles 𝒟_Gen and 𝒟_Rationale from regression trajectories.

Each trajectory is a list of TrajectoryStep objects ordered by increasing fitness.
The builder:
  1. Validates each step with the CI/CD validator.
  2. Extracts adjacent pairs → 𝒟_Gen JSONL.
  3. Uses an LLM to generate rationale for each transition → 𝒟_Rationale JSONL.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from corpus.regression_pipeline.cicd_validator import CICDValidator
from corpus.regression_pipeline.schemas import (
    RationaleRecord,
    TrajectoryPair,
    TrajectoryStep,
)
from etft.agent import AgentClient
from etft.skills.code_skills import SyntaxCheckSkill, ValidateCodeSkill

logger = logging.getLogger(__name__)

DEFAULT_SKILLS = [ValidateCodeSkill(), SyntaxCheckSkill()]

_RATIONALE_SYSTEM = (
    "You are an expert in machine learning architecture design. "
    "Analyse two versions of an algorithm and explain, in plain language, "
    "what changed and why the newer version is better."
)

_RATIONALE_TEMPLATE = """=== BEFORE ===
{before}

=== AFTER ===
{after}

Describe the architectural changes made (list the changed components) and estimate
the fractional performance gain (0.0–1.0) introduced by this transition.

Respond in JSON with this schema:
{{
  "delta_summary": "<string>",
  "changed_components": ["<component1>", ...],
  "performance_impact": <float 0-1>
}}
"""


def _parse_rationale_json(text: str) -> dict:
    """Extract and parse JSON from LLM output (may be wrapped in fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"delta_summary": text.strip(), "changed_components": [], "performance_impact": 0.0}


class DatasetBuilder:
    """Builds 𝒟_Gen and 𝒟_Rationale from a validated trajectory."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        self._validator = CICDValidator(cfg)
        self._agent = AgentClient(cfg, skills=DEFAULT_SKILLS)

    # ------------------------------------------------------------------
    def build_from_trajectory(
        self,
        steps: list[TrajectoryStep],
        output_dir: Path,
        trajectory_id: str | None = None,
    ) -> tuple[Path, Path]:
        """
        Process a list of TrajectorySteps and write JSONL files.

        Returns
        -------
        tuple[Path, Path]
            Paths to the written (𝒟_Gen file, 𝒟_Rationale file).
        """
        tid = trajectory_id or str(uuid.uuid4())
        output_dir.mkdir(parents=True, exist_ok=True)

        d_gen_path = output_dir / f"{tid}_d_gen.jsonl"
        d_rationale_path = output_dir / f"{tid}_d_rationale.jsonl"

        # Sort by fitness (ascending = oldest first)
        sorted_steps = sorted(steps, key=lambda s: s.fitness_score)

        d_gen_records: list[dict] = []
        d_rationale_records: list[dict] = []

        for i in range(1, len(sorted_steps)):
            before = sorted_steps[i - 1]
            after = sorted_steps[i]

            # Validate both steps
            val_before = self._validator.validate(before.code)
            val_after = self._validator.validate(after.code)

            if not val_before.passed or not val_after.passed:
                logger.warning(
                    "Skipping pair (%s → %s): validation failed", before.algorithm_id, after.algorithm_id
                )
                continue

            pair = TrajectoryPair(
                trajectory_id=tid,
                step_before=before,
                step_after=after,
            )
            d_gen_records.append(pair.to_training_example())

            # Generate rationale
            rationale_data = self._generate_rationale(before.code, after.code)
            rationale = RationaleRecord(
                trajectory_id=tid,
                step_index_before=before.step_index,
                step_index_after=after.step_index,
                **rationale_data,
            )
            d_rationale_records.append(rationale.to_training_example())

        _write_jsonl(d_gen_path, d_gen_records)
        _write_jsonl(d_rationale_path, d_rationale_records)

        logger.info(
            "Built trajectory %s: %d gen pairs, %d rationale records",
            tid,
            len(d_gen_records),
            len(d_rationale_records),
        )
        return d_gen_path, d_rationale_path

    # ------------------------------------------------------------------
    def _generate_rationale(self, before_code: str, after_code: str) -> dict:
        task = (
            "Analyse the two algorithm versions (BEFORE and AFTER) and explain what "
            "changed and why the newer version is better. Respond in JSON with this schema: "
            '{"delta_summary": "<string>", "changed_components": ["<component1>", ...], '
            '"performance_impact": <float 0-1>}'
        )
        result = self._agent.run_task(
            task=task,
            context={"before_code": before_code, "after_code": after_code},
            system=_RATIONALE_SYSTEM,
        )
        return _parse_rationale_json(result.output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.debug("Wrote %d records to %s", len(records), path)
