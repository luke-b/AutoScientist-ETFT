"""
calibration/replay.py — ReplaySession: model-driven trajectory reconstruction.

The Evolutionary Replay protocol (§2, Benda 2026) asks the model to
re-run the history of an algorithm trajectory before it may extrapolate
toward SOTA+1.  For each adjacent pair (a_{i-1}, a_i) in the trajectory
the model receives a_{i-1} and the causal meta-data M_i (ΔLoss, ΔMemory)
and must synthesise the functional successor a_i.

A ``ReplaySession`` encapsulates one complete run of this protocol over an
entire trajectory and exposes the per-step similarity scores together with
the resulting Confidence Level C.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from corpus.regression_pipeline.schemas import ReconstructionResult, TrajectoryStep
from etft.agent import AgentClient

logger = logging.getLogger(__name__)

_REPLAY_SYSTEM = (
    "You are an elite ML researcher. "
    "Given a simpler algorithm and the performance improvements achieved by its successor "
    "(described as ΔLoss and ΔMemory), reconstruct the successor algorithm in Python. "
    "Respond ONLY with a Python code block."
)

_REPLAY_TEMPLATE = """\
=== PREDECESSOR ALGORITHM (a_{before_idx}) ===
{before_code}

=== PERFORMANCE DELTA TO ACHIEVE ===
ΔLoss    : {delta_loss}
ΔMemory  : {delta_memory}
{causal_hint}

Reconstruct the successor algorithm a_{after_idx} that achieves the above deltas.

```python
<reconstructed code here>
```
"""


def _extract_code(text: str) -> str:
    """Extract the first Python fenced code block from an LLM response."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole response stripped of markdown fences
    return re.sub(r"```[a-z]*", "", text).strip()


@dataclass
class ReplaySession:
    """
    Runs the Evolutionary Replay protocol over one trajectory.

    Parameters
    ----------
    agent:
        An initialised ``AgentClient`` used for reconstruction.
    similarity_fn:
        Callable(pred: str, true: str) -> float in [0, 1].
        Defaults to the composite similarity metric when not provided.
    """

    agent: AgentClient
    similarity_fn: object = None  # Callable[[str, str], float]
    results: list[ReconstructionResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.similarity_fn is None:
            from calibration.similarity import code_similarity
            self.similarity_fn = code_similarity

    # ------------------------------------------------------------------
    def reconstruct_step(
        self,
        before: TrajectoryStep,
        after: TrajectoryStep,
    ) -> ReconstructionResult:
        """
        Ask the model to reconstruct *after* given *before* and the delta.

        Returns a ``ReconstructionResult`` with the predicted code and
        its similarity to the ground-truth successor.
        """
        # Build causal meta-data from trajectory step metadata
        meta = after.metadata or {}
        delta_loss = meta.get("delta_loss", "unspecified")
        delta_memory = meta.get("delta_memory_mb", "unspecified")
        causal_hint_raw = meta.get("causal_explanation", "")
        causal_hint = f"Causal note: {causal_hint_raw}" if causal_hint_raw else ""

        prompt = _REPLAY_TEMPLATE.format(
            before_idx=before.step_index,
            before_code=before.code[:4000],
            after_idx=after.step_index,
            delta_loss=delta_loss,
            delta_memory=delta_memory,
            causal_hint=causal_hint,
        )

        agent_result = self.agent.run_task(
            task=prompt,
            system=_REPLAY_SYSTEM,
        )
        predicted_code = _extract_code(agent_result.output)
        sim = float(self.similarity_fn(predicted_code, after.code))

        result = ReconstructionResult(
            trajectory_id=before.algorithm_family,
            step_index_before=before.step_index,
            step_index_after=after.step_index,
            predicted_code=predicted_code,
            true_code=after.code,
            similarity_score=sim,
            agent_success=agent_result.success,
        )
        logger.info(
            "Replay step %d→%d: similarity=%.3f",
            before.step_index, after.step_index, sim,
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    def run(self, steps: list[TrajectoryStep]) -> list[ReconstructionResult]:
        """
        Replay all adjacent pairs in *steps* (sorted by step_index).

        Parameters
        ----------
        steps:
            Full list of ``TrajectoryStep`` objects for one trajectory.

        Returns
        -------
        list[ReconstructionResult]
            One result per adjacent pair (n-1 results for n steps).
        """
        sorted_steps = sorted(steps, key=lambda s: s.step_index)
        self.results = []

        if len(sorted_steps) < 2:
            logger.warning("Trajectory has fewer than 2 steps — no pairs to replay.")
            return self.results

        for i in range(1, len(sorted_steps)):
            self.reconstruct_step(sorted_steps[i - 1], sorted_steps[i])

        return self.results

    # ------------------------------------------------------------------
    @property
    def confidence_level(self) -> float:
        """
        C = (1/n) Σ Sim(a_pred_i, a_true_i)   [Eq. 1, Benda 2026]

        Returns 0.0 when no steps have been replayed yet.
        """
        if not self.results:
            return 0.0
        return sum(r.similarity_score for r in self.results) / len(self.results)
