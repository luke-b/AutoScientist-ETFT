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

# Blind-mode template — omits exact delta values and causal explanation to
# test how much information is leaked by the hints (see A3 plan item).
_REPLAY_TEMPLATE_BLIND = """\
=== PREDECESSOR ALGORITHM (a_{before_idx}) ===
{before_code}

=== PERFORMANCE DELTA TO ACHIEVE ===
ΔLoss    : {delta_loss_ordinal}
ΔMemory  : {delta_memory_ordinal}

Reconstruct the successor algorithm a_{after_idx} that achieves the above deltas.

```python
<reconstructed code here>
```
"""

_ORDINAL_THRESHOLDS = [
    (0.01, "negligible"),
    (0.05, "small"),
    (0.15, "moderate"),
    (0.30, "large"),
]


def _to_ordinal(value) -> str:
    """Convert a numeric delta to an ordinal description for blind mode."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "unspecified"
    for threshold, label in _ORDINAL_THRESHOLDS:
        if v <= threshold:
            return label
    return "very large"


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
    max_code_chars:
        Maximum number of characters from ``before.code`` included in the
        prompt.  When the code is longer it is truncated and a warning is
        logged.  Configurable via ``calibration.max_code_chars`` in
        ``config.yaml``.  Defaults to 8000.
    blind_mode:
        When *True*, the prompt omits exact delta values and the causal
        explanation, replacing them with ordinal descriptions (e.g.
        "moderate improvement in loss").  Useful for quantifying how much
        information leakage occurs when hints are provided.
        Configurable via ``calibration.blind_mode`` in ``config.yaml``.
    max_step_gap:
        Maximum allowed index gap between adjacent step pairs.  Pairs with
        a larger gap are skipped with a warning.  ``None`` means no limit.
        Configurable via ``calibration.max_step_gap`` in ``config.yaml``.
    """

    agent: AgentClient
    similarity_fn: object = None  # Callable[[str, str], float]
    max_code_chars: int = 8000
    blind_mode: bool = False
    max_step_gap: int | None = None
    results: list[ReconstructionResult] = field(default_factory=list, init=False)
    n_agent_failures: int = field(default=0, init=False)
    n_steps_attempted: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.similarity_fn is None:
            from calibration.similarity import code_similarity
            self.similarity_fn = code_similarity

    # ------------------------------------------------------------------
    def reconstruct_step(
        self,
        before: TrajectoryStep,
        after: TrajectoryStep,
    ) -> ReconstructionResult | None:
        """
        Ask the model to reconstruct *after* given *before* and the delta.

        Returns a ``ReconstructionResult`` with the predicted code and
        its similarity to the ground-truth successor, or ``None`` when the
        step pair is skipped due to a gap limit violation.

        When the agent call fails (``success=False``), a result with
        ``similarity_score=None`` and ``agent_success=False`` is stored in
        ``self._failed_results`` but is **not** appended to ``self.results``
        so it does not bias the mean confidence level.
        """
        # ---- C3: non-contiguous step gap check ----
        gap = after.step_index - before.step_index
        if gap > 1:
            logger.warning(
                "[replay] Non-contiguous step gap detected: %d→%d (gap=%d). "
                "Calibration covers a multi-step evolutionary leap.",
                before.step_index, after.step_index, gap,
            )
        if self.max_step_gap is not None and gap > self.max_step_gap:
            logger.warning(
                "[replay] Step gap %d exceeds max_step_gap=%d — skipping pair %d→%d.",
                gap, self.max_step_gap, before.step_index, after.step_index,
            )
            # Skipped pairs are not counted as attempted steps
            return None

        self.n_steps_attempted += 1

        # ---- B1: configurable truncation with warning ----
        code_for_prompt = before.code
        truncated = len(before.code) > self.max_code_chars
        if truncated:
            logger.warning(
                "[replay] Algorithm '%s' step %d code truncated from %d to %d chars "
                "for prompt (calibration.max_code_chars=%d).",
                before.algorithm_id, before.step_index,
                len(before.code), self.max_code_chars, self.max_code_chars,
            )
            code_for_prompt = before.code[:self.max_code_chars]

        # ---- Build prompt ----
        meta = after.metadata or {}
        delta_loss = meta.get("delta_loss", "unspecified")
        delta_memory = meta.get("delta_memory_mb", "unspecified")
        causal_hint_raw = meta.get("causal_explanation", "")

        if self.blind_mode:
            prompt = _REPLAY_TEMPLATE_BLIND.format(
                before_idx=before.step_index,
                before_code=code_for_prompt,
                after_idx=after.step_index,
                delta_loss_ordinal=_to_ordinal(delta_loss),
                delta_memory_ordinal=_to_ordinal(delta_memory),
            )
        else:
            causal_hint = f"Causal note: {causal_hint_raw}" if causal_hint_raw else ""
            prompt = _REPLAY_TEMPLATE.format(
                before_idx=before.step_index,
                before_code=code_for_prompt,
                after_idx=after.step_index,
                delta_loss=delta_loss,
                delta_memory=delta_memory,
                causal_hint=causal_hint,
            )

        agent_result = self.agent.run_task(
            task=prompt,
            system=_REPLAY_SYSTEM,
        )

        # ---- B2: agent failure isolation ----
        if not agent_result.success:
            logger.warning(
                "[replay] Agent call failed for step %d→%d (algorithm '%s'). "
                "Step excluded from confidence computation.",
                before.step_index, after.step_index, before.algorithm_id,
            )
            self.n_agent_failures += 1
            # Store a failure record but do NOT append to self.results
            return ReconstructionResult(
                trajectory_id=before.algorithm_family,
                step_index_before=before.step_index,
                step_index_after=after.step_index,
                predicted_code="",
                true_code=after.code,
                similarity_score=None,
                agent_success=False,
                metadata={
                    "agent_failure": True,
                    "truncated": truncated,
                    "step_gap": gap,
                },
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
            agent_success=True,
            metadata={
                "truncated": truncated,
                "step_gap": gap,
            },
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

        Only ``"calibration"``-split steps are replayed when any step carries
        a ``split`` field (A1 held-out support).  If no steps have
        ``split="calibration"``, all steps are used (backwards-compatible).

        Parameters
        ----------
        steps:
            Full list of ``TrajectoryStep`` objects for one trajectory.

        Returns
        -------
        list[ReconstructionResult]
            One result per successfully evaluated adjacent pair.
            Agent-failure results are NOT included here; inspect
            ``self.n_agent_failures`` to account for them.
        """
        sorted_steps = sorted(steps, key=lambda s: s.step_index)
        self.results = []
        self.n_agent_failures = 0
        self.n_steps_attempted = 0

        if len(sorted_steps) < 2:
            logger.warning("Trajectory has fewer than 2 steps — no pairs to replay.")
            return self.results

        # A1: honour the held-out split — only replay "calibration"-tagged steps
        cal_steps = [s for s in sorted_steps if getattr(s, "split", "train") == "calibration"]
        if cal_steps:
            logger.info(
                "[replay] Using %d held-out 'calibration' steps (A1 held-out split).",
                len(cal_steps),
            )
            sorted_steps = cal_steps

        for i in range(1, len(sorted_steps)):
            self.reconstruct_step(sorted_steps[i - 1], sorted_steps[i])

        return self.results

    # ------------------------------------------------------------------
    @property
    def confidence_level(self) -> float:
        """
        C = (1/n) Σ Sim(a_pred_i, a_true_i)   [Eq. 1, Benda 2026]

        Only steps with a valid (non-None) similarity score are counted.
        Returns 0.0 when no steps have been replayed yet.
        """
        scored = [r.similarity_score for r in self.results if r.similarity_score is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)
