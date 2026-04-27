"""
corpus/regression_pipeline/fitness_evaluator.py — Measures the fitness score of a
generated predecessor algorithm by running it as a micro-experiment.

If the script emits ``METRIC: <name>=<value>`` lines, the first recognised metric
is used as the fitness score.  If execution fails (or no METRIC lines are found),
the evaluator falls back to the ``0.85 * current_fitness`` heuristic used by the
regression pipeline before this module existed.
"""

from __future__ import annotations

import logging

from agents.empirical.runner import ExperimentRunner

logger = logging.getLogger(__name__)

_FALLBACK_FACTOR = 0.85


class FitnessEvaluator:
    """
    Evaluates the fitness of a generated algorithm by executing it as a
    micro-experiment and reading ``METRIC:`` lines from stdout.

    Parameters
    ----------
    cfg:
        Full config dict (forwarded to ``ExperimentRunner``).
    primary_metric:
        The metric name to use as the fitness score.  Falls back to the
        first metric found in stdout if this name is not present.
    fallback_factor:
        Multiplier applied to ``current_fitness`` when execution fails.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        primary_metric: str = "fitness",
        fallback_factor: float = _FALLBACK_FACTOR,
    ) -> None:
        self._runner = ExperimentRunner(cfg)
        self._primary_metric = primary_metric
        self._fallback_factor = fallback_factor

    # ------------------------------------------------------------------
    def evaluate(self, code: str, current_fitness: float) -> float:
        """
        Run *code* and return a fitness score.

        The algorithm code is expected to print at least one
        ``METRIC: <name>=<value>`` line.  If the primary metric is not
        found, the first available metric value is used.  If execution
        fails entirely, ``current_fitness * fallback_factor`` is returned.

        Parameters
        ----------
        code:
            Full Python source of the predecessor algorithm.
        current_fitness:
            Fitness of the algorithm being de-optimised (used for the
            fallback calculation).

        Returns
        -------
        float
            Measured or estimated fitness of *code*.
        """
        result = self._runner.run(code)

        if not result.success:
            logger.warning(
                "FitnessEvaluator: execution failed (%s) — using fallback factor %.2f.",
                result.error_message,
                self._fallback_factor,
            )
            return current_fitness * self._fallback_factor

        metrics = result.metrics
        if not metrics:
            logger.warning(
                "FitnessEvaluator: no METRIC lines in stdout — using fallback factor %.2f.",
                self._fallback_factor,
            )
            return current_fitness * self._fallback_factor

        if self._primary_metric in metrics:
            score = metrics[self._primary_metric]
        else:
            # Use the first metric value found
            score = next(iter(metrics.values()))
            logger.debug(
                "FitnessEvaluator: primary metric %r not found; using %r=%.4f.",
                self._primary_metric,
                next(iter(metrics)),
                score,
            )

        logger.info("FitnessEvaluator: measured fitness=%.4f.", score)
        return score
