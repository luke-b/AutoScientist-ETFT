"""
calibration/similarity.py — Similarity metrics for code reconstruction evaluation.

The Calibration Engine computes a Confidence Level C as the mean similarity
between a model's reconstructed algorithm and the true historical algorithm
for each step in the trajectory:

    C = (1/n) Σ Sim(a_pred_i, a_true_i)          [Eq. 1, Benda 2026]

This module provides multiple Sim() implementations of increasing fidelity:

- ``token_jaccard``   – Fast, bag-of-words token overlap (default)
- ``line_lcs``        – Longest-common-subsequence over source lines
- ``ast_edit``        – Tree-edit distance on Python AST nodes (structural)
- ``code_similarity`` – Composite metric combining all three

The composite metric is the recommended choice: it is robust to superficial
reformatting while still capturing structural equivalence.
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import Callable


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------


def _tokenise(code: str) -> list[str]:
    """Split code into normalised token bags (strips comments / whitespace)."""
    code = re.sub(r"#.*", "", code)              # strip line comments
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)  # strip docstrings
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    return re.findall(r"\w+", code.lower())


# ---------------------------------------------------------------------------
# Individual similarity functions
# ---------------------------------------------------------------------------


def token_jaccard(pred: str, true: str) -> float:
    """
    Jaccard similarity on normalised token bags.

    Returns a value in [0, 1] where 1 = identical token bags.
    Robust to reordering and minor renames; fast on large files.
    """
    pred_tokens = set(_tokenise(pred))
    true_tokens = set(_tokenise(true))
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    return len(pred_tokens & true_tokens) / len(pred_tokens | true_tokens)


def line_lcs(pred: str, true: str) -> float:
    """
    Normalised longest-common-subsequence similarity over source lines.

    Returns a value in [0, 1] where 1 = all lines match.
    Captures ordering and sequential structure.
    """
    pred_lines = pred.splitlines()
    true_lines = true.splitlines()
    if not pred_lines and not true_lines:
        return 1.0
    matcher = difflib.SequenceMatcher(None, pred_lines, true_lines)
    # ratio() = 2 * M / T where M = matching blocks, T = total elements
    return matcher.ratio()


def ast_edit(pred: str, true: str) -> float:
    """
    Structural similarity based on AST node-type sequences.

    Computes the sequence-match ratio of the flattened AST node type lists.
    Falls back to 0.0 if either code is syntactically invalid.
    Returns a value in [0, 1].
    """
    def _node_sequence(code: str) -> list[str]:
        try:
            tree = ast.parse(code)
            return [type(n).__name__ for n in ast.walk(tree)]
        except SyntaxError:
            return []

    pred_nodes = _node_sequence(pred)
    true_nodes = _node_sequence(true)

    if not pred_nodes and not true_nodes:
        return 1.0
    if not pred_nodes or not true_nodes:
        return 0.0

    matcher = difflib.SequenceMatcher(None, pred_nodes, true_nodes)
    return matcher.ratio()


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


def code_similarity(
    pred: str,
    true: str,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    """
    Composite code similarity: weighted combination of token Jaccard,
    line LCS, and AST structural similarity.

    Parameters
    ----------
    pred:
        The model's reconstructed code (a_pred_i).
    true:
        The ground-truth historical code (a_true_i).
    weights:
        (w_jaccard, w_lcs, w_ast) — must sum to 1.0.

    Returns
    -------
    float
        Composite similarity in [0, 1]; higher is better.
    """
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0

    w_j, w_l, w_a = weights
    score = (
        w_j * token_jaccard(pred, true)
        + w_l * line_lcs(pred, true)
        + w_a * ast_edit(pred, true)
    )
    return float(min(max(score, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Registry: named similarity functions for config-driven selection
# ---------------------------------------------------------------------------

SIMILARITY_FUNCTIONS: dict[str, Callable[[str, str], float]] = {
    "token_jaccard": token_jaccard,
    "line_lcs": line_lcs,
    "ast_edit": ast_edit,
    "composite": code_similarity,
}
