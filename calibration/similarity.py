"""
calibration/similarity.py — Similarity metrics for code reconstruction evaluation.

The Calibration Engine computes a Confidence Level C as the mean similarity
between a model's reconstructed algorithm and the true historical algorithm
for each step in the trajectory:

    C = (1/n) Σ Sim(a_pred_i, a_true_i)          [Eq. 1, Benda 2026]

This module provides multiple Sim() implementations of increasing fidelity:

- ``token_jaccard``    – Fast, bag-of-words token overlap
- ``line_lcs``         – Longest-common-subsequence over source lines
- ``ast_node_sequence``– Breadth-first AST node-type sequences (fast bag metric)
- ``ast_edit``         – Structurally-aware parent-annotated AST node sequences
                         (falls back to ``zss`` tree-edit distance when available)
- ``code_similarity``  – Composite metric combining all three

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


def ast_node_sequence(pred: str, true: str) -> float:
    """
    Structural similarity based on breadth-first AST node-type sequences.

    Computes the sequence-match ratio of the flattened (unordered walk) AST
    node-type lists.  This is a fast node-type bag metric — it captures the
    distribution of language constructs but loses parent-child relationships.

    Falls back to 0.0 if either code is syntactically invalid.
    Returns a value in [0, 1].

    .. note::
        For a structurally-aware metric that preserves at least one level of
        parent context, use :func:`ast_edit` instead.
    """
    def _flat_sequence(code: str) -> list[str]:
        try:
            tree = ast.parse(code)
            return [type(n).__name__ for n in ast.walk(tree)]
        except SyntaxError:
            return []

    pred_nodes = _flat_sequence(pred)
    true_nodes = _flat_sequence(true)

    if not pred_nodes and not true_nodes:
        return 1.0
    if not pred_nodes or not true_nodes:
        return 0.0

    matcher = difflib.SequenceMatcher(None, pred_nodes, true_nodes)
    return matcher.ratio()


def _parent_annotated_sequence(code: str) -> list[str]:
    """
    Build a parent-annotated node sequence from a Python AST.

    Each element is ``"ParentType::ChildType"`` (or just ``"ChildType"`` for the
    module root).  This preserves one level of structural context — e.g.
    ``"FunctionDef::Return"`` is distinguishable from ``"If::Return"`` — making
    the metric sensitive to structural differences that a flat bag misses.

    Returns an empty list on ``SyntaxError``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    sequence: list[str] = []
    for node in ast.walk(tree):
        parent_name = type(node).__name__
        for child in ast.iter_child_nodes(node):
            sequence.append(f"{parent_name}::{type(child).__name__}")
        # Also record the node itself so leaf nodes (no children) are counted
        sequence.append(type(node).__name__)
    return sequence


def _zss_tree_edit(pred: str, true: str) -> float | None:
    """
    Attempt a proper tree-edit distance using the ``zss`` library.

    Returns a normalised similarity in [0, 1], or ``None`` when ``zss`` is
    not installed so the caller can fall back to the parent-annotated metric.
    """
    try:
        import zss  # type: ignore[import]
    except ImportError:
        return None

    def _build_zss_tree(code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        def _convert(node) -> zss.Node:
            z = zss.Node(type(node).__name__)
            for child in ast.iter_child_nodes(node):
                z.addkid(_convert(child))
            return z

        return _convert(tree)

    pred_tree = _build_zss_tree(pred)
    true_tree = _build_zss_tree(true)
    if pred_tree is None and true_tree is None:
        return 1.0
    if pred_tree is None or true_tree is None:
        return 0.0

    try:
        dist = zss.simple_distance(pred_tree, true_tree)
        # Normalise by the sum of sizes so the result is in [0, 1]
        pred_size = sum(1 for _ in ast.walk(ast.parse(pred)))
        true_size = sum(1 for _ in ast.walk(ast.parse(true)))
        max_dist = pred_size + true_size  # upper bound: delete all + insert all
        if max_dist == 0:
            return 1.0
        return float(max(0.0, 1.0 - dist / max_dist))
    except Exception:
        return None


def ast_edit(pred: str, true: str) -> float:
    """
    Structurally-aware AST similarity.

    Uses a parent-annotated node-pair sequence (``"ParentType::ChildType"``) to
    preserve one level of tree structure context.  This is significantly more
    discriminative than the plain node-type bag produced by :func:`ast_node_sequence`.

    When the optional ``zss`` library is installed (``pip install zss``), a proper
    normalised tree-edit distance is used instead and the parent-annotated fallback
    is bypassed.

    Falls back to 0.0 if either code is syntactically invalid.
    Returns a value in [0, 1].
    """
    # Try proper tree-edit distance first (requires optional dependency)
    zss_score = _zss_tree_edit(pred, true)
    if zss_score is not None:
        return zss_score

    # Fallback: parent-annotated node sequence
    pred_nodes = _parent_annotated_sequence(pred)
    true_nodes = _parent_annotated_sequence(true)

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
        Configurable via ``calibration.similarity_weights`` in ``config.yaml``.

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
    "ast_node_sequence": ast_node_sequence,
    "ast_edit": ast_edit,
    "composite": code_similarity,
}
