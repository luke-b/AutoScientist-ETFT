"""
calibration/tests/test_similarity.py — Unit tests for similarity metrics.
"""

import pytest
from calibration.similarity import (
    ast_edit,
    ast_node_sequence,
    code_similarity,
    line_lcs,
    token_jaccard,
)

_CODE_A = """\
def train(x, y, lr=0.01):
    weights = [0.0] * len(x[0])
    for epoch in range(100):
        for xi, yi in zip(x, y):
            pred = sum(w * v for w, v in zip(weights, xi))
            err = yi - pred
            weights = [w + lr * err * v for w, v in zip(weights, xi)]
    return weights
"""

_CODE_B = """\
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)
"""

_CODE_NEAR_IDENTICAL = _CODE_A + "\n# added comment\n"

# Two structurally different snippets that share the same *flat* node-type bag.
# A bare ``return x`` and a ``return x`` inside an if-block both produce
# 'Return' and 'Name' nodes, but their parent context differs.
_CODE_BARE_RETURN = """\
def f(x):
    return x
"""

_CODE_CONDITIONAL_RETURN = """\
def f(x):
    if x:
        return x
    return None
"""


class TestTokenJaccard:
    def test_identical(self):
        assert token_jaccard(_CODE_A, _CODE_A) == pytest.approx(1.0)

    def test_empty_both(self):
        assert token_jaccard("", "") == pytest.approx(1.0)

    def test_empty_one(self):
        assert token_jaccard(_CODE_A, "") == pytest.approx(0.0)

    def test_dissimilar(self):
        score = token_jaccard(_CODE_A, _CODE_B)
        assert 0.0 <= score < 0.5

    def test_near_identical(self):
        score = token_jaccard(_CODE_A, _CODE_NEAR_IDENTICAL)
        assert score > 0.8


class TestLineLCS:
    def test_identical(self):
        assert line_lcs(_CODE_A, _CODE_A) == pytest.approx(1.0)

    def test_empty_both(self):
        assert line_lcs("", "") == pytest.approx(1.0)

    def test_empty_one(self):
        assert line_lcs(_CODE_A, "") == pytest.approx(0.0)

    def test_near_identical(self):
        score = line_lcs(_CODE_A, _CODE_NEAR_IDENTICAL)
        assert score > 0.85


class TestASTNodeSequence:
    """Tests for the (fast, flat) ast_node_sequence metric."""

    def test_identical(self):
        assert ast_node_sequence(_CODE_A, _CODE_A) == pytest.approx(1.0)

    def test_invalid_syntax(self):
        assert ast_node_sequence("def broken(", _CODE_A) == pytest.approx(0.0)

    def test_empty_both(self):
        assert ast_node_sequence("", "") == pytest.approx(1.0)

    def test_structurally_similar(self):
        code_c = """\
def train(x, y, lr=0.001):
    weights = [0.0] * len(x[0])
    for epoch in range(50):
        for xi, yi in zip(x, y):
            pred = sum(w * v for w, v in zip(weights, xi))
            err = yi - pred
            weights = [w + lr * err * v for w, v in zip(weights, xi)]
    return weights
"""
        score = ast_node_sequence(_CODE_A, code_c)
        assert score > 0.85

    def test_in_unit_range(self):
        assert 0.0 <= ast_node_sequence(_CODE_A, _CODE_B) <= 1.0


class TestASTEdit:
    """Tests for the structurally-aware (parent-annotated) ast_edit metric."""

    def test_identical(self):
        assert ast_edit(_CODE_A, _CODE_A) == pytest.approx(1.0)

    def test_invalid_syntax(self):
        assert ast_edit("def broken(", _CODE_A) == pytest.approx(0.0)

    def test_empty_both(self):
        assert ast_edit("", "") == pytest.approx(1.0)

    def test_structurally_similar(self):
        code_c = """\
def train(x, y, lr=0.001):
    weights = [0.0] * len(x[0])
    for epoch in range(50):
        for xi, yi in zip(x, y):
            pred = sum(w * v for w, v in zip(weights, xi))
            err = yi - pred
            weights = [w + lr * err * v for w, v in zip(weights, xi)]
    return weights
"""
        score = ast_edit(_CODE_A, code_c)
        assert score > 0.85

    def test_in_unit_range(self):
        assert 0.0 <= ast_edit(_CODE_A, _CODE_B) <= 1.0

    def test_structural_sensitivity(self):
        """
        ast_edit must distinguish structurally different code that shares the
        same flat node-type bag.  A bare 'return x' and a 'return x' inside
        an 'if' block should score below the identical-structure baseline.
        """
        score_identical = ast_edit(_CODE_BARE_RETURN, _CODE_BARE_RETURN)
        score_different = ast_edit(_CODE_BARE_RETURN, _CODE_CONDITIONAL_RETURN)
        assert score_identical > score_different, (
            "ast_edit should penalise structural differences; "
            f"got identical={score_identical:.3f}, different={score_different:.3f}"
        )


class TestCodeSimilarity:
    def test_identical(self):
        assert code_similarity(_CODE_A, _CODE_A) == pytest.approx(1.0)

    def test_empty_both(self):
        assert code_similarity("", "") == pytest.approx(1.0)

    def test_empty_one(self):
        assert code_similarity(_CODE_A, "") == pytest.approx(0.0)

    def test_dissimilar_lower_than_identical(self):
        score_same = code_similarity(_CODE_A, _CODE_A)
        score_diff = code_similarity(_CODE_A, _CODE_B)
        assert score_same > score_diff

    def test_in_unit_range(self):
        score = code_similarity(_CODE_A, _CODE_B)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        """Weights are accepted and the result remains in [0, 1]."""
        score = code_similarity(_CODE_A, _CODE_B, weights=(0.5, 0.3, 0.2))
        assert 0.0 <= score <= 1.0
