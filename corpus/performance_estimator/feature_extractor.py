"""
feature_extractor.py — Extracts structural features from a Python algorithm.

Features are numeric scalars fed to the sklearn Probabilistic Heuristic Filter.
All extraction is static (AST-based) so no execution is required.
"""

from __future__ import annotations

import ast
import math

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(code: str) -> dict[str, float]:
    """
    Parse *code* and return a flat dict of numeric structural features.

    Returns a zero-filled dict on parse failure so the pipeline never crashes
    on malformed code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _zero_features()

    extractor = _Extractor(len(code))
    extractor.visit(tree)
    return extractor.features()


def feature_names() -> list[str]:
    """Return the ordered list of feature names (matches extract_features keys)."""
    return list(_zero_features().keys())


# ---------------------------------------------------------------------------
# Internal AST visitor
# ---------------------------------------------------------------------------


class _Extractor(ast.NodeVisitor):
    def __init__(self, code_chars: int = 0) -> None:
        self._code_chars = code_chars
        self._num_functions = 0
        self._num_classes = 0
        self._num_loops = 0
        self._num_imports = 0
        self._num_assignments = 0
        self._num_calls = 0
        self._num_returns = 0
        self._num_decorators = 0
        self._max_nesting = 0
        self._current_nesting = 0
        self._num_try_except = 0
        self._num_comprehensions = 0
        self._num_lambda = 0
        self._num_yield = 0

    # --- visitors ---

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._num_functions += 1
        self._num_decorators += len(node.decorator_list)
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._num_classes += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_For(self, node: ast.For) -> None:
        self._num_loops += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    visit_While = visit_For
    visit_AsyncFor = visit_For

    def visit_Import(self, node: ast.Import) -> None:
        self._num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._num_imports += len(node.names)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._num_assignments += 1
        self.generic_visit(node)

    visit_AugAssign = visit_Assign
    visit_AnnAssign = visit_Assign

    def visit_Call(self, node: ast.Call) -> None:
        self._num_calls += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._num_returns += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._num_try_except += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._num_comprehensions += 1
        self.generic_visit(node)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._num_lambda += 1
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._num_yield += 1
        self.generic_visit(node)

    visit_YieldFrom = visit_Yield

    # --- helpers ---

    def _enter_block(self) -> None:
        self._current_nesting += 1
        if self._current_nesting > self._max_nesting:
            self._max_nesting = self._current_nesting

    def _exit_block(self) -> None:
        self._current_nesting -= 1

    # --- output ---

    def features(self) -> dict[str, float]:
        return {
            "num_functions": float(self._num_functions),
            "num_classes": float(self._num_classes),
            "num_loops": float(self._num_loops),
            "num_imports": float(self._num_imports),
            "num_assignments": float(self._num_assignments),
            "num_calls": float(self._num_calls),
            "num_returns": float(self._num_returns),
            "num_decorators": float(self._num_decorators),
            "max_nesting_depth": float(self._max_nesting),
            "num_try_except": float(self._num_try_except),
            "num_comprehensions": float(self._num_comprehensions),
            "num_lambda": float(self._num_lambda),
            "num_yield": float(self._num_yield),
            "code_length_chars": float(self._code_chars),
            "calls_per_function": (
                self._num_calls / self._num_functions if self._num_functions else 0.0
            ),
            "log_code_chars": math.log1p(self._code_chars),
        }


def _zero_features() -> dict[str, float]:
    return {
        "num_functions": 0.0,
        "num_classes": 0.0,
        "num_loops": 0.0,
        "num_imports": 0.0,
        "num_assignments": 0.0,
        "num_calls": 0.0,
        "num_returns": 0.0,
        "num_decorators": 0.0,
        "max_nesting_depth": 0.0,
        "num_try_except": 0.0,
        "num_comprehensions": 0.0,
        "num_lambda": 0.0,
        "num_yield": 0.0,
        "code_length_chars": 0.0,
        "calls_per_function": 0.0,
        "log_code_chars": 0.0,
    }


