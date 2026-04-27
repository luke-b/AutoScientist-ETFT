"""
feature_extractor.py — Extracts structural features from a Python algorithm.

Features are numeric scalars fed to the sklearn Probabilistic Heuristic Filter.
All extraction is static (AST-based + TF-IDF) so no execution is required.

Feature set (66 total)
----------------------
  AST features (16): classic structural counts derived from the parsed AST.
  TF-IDF features (50): term-frequency scores for a fixed ML-vocabulary
      that captures optimizer names, layer types, regularisation terms, and
      other semantic signals relevant to predicting run-time failures.
"""

from __future__ import annotations

import ast
import math
import re

# ---------------------------------------------------------------------------
# Fixed ML vocabulary for TF-IDF (50 tokens)
# ---------------------------------------------------------------------------

_ML_VOCAB: list[str] = [
    # Optimisers
    "adam", "sgd", "adamw", "adagrad", "rmsprop", "nadam", "adadelta",
    # Layers / blocks
    "linear", "conv2d", "conv1d", "batchnorm", "layernorm", "groupnorm",
    "dropout", "relu", "gelu", "sigmoid", "softmax", "embedding",
    "transformer", "attention", "lstm", "gru", "rnn",
    # Regularisation
    "weight_decay", "l2", "l1", "regularization", "clip_grad",
    # Training constructs
    "scheduler", "criterion", "loss", "backward", "optimizer", "zero_grad",
    "step", "epoch", "batch", "dataloader", "dataset",
    # Common failure-correlated patterns
    "cuda", "gpu", "device", "float16", "amp", "autocast",
    "alloc", "oom", "memory", "nan", "inf",
    # Architecture scale signals
    "hidden_size", "num_layers", "num_heads", "d_model",
]

# Build a term→index mapping once at module level
_VOCAB_INDEX: dict[str, int] = {term: i for i, term in enumerate(_ML_VOCAB)}
_N_TFIDF = len(_ML_VOCAB)  # == 50

_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*")


def _tfidf_features(code: str) -> dict[str, float]:
    """
    Compute sublinear TF scores for each term in *_ML_VOCAB*.

    Uses ``tf = 1 + log(count)`` (sublinear TF) when count > 0, 0 otherwise.
    IDF is omitted because we operate on single documents; the vocabulary
    itself encodes the domain-relevance signal.
    """
    lowered = code.lower()
    tokens = _TOKEN_RE.findall(lowered)
    counts: dict[str, int] = {}
    for tok in tokens:
        if tok in _VOCAB_INDEX:
            counts[tok] = counts.get(tok, 0) + 1

    features: dict[str, float] = {}
    for term in _ML_VOCAB:
        key = f"tfidf_{term}"
        cnt = counts.get(term, 0)
        features[key] = (1.0 + math.log(cnt)) if cnt > 0 else 0.0
    return features


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
    ast_feats = extractor.features()
    tfidf_feats = _tfidf_features(code)
    return {**ast_feats, **tfidf_feats}


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

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

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

    visit_While = visit_For  # noqa: N815
    visit_AsyncFor = visit_For  # noqa: N815

    def visit_Import(self, node: ast.Import) -> None:
        self._num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._num_imports += len(node.names)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._num_assignments += 1
        self.generic_visit(node)

    visit_AugAssign = visit_Assign  # noqa: N815
    visit_AnnAssign = visit_Assign  # noqa: N815

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

    visit_SetComp = visit_ListComp  # noqa: N815
    visit_DictComp = visit_ListComp  # noqa: N815
    visit_GeneratorExp = visit_ListComp  # noqa: N815

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._num_lambda += 1
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._num_yield += 1
        self.generic_visit(node)

    visit_YieldFrom = visit_Yield  # noqa: N815

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
    ast_zeros = {
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
    tfidf_zeros = {f"tfidf_{term}": 0.0 for term in _ML_VOCAB}
    return {**ast_zeros, **tfidf_zeros}

