"""
corpus/performance_estimator/bootstrap.py — Synthetic 𝒟_Perf bootstrap generator.

Generates labelled training data for the Probabilistic Heuristic Filter when
no real CI/CD validation history is available.  Produces N "safe" code snippets
(label=0) and N "risky" code snippets (label=1) and writes them to a JSONL file.

Usage (standalone):
    python -m corpus.performance_estimator.bootstrap \\
        [--samples 100] [--output ./data/d_perf/bootstrap.jsonl]

CLI entry point (installed package):
    etft-bootstrap-filter [--samples 100] [--output ./data/d_perf/bootstrap.jsonl]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from corpus.performance_estimator.dataset_builder import PerfDatasetBuilder
from corpus.performance_estimator.feature_extractor import extract_features
from corpus.regression_pipeline.schemas import PerfSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code templates
# ---------------------------------------------------------------------------

# Safe snippets — small, well-formed, no CUDA/GPU/large-allocation patterns
_SAFE_TEMPLATES: list[str] = [
    # Minimal function
    "def f(x):\n    return x * 2\n\nresult = f(5)\nprint(result)\n",
    # Simple loop with numpy
    "import numpy as np\n\nx = np.arange(10)\ny = x ** 2\nprint(y.sum())\n",
    # scipy stats
    (
        "import numpy as np\nfrom scipy import stats\n\n"
        "data = np.random.normal(0, 1, 100)\nprint(stats.describe(data))\n"
    ),
    # sklearn classification
    (
        "from sklearn.datasets import make_classification\n"
        "from sklearn.linear_model import LogisticRegression\n\n"
        "X, y = make_classification(n_samples=50, random_state=0)\n"
        "clf = LogisticRegression(max_iter=200)\nclf.fit(X, y)\nprint(clf.score(X, y))\n"
    ),
    # Math utilities
    "import math\n\nresult = math.factorial(10)\nprint(result)\n",
    # Statistics module
    "import statistics\n\ndata = [1, 2, 3, 4, 5]\nprint(statistics.mean(data))\n",
    # Random sampling
    "import random\n\nsamples = [random.gauss(0, 1) for _ in range(20)]\nprint(sum(samples))\n",
    # List comprehension
    "squares = [x**2 for x in range(50)]\nprint(squares[:5])\n",
    # Simple class
    (
        "class Counter:\n    def __init__(self):\n        self.count = 0\n"
        "    def increment(self):\n        self.count += 1\n\n"
        "c = Counter()\nfor _ in range(10):\n    c.increment()\nprint(c.count)\n"
    ),
    # Try/except with safe code
    (
        "import math\n\ntry:\n    result = math.log(100)\nexcept ValueError as e:\n"
        "    result = 0\nprint(result)\n"
    ),
    # Numpy array ops
    (
        "import numpy as np\n\nA = np.random.randn(10, 10)\nb = np.random.randn(10)\n"
        "x = np.linalg.solve(A, b)\nprint(x)\n"
    ),
    # sklearn cross-validation
    (
        "from sklearn.datasets import make_regression\n"
        "from sklearn.linear_model import Ridge\n"
        "from sklearn.model_selection import cross_val_score\n\n"
        "X, y = make_regression(n_samples=50, n_features=5, random_state=0)\n"
        "scores = cross_val_score(Ridge(), X, y, cv=3)\nprint(scores.mean())\n"
    ),
]

# Risky snippets — patterns strongly correlated with runtime failures:
# CUDA/GPU access, huge tensor allocations, deep recursion, missing error handling
_RISKY_TEMPLATES: list[str] = [
    # CUDA allocation without availability check
    (
        "import torch\n\n"
        "device = torch.device('cuda')\n"
        "x = torch.randn(1000, 1000, device=device)\n"
        "print(x.sum())\n"
    ),
    # Very large tensor (OOM-risk)
    (
        "import torch\n\n"
        "x = torch.zeros(10_000, 10_000, dtype=torch.float32)\n"
        "y = torch.zeros(10_000, 10_000, dtype=torch.float32)\n"
        "z = x @ y\nprint(z.shape)\n"
    ),
    # amp / autocast without fallback
    (
        "import torch\nfrom torch.cuda.amp import autocast\n\n"
        "with autocast():\n    x = torch.randn(512, 512, device='cuda')\n"
        "print(x.sum())\n"
    ),
    # Unbounded recursion
    (
        "def fib(n):\n    return fib(n - 1) + fib(n - 2)\n\n"
        "print(fib(10000))\n"
    ),
    # float16 / half precision without device check
    (
        "import torch\n\n"
        "model = torch.nn.Linear(1024, 1024).half().cuda()\n"
        "x = torch.randn(256, 1024, device='cuda', dtype=torch.float16)\n"
        "print(model(x))\n"
    ),
    # GPU dataloader with num_workers
    (
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n\n"
        "ds = TensorDataset(torch.randn(1000, 512))\n"
        "dl = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True)\n"
        "for batch in dl:\n    print(batch[0].cuda().sum())\n"
    ),
    # NaN-propagating arithmetic
    (
        "import torch\n\n"
        "x = torch.tensor([1.0, float('nan'), 3.0])\n"
        "y = x / 0\nprint(y)\n"
    ),
    # Huge numpy allocation
    (
        "import numpy as np\n\n"
        "# Attempt to allocate ~8 GB\n"
        "x = np.zeros((100_000, 10_000), dtype=np.float64)\n"
        "print(x.shape)\n"
    ),
    # import torch + cuda check without try/except
    (
        "import torch\n\n"
        "assert torch.cuda.is_available(), 'CUDA required'\n"
        "x = torch.randn(4096, 4096, device='cuda')\n"
        "print(x.mean())\n"
    ),
    # Implicit GPU memory growth
    (
        "import torch\n\n"
        "results = []\n"
        "for i in range(1000):\n"
        "    results.append(torch.randn(1024, 1024, device='cuda'))\n"
        "print(len(results))\n"
    ),
    # torch.alloc large buffer
    (
        "import torch\n\n"
        "buf = torch.cuda.ByteTensor(2 ** 30)  # 1 GB\n"
        "print(buf.shape)\n"
    ),
    # inf / nan loss
    (
        "import torch\n\n"
        "logits = torch.tensor([1e38, 1e38])\n"
        "loss = torch.nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([0]))\n"
        "loss.backward()\nprint(loss)\n"
    ),
]


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_bootstrap_data(
    output_path: Path,
    n_samples: int = 100,
) -> int:
    """
    Generate synthetic 𝒟_Perf bootstrap data and write it to *output_path*.

    Each template is repeated round-robin until *n_samples* safe examples and
    *n_samples* risky examples have been written (total = 2 × *n_samples*).

    Parameters
    ----------
    output_path:
        Destination JSONL file.  Parent directories are created automatically.
    n_samples:
        Number of examples *per class* (safe / risky) to generate.

    Returns
    -------
    int
        Total number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "a") as fh:
        # Safe examples (label = 0)
        for i in range(n_samples):
            code = _SAFE_TEMPLATES[i % len(_SAFE_TEMPLATES)]
            features = extract_features(code)
            sample = PerfSample(
                algorithm_id=f"bootstrap_safe_{i:04d}",
                features=features,
                label=0,
                failure_reason=None,
                metadata={"bootstrap": True, "class": "safe"},
            )
            fh.write(sample.model_dump_json() + "\n")
            total += 1

        # Risky examples (label = 1)
        for i in range(n_samples):
            code = _RISKY_TEMPLATES[i % len(_RISKY_TEMPLATES)]
            features = extract_features(code)
            sample = PerfSample(
                algorithm_id=f"bootstrap_risky_{i:04d}",
                features=features,
                label=1,
                failure_reason="Bootstrap synthetic risky pattern",
                metadata={"bootstrap": True, "class": "risky"},
            )
            fh.write(sample.model_dump_json() + "\n")
            total += 1

    logger.info(
        "Bootstrap complete: %d safe + %d risky = %d total samples → %s",
        n_samples, n_samples, total, output_path,
    )
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic 𝒟_Perf bootstrap data for the Probabilistic Heuristic Filter."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of examples per class (safe / risky) to generate. Default: 100.",
    )
    parser.add_argument(
        "--output",
        default="./data/d_perf/bootstrap.jsonl",
        help="Output JSONL file path. Default: ./data/d_perf/bootstrap.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    generate_bootstrap_data(Path(args.output), n_samples=args.samples)


if __name__ == "__main__":
    main()
