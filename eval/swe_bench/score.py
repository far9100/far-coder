"""Thin wrapper around the official SWE-bench evaluation harness.

Usage:

    python -m eval.swe_bench.score \\
        --predictions out/predictions-full.jsonl \\
        --dataset princeton-nlp/SWE-bench_Lite \\
        --run-id full-lite-v1 [--max-workers 4]

Requires Docker and the `swebench` Python package — install via:

    pip install -e '.[eval]'

The harness writes its report JSON to `<run_id>.<dataset_slug>.json` in the
current directory; we just surface that path on completion.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def score(
    predictions: Path,
    dataset: str,
    run_id: str,
    *,
    max_workers: int = 4,
    split: str = "test",
) -> int:
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", str(predictions),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
    ]
    print(f"[score] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score predictions with the official SWE-bench harness.")
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-workers", type=int, default=4,
                   help="Parallel Docker containers for scoring (CPU bound, not GPU).")
    args = p.parse_args(argv)
    return score(
        args.predictions, args.dataset, args.run_id,
        max_workers=args.max_workers, split=args.split,
    )


if __name__ == "__main__":
    raise SystemExit(main())
