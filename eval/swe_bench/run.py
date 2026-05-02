"""Batch driver — iterate SWE-bench instances, isolate each solve in a
subprocess, append predictions to a JSONL file (resumable).

CLI:

    python -m eval.swe_bench.run \\
        --dataset princeton-nlp/SWE-bench_Lite \\
        --split test \\
        --predictions out/predictions-full.jsonl \\
        --runlog out/runlog-full.jsonl \\
        --model qwen3.5:4b \\
        --ablation full \\
        [--limit 30] [--instance-ids id1,id2] [--solve-timeout 1200]

Strict serial — Ollama is single-process, so concurrency just thrashes the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from . import ablation
from .isolation import cleanup_workspace, prepare_workspace


def _load_dataset(name: str, split: str) -> list[dict]:
    """Pull a SWE-bench dataset via HuggingFace `datasets`. We materialize the
    list eagerly so we can re-iterate, count, and slice. SWE-bench Lite is
    300 rows — peanuts."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with: "
            "pip install -e '.[eval]'"
        ) from e
    ds = load_dataset(name, split=split)
    return [dict(row) for row in ds]


def _completed_ids(predictions: Path) -> set[str]:
    if not predictions.exists():
        return set()
    done: set[str] = set()
    for line in predictions.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["instance_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _run_one(
    instance: dict,
    *,
    model: str,
    ablation_name: str,
    num_ctx: int,
    num_predict: int,
    max_tools: int,
    solve_timeout: int,
    subprocess_timeout: int,
) -> dict:
    """Run solve.py in a fresh subprocess; return the SolveResult dict (or a
    synthetic failure dict on subprocess error)."""
    workspace = prepare_workspace(instance)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            instance_path = tmp_dir / "instance.json"
            output_path = tmp_dir / "result.json"
            instance_path.write_text(json.dumps(instance), encoding="utf-8")

            env = dict(os.environ)
            env.update(ablation.env_for(ablation_name))

            cmd = [
                sys.executable, "-m", "eval.swe_bench.solve",
                "--instance-json", str(instance_path),
                "--workspace", str(workspace),
                "--output", str(output_path),
                "--model", model,
                "--num-ctx", str(num_ctx),
                "--num-predict", str(num_predict),
                "--max-tools", str(max_tools),
                "--timeout", str(solve_timeout),
            ]
            try:
                # encoding="utf-8" / errors="replace" — Windows defaults to
                # cp950 here when the system locale is Traditional Chinese,
                # which can't decode UTF-8 bytes that the child emits (em-dashes
                # in tool output, etc.) and crashes the reader thread.
                proc = subprocess.run(
                    cmd, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=subprocess_timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                return _failure_dict(
                    instance, model, "subprocess_timeout",
                    "subprocess exceeded outer timeout",
                )
            if proc.returncode != 0 or not output_path.exists():
                return _failure_dict(
                    instance, model, "subprocess_error",
                    (proc.stderr or proc.stdout or "")[-2000:],
                )
            return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        cleanup_workspace(workspace)


def _failure_dict(instance: dict, model: str, exit_reason: str, detail: str) -> dict:
    return {
        "instance_id": instance["instance_id"],
        "model_patch": "",
        "model_name": model,
        "exit_reason": exit_reason,
        "turns_used": 0,
        "tool_calls_used": 0,
        "elapsed_s": 0.0,
        "dropped_test_edits": [],
        "subprocess_detail": detail,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch SWE-bench evaluation driver for farcode.")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HuggingFace dataset name.")
    p.add_argument("--split", default="test")
    p.add_argument("--predictions", required=True, type=Path,
                   help="JSONL output for SWE-bench harness scoring.")
    p.add_argument("--runlog", type=Path, default=None,
                   help="Optional sidecar JSONL with per-instance metadata. "
                        "Defaults to predictions path with .runlog.jsonl suffix.")
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--ablation", default="full", choices=ablation.names())
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N instances (after resume filter).")
    p.add_argument("--instance-ids", default=None,
                   help="Comma-separated list of instance_ids to run (overrides --limit).")
    p.add_argument("--num-ctx", type=int, default=65536)
    p.add_argument("--num-predict", type=int, default=4096)
    # Bumped from 60 → 100 in M5: the SWE-mode prompt + locator reduce
    # wasted reads, but the patch-shape discipline + tests-as-contract guidance
    # nudge the model toward more deliberate steps. 100 leaves headroom on
    # bigger investigations without changing the pass/fail behavior on small ones.
    p.add_argument("--max-tools", type=int, default=100)
    p.add_argument("--solve-timeout", type=int, default=900,
                   help="Per-instance internal time budget (seconds).")
    p.add_argument("--subprocess-timeout", type=int, default=1200,
                   help="Outer subprocess timeout, slack over --solve-timeout.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runlog_path = args.runlog or args.predictions.with_suffix(args.predictions.suffix + ".runlog.jsonl")

    print(f"[run] dataset={args.dataset} split={args.split} model={args.model} ablation={args.ablation}")
    instances = _load_dataset(args.dataset, args.split)
    print(f"[run] loaded {len(instances)} instances")

    if args.instance_ids:
        wanted = {x.strip() for x in args.instance_ids.split(",") if x.strip()}
        instances = [i for i in instances if i["instance_id"] in wanted]
        print(f"[run] filtered to {len(instances)} requested instance(s)")

    completed = _completed_ids(args.predictions)
    if completed:
        print(f"[run] resuming — {len(completed)} instance(s) already in {args.predictions}")
    instances = [i for i in instances if i["instance_id"] not in completed]

    if args.limit is not None:
        instances = instances[: args.limit]
        print(f"[run] capped to first {args.limit}")

    total = len(instances)
    print(f"[run] {total} instance(s) to process\n")

    model_name_or_path = f"farcode-{args.model}-{args.ablation}"
    started_all = time.monotonic()

    for idx, instance in enumerate(instances, start=1):
        iid = instance["instance_id"]
        t0 = time.monotonic()
        print(f"[{idx}/{total}] {iid} ... ", end="", flush=True)
        try:
            result = _run_one(
                instance,
                model=args.model,
                ablation_name=args.ablation,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                max_tools=args.max_tools,
                solve_timeout=args.solve_timeout,
                subprocess_timeout=args.subprocess_timeout,
            )
        except Exception as e:
            result = _failure_dict(instance, args.model, "driver_error", repr(e)[:500])

        elapsed = time.monotonic() - t0
        prediction = {
            "instance_id": result["instance_id"],
            "model_patch": result["model_patch"],
            "model_name_or_path": model_name_or_path,
        }
        runlog_entry = dict(result)
        runlog_entry["model_name_or_path"] = model_name_or_path
        runlog_entry["wallclock_s"] = elapsed

        _append_jsonl(args.predictions, prediction)
        _append_jsonl(runlog_path, runlog_entry)

        diff_lines = result["model_patch"].count("\n") if result.get("model_patch") else 0
        print(
            f"reason={result.get('exit_reason')} turns={result.get('turns_used', 0)} "
            f"tools={result.get('tool_calls_used', 0)} diff_lines={diff_lines} "
            f"elapsed={elapsed:.1f}s"
        )

    total_elapsed = time.monotonic() - started_all
    print(f"\n[run] done — {total} instance(s) in {total_elapsed/60:.1f}min")
    print(f"[run] predictions → {args.predictions}")
    print(f"[run] runlog      → {runlog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
