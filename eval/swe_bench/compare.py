"""Compare two ablation runs without requiring SWE-bench scoring.

Inputs: two `*.runlog.jsonl` sidecar files (one per ablation). Produces a
side-by-side report of metrics that are *measurable without Docker scoring*:

- exit_reason distribution
- per-instance tool-call counts
- patch produced rate (non-empty model_patch)
- patch size distribution
- wallclock time

Usage:
    python -m eval.swe_bench.compare \\
        --full out/cmp-full.jsonl.runlog.jsonl \\
        --bare out/cmp-bare.jsonl.runlog.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[row["instance_id"]] = row
    return rows


def _summary(label: str, rows: dict[str, dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"label": label, "n": 0}
    exit_dist = Counter(r["exit_reason"] for r in rows.values())
    tools = [r["tool_calls_used"] for r in rows.values()]
    diffs = [r["model_patch"].count("\n") if r.get("model_patch") else 0 for r in rows.values()]
    walls = [r["wallclock_s"] for r in rows.values() if r.get("wallclock_s")]
    nonempty = sum(1 for r in rows.values() if r.get("model_patch", "").strip())
    return {
        "label": label,
        "n": n,
        "patch_produced": nonempty,
        "exit_reasons": dict(exit_dist.most_common()),
        "tool_calls_median": statistics.median(tools) if tools else 0,
        "tool_calls_mean": round(statistics.mean(tools), 1) if tools else 0,
        "diff_lines_median": statistics.median(diffs) if diffs else 0,
        "diff_lines_mean": round(statistics.mean(diffs), 1) if diffs else 0,
        "wall_total_s": round(sum(walls), 1),
        "wall_mean_s": round(statistics.mean(walls), 1) if walls else 0,
    }


def _per_instance_table(full: dict[str, dict], bare: dict[str, dict]) -> str:
    keys = sorted(set(full) | set(bare))
    cols = (
        f"{'instance_id':<40} | "
        f"{'full exit':<14} {'full tools':>10} {'full diff':>9} | "
        f"{'bare exit':<14} {'bare tools':>10} {'bare diff':>9}"
    )
    lines = [cols, "-" * len(cols)]
    for k in keys:
        f = full.get(k) or {}
        b = bare.get(k) or {}
        f_diff = f.get("model_patch", "").count("\n") if f.get("model_patch") else 0
        b_diff = b.get("model_patch", "").count("\n") if b.get("model_patch") else 0
        lines.append(
            f"{k:<40} | "
            f"{f.get('exit_reason', '-'):<14} {f.get('tool_calls_used', '-'):>10} {f_diff:>9} | "
            f"{b.get('exit_reason', '-'):<14} {b.get('tool_calls_used', '-'):>10} {b_diff:>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two ablation runlog JSONLs.")
    p.add_argument("--full", required=True, type=Path)
    p.add_argument("--bare", required=True, type=Path)
    args = p.parse_args(argv)

    full = _load(args.full)
    bare = _load(args.bare)

    print("=" * 80)
    print("FARCODE SWE-BENCH ABLATION COMPARISON (without resolved% — pre-scoring)")
    print("=" * 80)

    sf = _summary("full", full)
    sb = _summary("bare", bare)
    for s in (sf, sb):
        print(f"\n--- {s['label']} ablation (n={s['n']}) ---")
        if s["n"] == 0:
            continue
        print(f"  patch_produced     : {s['patch_produced']}/{s['n']} "
              f"({100*s['patch_produced']/s['n']:.0f}%)")
        print(f"  exit_reasons       : {s['exit_reasons']}")
        print(f"  tool_calls (med/avg): {s['tool_calls_median']}/{s['tool_calls_mean']}")
        print(f"  diff_lines (med/avg): {s['diff_lines_median']}/{s['diff_lines_mean']}")
        print(f"  wallclock (sum/avg): {s['wall_total_s']:.0f}s / {s['wall_mean_s']:.0f}s")

    print("\n" + "=" * 80)
    print("PER-INSTANCE COMPARISON")
    print("=" * 80)
    print(_per_instance_table(full, bare))

    print("\n" + "=" * 80)
    print("CAVEAT")
    print("=" * 80)
    print("These numbers measure activity, not correctness. A non-empty patch\n"
          "doesn't mean the issue is resolved — it just means the agent edited\n"
          "something. The real metric (resolved%) requires running the official\n"
          "SWE-bench harness against these predictions in a Linux environment\n"
          "with Docker. Use this report to detect gross failure modes (one\n"
          "ablation timing out everywhere, or producing no patches) and as a\n"
          "directional signal for whether the architecture changes behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
