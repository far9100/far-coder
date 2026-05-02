# SWE-bench evaluation harness for farcode

End-to-end pipeline for measuring farcode against [SWE-bench](https://www.swebench.com).

The harness is intentionally kept out of `src/farcode/` so eval code never
ships in the wheel and never gets imported from the production CLI.

## What this measures

farcode is positioned as a local AI coding agent that "makes 4B-parameter local
models reliable at tool-calling" — through a stack of features (read-only
subagents, repo map, syntax-check feedback, grammar-constrained retry,
whitespace-tolerant edit, tool-per-turn cap, CODER.md / facts / memory injection).

Two ablations quantify the architecture's contribution:

- `full` — every subsystem on (default farcode behaviour).
- `bare` — `FARCODE_DISABLE_*` env vars turn every gated subsystem off,
  leaving raw Ollama tool-calling. The delta vs. `full` is the architecture's
  measured lift on resolved%.

## Requirements

- Docker (the official SWE-bench harness uses per-repo Docker images for
  reproducible test environments)
- Ollama running locally with the target model pulled
  (e.g. `ollama pull qwen3.5:4b`)
- Python 3.11+
- `pip install -e '.[eval]'` to install `datasets` and `swebench`

## Pipeline

```
SWE-bench Lite                 farcode                 Docker harness
──────────────                 ────────                ──────────────
load_dataset      ──┐
   │                │
   ▼                │
prepare_workspace   │
(bare clone +       │
 worktree per       │
 instance)          │
   │                │
   ▼                │
solve.py            │   (subprocess per instance)
   ├ build_system_  │
   │  messages      │
   ├ _run_agent_    │
   │  turn loop     │
   ├ stuck detect   │
   └ extract_patch  │
   │                │
   ▼                │
predictions.jsonl ──┴──▶  swebench.harness.run_evaluation  ──▶  report.json
                          (Docker per repo,
                           applies model_patch
                           + test_patch, runs tests)
```

## Step-by-step

### 1. Smoke test — one known-easy instance

Pick a Lite instance reported as a trivial fix (e.g. `astropy__astropy-12907`):

```
python -m eval.swe_bench.run \
    --dataset princeton-nlp/SWE-bench_Lite \
    --instance-ids astropy__astropy-12907 \
    --predictions out/smoke.jsonl \
    --model qwen3.5:4b \
    --ablation full
```

Then run the official scorer:

```
python -m eval.swe_bench.score \
    --predictions out/smoke.jsonl \
    --dataset princeton-nlp/SWE-bench_Lite \
    --run-id smoke
```

Confirm `out/smoke.jsonl` is well-formed JSONL (one row), the diff applies
inside Docker, the harness emits a report JSON. Even a fail is fine — we're
verifying pipeline patency.

### 2. Dev subset — 30 instances, both ablations

Run overnight:

```
python -m eval.swe_bench.run --limit 30 --predictions out/dev-full.jsonl --ablation full
python -m eval.swe_bench.run --limit 30 --predictions out/dev-bare.jsonl --ablation bare

python -m eval.swe_bench.score --predictions out/dev-full.jsonl --run-id dev-full
python -m eval.swe_bench.score --predictions out/dev-bare.jsonl --run-id dev-bare
```

Look for:
- Resolved-count delta ≥ 5 between the two — under that, small-N noise
  dominates and we should expand to full Lite before drawing conclusions.
- `out/dev-*.jsonl.runlog.jsonl` `exit_reason` distribution
  (`natural` / `stuck` / `max_tools` / `timeout` / `no_edits`) — this is the
  M5-priority signal: where is the agent failing?

### 3. Full Lite — 300 instances × 2 ablations

Only after dev-subset signal is real:

```
python -m eval.swe_bench.run --predictions out/lite-full.jsonl --ablation full
python -m eval.swe_bench.run --predictions out/lite-bare.jsonl --ablation bare

python -m eval.swe_bench.score --predictions out/lite-full.jsonl --run-id lite-full
python -m eval.swe_bench.score --predictions out/lite-bare.jsonl --run-id lite-bare
```

At ~15 min/instance × 300 instances × 2 ablations this is roughly 3-4 days
sequential. The runs are **resumable** — a restart picks up where the
JSONL left off. Don't try to parallelize; Ollama is single-process and
concurrent solves just thrash the GPU.

## Files

| File | Purpose |
|---|---|
| `solve.py` | Per-instance driver. Runs the agent loop, detects stuck/budget exit, extracts the unified diff. Invoked as a subprocess by `run.py`. |
| `run.py` | Batch driver. Loads dataset, prepares per-instance worktrees, spawns `solve.py`, appends to predictions JSONL + runlog JSONL. |
| `isolation.py` | Bare clone + git worktree per instance. Caches under `~/.farcode_swe_cache/`. |
| `ablation.py` | `full` / `bare` env-var preset definitions. |
| `score.py` | Thin wrapper around `python -m swebench.harness.run_evaluation`. |

## Output

- `predictions.jsonl` — one line per instance, format expected by SWE-bench
  harness: `{"instance_id", "model_patch", "model_name_or_path"}`.
- `predictions.jsonl.runlog.jsonl` — sidecar with per-instance metadata
  (`exit_reason`, `turns_used`, `tool_calls_used`, `elapsed_s`,
  `dropped_test_edits`, `wallclock_s`). Use this for post-hoc analysis.
- `<run_id>.<dataset>.json` (written by the harness in cwd) — the official
  resolved% report.

## Storage

The bare-clone cache lives at `~/.farcode_swe_cache/repos/`. SWE-bench Lite
spans ~12 distinct repos; expect 1-3 GB total. Worktrees are removed after
each instance, so they only briefly occupy ~100-500 MB each.

## Notes on the design

- **Subprocess per instance** is non-negotiable. An Ollama wedge or a single
  OOM crashing the whole batch is unacceptable on a multi-day run; subprocess
  isolation also gives a hard timeout the in-process loop can't escape.
- **Test-file exclusions** in the diff are conservative (`tests/**`,
  `test_*.py`, `*_test.py`, `conftest.py`). The harness applies its own
  test_patch — if our model_patch also touches tests, the apply step
  collides. We log dropped paths in `dropped_test_edits` so we can audit
  cases where the model thought tests needed editing.
- **Stuck detection** suppresses identical mutating tool calls after the
  3rd-in-a-row and aborts the loop on the 4th. Small models occasionally
  loop on the same failed edit indefinitely; without this they burn the
  whole turn budget on one instance.
- **No `farcode solve` CLI subcommand** is exposed. Eval code lives outside
  `src/farcode/` deliberately — it should never ship in the production wheel.
