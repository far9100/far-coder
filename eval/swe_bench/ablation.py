"""Ablation presets for SWE-bench evaluation.

`full`  — all of farcode's reliability/context architecture enabled (default).
`bare`  — every gated subsystem disabled, leaving raw Ollama tool-calling.

The point of running both is to quantify how much the architecture lifts
resolved%. Each preset is a dict of env vars exported into the solve.py
subprocess; the gates themselves live in src/farcode/{client,tools}.py.
"""
from __future__ import annotations


ABLATIONS: dict[str, dict[str, str]] = {
    "full": {},
    "bare": {
        "FARCODE_DISABLE_SUBAGENT": "1",
        "FARCODE_DISABLE_REPOMAP": "1",
        "FARCODE_DISABLE_CODER_MD": "1",
        "FARCODE_DISABLE_MEMORY": "1",
        "FARCODE_DISABLE_RELIABILITY": "1",
        "FARCODE_MAX_TOOLS_PER_TURN": "0",
    },
    # Single-knob ablations: each disables exactly one subsystem, all else FULL.
    # Used to attribute which subsystem caused FULL to lose django-11179 to BARE.
    "only_no_serial":   {"FARCODE_MAX_TOOLS_PER_TURN": "0"},
    "only_no_repomap":  {"FARCODE_DISABLE_REPOMAP": "1"},
    "only_no_memory":   {"FARCODE_DISABLE_MEMORY": "1"},
    "only_no_codermd":  {"FARCODE_DISABLE_CODER_MD": "1"},
    "only_no_subagent": {"FARCODE_DISABLE_SUBAGENT": "1"},
    # M5 — SWE-bench-tuned presets. `swe_v3` bundles the localization aids
    # (locator + focused repomap), patch-shape prompt addendum, and the
    # edit-failure / read-loop reliability fixes. `swe_v3_critique` adds the
    # patch critique subagent. Two single-knob ablations let us attribute
    # the lift between locator vs prompt addendum.
    "swe_v3": {
        "FARCODE_SWE_MODE": "1",
        "FARCODE_SWE_LOCATOR": "1",
    },
    "swe_v3_critique": {
        "FARCODE_SWE_MODE": "1",
        "FARCODE_SWE_LOCATOR": "1",
        "FARCODE_SWE_CRITIQUE": "1",
    },
    # Same as swe_v3 minus the locator (kept the prompt addendum).
    "swe_v3_no_locator": {
        "FARCODE_SWE_MODE": "1",
    },
    # Same as swe_v3 minus the prompt addendum (kept the locator).
    "swe_v3_no_prompt": {
        "FARCODE_SWE_LOCATOR": "1",
    },
    # M6 — full suite. Locator + slimmed SWE prompt (auto-on via FARCODE_SWE_MODE)
    # + test contract injection (FAIL_TO_PASS / PASS_TO_PASS hints) + post-loop
    # syntax gate (one bonus retry on broken Python). Targets the three blockers
    # diagnosed in the post-M5 failure dissection.
    "swe_v4": {
        "FARCODE_SWE_MODE": "1",
        "FARCODE_SWE_LOCATOR": "1",
        "FARCODE_SWE_TEST_CONTRACT": "1",
        "FARCODE_SWE_SYNTAX_GATE": "1",
    },
    # Attribution probes for M6: one knob removed at a time, all else equal.
    # Use these once swe_v4 has a baseline number to measure each phase's lift.
    "swe_v4_no_contract": {
        "FARCODE_SWE_MODE": "1",
        "FARCODE_SWE_LOCATOR": "1",
        "FARCODE_SWE_SYNTAX_GATE": "1",
    },
    "swe_v4_no_gate": {
        "FARCODE_SWE_MODE": "1",
        "FARCODE_SWE_LOCATOR": "1",
        "FARCODE_SWE_TEST_CONTRACT": "1",
    },
    # Variance-reduction probe — same as swe_v3 (the best-known config) plus
    # FARCODE_TEMPERATURE=0.2. Post-M6 analysis identified single-run sampling
    # variance as the dominant blocker (e.g. sympy-15345 swung from
    # nat/11t/13d to max_tools/101t/33d on the SAME config across runs).
    # Lowering temperature should make the model converge on the highest-
    # probability tool calls more reliably, reducing wild-output cases.
    "swe_v3_lowtemp": {
        "FARCODE_SWE_LOCATOR": "1",
        "FARCODE_TEMPERATURE": "0.2",
    },
}


def env_for(name: str) -> dict[str, str]:
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation '{name}'; choose from {sorted(ABLATIONS)}")
    return dict(ABLATIONS[name])


def names() -> list[str]:
    return sorted(ABLATIONS)
