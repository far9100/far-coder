"""Tiny helper for boolean env-var gates used by the SWE-bench ablations."""
import os


def env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
