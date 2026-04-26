import json
import uuid
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path.home() / ".farcode_memory.jsonl"
_TRUNC = 150  # chars per bullet injected into system prompt


def append_entry(session_id: str, summary: str) -> None:
    entry = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "summary": summary,
    }
    with MEMORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_all() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    out: list[dict] = []
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_recent(n: int = 5) -> list[dict]:
    return _load_all()[-n:]


def search(query: str, top_k: int = 5) -> list[dict]:
    tokens = query.lower().split()
    if not tokens:
        return []
    scored: list[tuple[int, dict]] = []
    for entry in _load_all():
        score = sum(1 for t in tokens if t in entry.get("summary", "").lower())
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def format_for_prompt(entries: list[dict]) -> str:
    if not entries:
        return ""
    lines = ["## Past Work"]
    for e in entries:
        date = e.get("created_at", "")[:10]
        s = e.get("summary", "")
        lines.append(f"- [{date}] {s[:_TRUNC]}{'...' if len(s) > _TRUNC else ''}")
    return "\n".join(lines)
