"""
Conversation session persistence.

Sessions are stored as JSON files in ~/.ai_coder_sessions/.
Each file is named <YYYYMMDD_HHMMSS_xxxxxx>.json and contains the full
message history along with metadata (title, model, timestamps).
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path.home() / ".ai_coder_sessions"


@dataclass
class Session:
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    messages: list  # list[dict] — always plain dicts, never Ollama objects

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.json"

    @property
    def turn_count(self) -> int:
        """Number of user turns (excludes system + tool messages)."""
        return sum(1 for m in self.messages if m.get("role") == "user")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _derive_title(messages: list) -> str:
    """Extract a human-readable title from the first real user message."""
    for m in messages:
        if m.get("role") != "user":
            continue
        text = str(m.get("content", "")).strip()
        # Skip synthetic file-injection messages (start with "File `...")
        if text.startswith("File `"):
            continue
        first_line = text.split("\n")[0].strip()
        if first_line:
            return first_line[:60] + ("..." if len(first_line) > 60 else "")
    return "(empty)"


def _validate_messages(raw: list) -> list[dict]:
    """Sanitise a raw JSON-decoded messages list before loading into a Session."""
    VALID_ROLES = {"system", "user", "assistant", "tool"}
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in VALID_ROLES:
            continue
        cleaned: dict = {"role": role}
        content = entry.get("content")
        if content is None:
            cleaned["content"] = None
        elif isinstance(content, str):
            cleaned["content"] = content
        else:
            cleaned["content"] = str(content)
        tool_calls = entry.get("tool_calls")
        if isinstance(tool_calls, list):
            cleaned["tool_calls"] = tool_calls
        out.append(cleaned)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def new_session(model: str) -> Session:
    """Create a new empty session."""
    now = datetime.now()
    sid = now.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    ts = now.isoformat(timespec="seconds")
    return Session(id=sid, title="(new)", model=model, created_at=ts, updated_at=ts, messages=[])


def save_session(session: Session) -> None:
    """Persist a session to disk (creates or overwrites its file)."""
    _ensure_dir()
    session.updated_at = datetime.now().isoformat(timespec="seconds")
    session.title = _derive_title(session.messages)
    session.path.write_text(
        json.dumps(
            {
                "id": session.id,
                "title": session.title,
                "model": session.model,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "messages": session.messages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_sessions(limit: int = 20) -> list[Session]:
    """Return up to `limit` sessions, most-recently-modified first."""
    _ensure_dir()
    files = sorted(
        SESSIONS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result: list[Session] = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            raw_messages = data.get("messages", [])
            data["messages"] = _validate_messages(raw_messages) if isinstance(raw_messages, list) else []
            result.append(Session(**data))
        except Exception:
            pass
    return result


def load_last_session() -> Session | None:
    """Return the most recently modified session, or None."""
    sessions = load_sessions(limit=1)
    return sessions[0] if sessions else None


def delete_session(session_id: str) -> bool:
    """Delete a session by exact ID or unique prefix. Returns True if deleted."""
    _ensure_dir()
    exact = SESSIONS_DIR / f"{session_id}.json"
    if exact.exists():
        exact.unlink()
        return True
    matches = [p for p in SESSIONS_DIR.glob("*.json") if p.stem.startswith(session_id)]
    if len(matches) == 1:
        matches[0].unlink()
        return True
    return False  # 0 = not found, 2+ = ambiguous


def search_sessions(query: str, limit: int = 20) -> list[Session]:
    """Return sessions whose title contains query (case-insensitive)."""
    q = query.lower()
    return [s for s in load_sessions(limit=200) if q in s.title.lower()][:limit]
