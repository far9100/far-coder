"""Scoped HTTP fetcher for package documentation lookup.

Strict allowlist: only PyPI, npm, crates.io, and pkg.go.dev are reachable.
Any other URL is refused. The caller (``fetch_doc`` tool) handles
opt-in gating; this module trusts that it is only invoked when the user
has explicitly enabled web access.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

MAX_RESPONSE_BYTES = 8192
TIMEOUT_SECONDS = 10.0

ECOSYSTEM_REGISTRIES: dict[str, callable] = {
    "python":     lambda pkg: f"https://pypi.org/pypi/{pkg}/json",
    "pypi":       lambda pkg: f"https://pypi.org/pypi/{pkg}/json",
    "javascript": lambda pkg: f"https://registry.npmjs.org/{pkg}",
    "npm":        lambda pkg: f"https://registry.npmjs.org/{pkg}",
    "rust":       lambda pkg: f"https://crates.io/api/v1/crates/{pkg}",
    "go":         lambda pkg: f"https://pkg.go.dev/{pkg}",
}

ALLOWED_HOSTS = frozenset({
    "pypi.org",
    "registry.npmjs.org",
    "crates.io",
    "pkg.go.dev",
})


def fetch(query: str, ecosystem: str = "auto") -> str:
    """Fetch package documentation. Returns trimmed text or an error string."""
    pkg = (query or "").strip()
    if not pkg:
        return "fetch_doc error: empty query."
    if not _is_safe_package_name(pkg):
        return "fetch_doc error: invalid package name (allowed: A-Z a-z 0-9 . _ - / @)."

    eco = (ecosystem or "auto").strip().lower()
    if eco == "auto":
        eco = "pypi"
    builder = ECOSYSTEM_REGISTRIES.get(eco)
    if not builder:
        opts = ", ".join(sorted(set(ECOSYSTEM_REGISTRIES.keys())))
        return f"fetch_doc error: unknown ecosystem '{ecosystem}'. Allowed: {opts}"

    url = builder(pkg)
    if not _url_in_allowlist(url):
        return f"fetch_doc error: URL not in allowlist: {url}"

    try:
        import httpx
    except ImportError:
        return "fetch_doc error: httpx is not installed."

    try:
        r = httpx.get(url, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    except Exception as e:
        return f"fetch_doc error: {type(e).__name__}: {e}"

    if r.status_code == 404:
        return f"fetch_doc: package '{pkg}' not found on {eco}."
    if r.status_code >= 400:
        return f"fetch_doc error: HTTP {r.status_code} from {url}"

    body = r.text
    content_type = r.headers.get("content-type", "").lower()
    if "json" in content_type:
        body = _summarize_json(body, eco)
    elif "html" in content_type:
        body = _strip_html(body)

    if len(body) > MAX_RESPONSE_BYTES:
        body = body[:MAX_RESPONSE_BYTES] + f"\n\n... (truncated at {MAX_RESPONSE_BYTES} chars)"
    return body


def _is_safe_package_name(s: str) -> bool:
    if ".." in s:
        return False
    return bool(re.match(r"^[A-Za-z0-9._@\-/]+$", s))


def _url_in_allowlist(url: str) -> bool:
    return any(
        url.startswith(f"https://{host}/") or url == f"https://{host}"
        for host in ALLOWED_HOSTS
    )


# ── HTML → text ──────────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth:
            return
        s = data.strip()
        if s:
            self.parts.append(s)

    def text(self) -> str:
        return "\n".join(self.parts)


def _strip_html(s: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(s)
    except Exception:
        return s[:MAX_RESPONSE_BYTES]
    return parser.text()


# ── JSON registry summary ────────────────────────────────────────────────────

def _summarize_json(body: str, ecosystem: str) -> str:
    try:
        data = json.loads(body)
    except Exception:
        return body

    if ecosystem in ("pypi", "python"):
        info = data.get("info", {}) if isinstance(data, dict) else {}
        return _format_kv({
            "name": info.get("name"),
            "version": info.get("version"),
            "summary": info.get("summary"),
            "home_page": info.get("home_page"),
            "requires_python": info.get("requires_python"),
            "license": info.get("license"),
        })
    if ecosystem in ("npm", "javascript") and isinstance(data, dict):
        dist_tags = data.get("dist-tags") or {}
        return _format_kv({
            "name": data.get("name"),
            "description": data.get("description"),
            "latest": dist_tags.get("latest"),
            "license": data.get("license"),
            "homepage": data.get("homepage"),
        })
    if ecosystem == "rust" and isinstance(data, dict):
        crate = data.get("crate", {}) or {}
        return _format_kv({
            "name": crate.get("name"),
            "description": crate.get("description"),
            "max_version": crate.get("max_version"),
            "documentation": crate.get("documentation"),
            "homepage": crate.get("homepage"),
            "repository": crate.get("repository"),
        })
    return body[:2000]


def _format_kv(d: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items() if v)
