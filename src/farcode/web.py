"""Scoped HTTP fetcher for package documentation lookup.

Strict allowlist: only PyPI, npm, crates.io, and pkg.go.dev are reachable.
Any other URL is refused. The caller (``fetch_doc`` tool) handles
opt-in gating; this module trusts that it is only invoked when the user
has explicitly enabled web access.
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

MAX_RESPONSE_BYTES = 8192
TIMEOUT_SECONDS = 10.0

CACHE_PATH = Path.home() / ".farcode_doc_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h
CACHE_MAX_ENTRIES = 200            # prune LRU-by-fetched_at when over

ECOSYSTEM_REGISTRIES: dict[str, callable] = {
    "python":     lambda pkg: f"https://pypi.org/pypi/{pkg}/json",
    "pypi":       lambda pkg: f"https://pypi.org/pypi/{pkg}/json",
    "javascript": lambda pkg: f"https://registry.npmjs.org/{pkg}",
    "npm":        lambda pkg: f"https://registry.npmjs.org/{pkg}",
    "rust":       lambda pkg: f"https://crates.io/api/v1/crates/{pkg}",
    "go":         lambda pkg: f"https://pkg.go.dev/{pkg}",
    "ruby":       lambda pkg: f"https://rubygems.org/api/v1/gems/{pkg}.json",
    "rubygems":   lambda pkg: f"https://rubygems.org/api/v1/gems/{pkg}.json",
    "dotnet":     lambda pkg: f"https://api.nuget.org/v3-flatcontainer/{pkg.lower()}/index.json",
    "nuget":      lambda pkg: f"https://api.nuget.org/v3-flatcontainer/{pkg.lower()}/index.json",
    "php":        lambda pkg: f"https://packagist.org/packages/{pkg}.json",
    "packagist":  lambda pkg: f"https://packagist.org/packages/{pkg}.json",
}

ALLOWED_HOSTS = frozenset({
    "pypi.org",
    "registry.npmjs.org",
    "crates.io",
    "pkg.go.dev",
    "rubygems.org",
    "api.nuget.org",
    "packagist.org",
})


def fetch(query: str, ecosystem: str = "auto") -> str:
    """Fetch package documentation. Returns trimmed text or an error string.

    Successful responses are cached on disk for ``CACHE_TTL_SECONDS`` so
    repeat lookups for the same package skip the network entirely. Error
    responses (4xx / 5xx / network failures) are NOT cached.
    """
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

    cached = _cache_get(url)
    if cached is not None:
        return cached + "\n\n[cached]"

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

    _cache_put(url, body)
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
    if ecosystem in ("ruby", "rubygems") and isinstance(data, dict):
        return _format_kv({
            "name": data.get("name"),
            "version": data.get("version"),
            "info": data.get("info"),
            "homepage_uri": data.get("homepage_uri"),
            "source_code_uri": data.get("source_code_uri"),
            "licenses": data.get("licenses"),
        })
    if ecosystem in ("dotnet", "nuget") and isinstance(data, dict):
        versions = data.get("versions") or []
        latest = versions[-1] if isinstance(versions, list) and versions else None
        count = len(versions) if isinstance(versions, list) else None
        return _format_kv({
            "latest": latest,
            "versions_count": count,
        })
    if ecosystem in ("php", "packagist") and isinstance(data, dict):
        pkg = data.get("package") or {}
        versions = pkg.get("versions") or {}
        latest = None
        if isinstance(versions, dict):
            for v in versions:
                if not str(v).lower().startswith("dev"):
                    latest = v
                    break
        return _format_kv({
            "name": pkg.get("name"),
            "description": pkg.get("description"),
            "type": pkg.get("type"),
            "latest": latest,
            "repository": pkg.get("repository"),
        })
    return body[:2000]


def _format_kv(d: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items() if v)


# ── Disk cache (24h TTL) ─────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    if len(cache) > CACHE_MAX_ENTRIES:
        # Drop the oldest entries (lowest fetched_at) to stay under the cap.
        sorted_items = sorted(cache.items(), key=lambda kv: kv[1].get("fetched_at", 0))
        cache = dict(sorted_items[-CACHE_MAX_ENTRIES:])
    try:
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _cache_get(url: str) -> str | None:
    cache = _load_cache()
    entry = cache.get(url)
    if not isinstance(entry, dict):
        return None
    fetched_at = entry.get("fetched_at")
    body = entry.get("body")
    if not isinstance(fetched_at, (int, float)) or not isinstance(body, str):
        return None
    if time.time() - fetched_at > CACHE_TTL_SECONDS:
        return None
    return body


def _cache_put(url: str, body: str) -> None:
    cache = _load_cache()
    cache[url] = {"fetched_at": time.time(), "body": body}
    _save_cache(cache)
