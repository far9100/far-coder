"""Tests for the scoped fetch_doc tool and web allowlist."""

import json

import pytest

from farcode import web
from farcode import tools


@pytest.fixture(autouse=True)
def web_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FARCODE_ALLOW_WEB", raising=False)
    tools.set_web_enabled(False)
    yield
    tools.set_web_enabled(False)


# ── Allowlist & input validation ─────────────────────────────────────────────

def test_url_allowlist_accepts_known_hosts():
    assert web._url_in_allowlist("https://pypi.org/pypi/httpx/json")
    assert web._url_in_allowlist("https://registry.npmjs.org/react")
    assert web._url_in_allowlist("https://crates.io/api/v1/crates/serde")
    assert web._url_in_allowlist("https://pkg.go.dev/github.com/gin-gonic/gin")


def test_url_allowlist_rejects_unknown_hosts():
    assert not web._url_in_allowlist("https://example.com/anything")
    assert not web._url_in_allowlist("http://pypi.org/pypi/httpx/json")  # http not allowed
    assert not web._url_in_allowlist("https://evil.pypi.org.attacker.com/")


def test_safe_package_name_accepts_normal_names():
    assert web._is_safe_package_name("httpx")
    assert web._is_safe_package_name("@types/node")
    assert web._is_safe_package_name("python-dotenv")
    assert web._is_safe_package_name("github.com/gin-gonic/gin")


def test_safe_package_name_rejects_injection_chars():
    assert not web._is_safe_package_name("foo;rm -rf")
    assert not web._is_safe_package_name("foo bar")
    assert not web._is_safe_package_name("foo?q=1")
    assert not web._is_safe_package_name("../etc/passwd")  # path traversal
    assert not web._is_safe_package_name("foo/../bar")     # embedded traversal


def test_fetch_rejects_empty_query():
    out = web.fetch("", "pypi")
    assert "error" in out.lower()


def test_fetch_rejects_unknown_ecosystem():
    out = web.fetch("httpx", "haskell")
    assert "error" in out.lower()
    assert "unknown ecosystem" in out.lower()


def test_fetch_rejects_unsafe_package_name():
    out = web.fetch("foo;rm", "pypi")
    assert "error" in out.lower()
    assert "invalid" in out.lower()


# ── Mocked HTTP responses ────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200, content_type: str = "application/json"):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}


@pytest.fixture
def mock_httpx(monkeypatch):
    captured = {"calls": []}

    def fake_get(url, **kwargs):
        captured["calls"].append({"url": url, **kwargs})
        return captured["next_response"]

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "get", fake_get)
    return captured


def test_fetch_pypi_summarizes_json(mock_httpx):
    payload = json.dumps({
        "info": {
            "name": "httpx",
            "version": "0.28.1",
            "summary": "The next generation HTTP client.",
            "home_page": "https://www.python-httpx.org/",
            "license": "BSD-3-Clause",
            "requires_python": ">=3.8",
        }
    })
    mock_httpx["next_response"] = _FakeResponse(payload, 200, "application/json")
    out = web.fetch("httpx", "pypi")
    assert "name: httpx" in out
    assert "version: 0.28.1" in out
    assert "license: BSD-3-Clause" in out
    assert mock_httpx["calls"][0]["url"] == "https://pypi.org/pypi/httpx/json"


def test_fetch_npm_summarizes_json(mock_httpx):
    payload = json.dumps({
        "name": "react",
        "description": "React is a JavaScript library for building user interfaces.",
        "dist-tags": {"latest": "18.3.1"},
        "license": "MIT",
        "homepage": "https://reactjs.org/",
    })
    mock_httpx["next_response"] = _FakeResponse(payload, 200, "application/json")
    out = web.fetch("react", "npm")
    assert "name: react" in out
    assert "latest: 18.3.1" in out
    assert "MIT" in out
    assert mock_httpx["calls"][0]["url"] == "https://registry.npmjs.org/react"


def test_fetch_crates_summarizes_json(mock_httpx):
    payload = json.dumps({
        "crate": {
            "name": "serde",
            "description": "A generic serialization/deserialization framework",
            "max_version": "1.0.219",
            "homepage": "https://serde.rs",
        }
    })
    mock_httpx["next_response"] = _FakeResponse(payload, 200, "application/json")
    out = web.fetch("serde", "rust")
    assert "name: serde" in out
    assert "max_version: 1.0.219" in out


def test_fetch_404_returns_friendly_message(mock_httpx):
    mock_httpx["next_response"] = _FakeResponse("Not found", 404, "text/plain")
    out = web.fetch("nonexistent-pkg", "pypi")
    assert "not found" in out.lower()


def test_fetch_5xx_surfaces_status(mock_httpx):
    mock_httpx["next_response"] = _FakeResponse("oops", 502, "text/plain")
    out = web.fetch("httpx", "pypi")
    assert "502" in out


def test_fetch_html_is_stripped(mock_httpx):
    html = (
        "<html><head><title>t</title>"
        "<script>alert('xss')</script>"
        "<style>body{color:red}</style>"
        "</head><body>"
        "<nav>menu</nav>"
        "<h1>gin web framework</h1>"
        "<p>HTTP web framework written in Go.</p>"
        "<footer>copyright</footer>"
        "</body></html>"
    )
    mock_httpx["next_response"] = _FakeResponse(html, 200, "text/html")
    out = web.fetch("github.com/gin-gonic/gin", "go")
    assert "gin web framework" in out
    assert "HTTP web framework" in out
    # script/style/nav/footer must be stripped
    assert "alert" not in out
    assert "color:red" not in out
    assert "menu" not in out
    assert "copyright" not in out


def test_fetch_truncates_oversize_response(mock_httpx):
    big = "x" * (web.MAX_RESPONSE_BYTES * 3)
    mock_httpx["next_response"] = _FakeResponse(big, 200, "text/plain")
    out = web.fetch("foo", "pypi")
    assert len(out) <= web.MAX_RESPONSE_BYTES + 200  # cap + truncation note
    assert "truncated" in out


def test_fetch_handles_network_exception(mock_httpx, monkeypatch):
    import httpx as _httpx
    def boom(*a, **kw):
        raise _httpx.ConnectError("simulated")
    monkeypatch.setattr(_httpx, "get", boom)
    out = web.fetch("httpx", "pypi")
    assert "error" in out.lower()
    assert "simulated" in out.lower()


# ── tool gating ──────────────────────────────────────────────────────────────

def test_fetch_doc_tool_disabled_by_default():
    out = tools.execute_tool("fetch_doc", {"query": "httpx"})
    assert "disabled" in out.lower()
    assert "--allow-web" in out


def test_fetch_doc_tool_enabled_via_setter(monkeypatch):
    captured = {}

    def fake_fetch(query, ecosystem):
        captured["q"] = query
        captured["e"] = ecosystem
        return "FAKE OK"

    monkeypatch.setattr(web, "fetch", fake_fetch)
    tools.set_web_enabled(True)
    out = tools.execute_tool("fetch_doc", {"query": "httpx", "ecosystem": "pypi"})
    assert out == "FAKE OK"
    assert captured == {"q": "httpx", "e": "pypi"}


def test_fetch_doc_tool_enabled_via_env(monkeypatch):
    captured = {}

    def fake_fetch(query, ecosystem):
        captured["q"] = query
        return "ENV OK"

    monkeypatch.setattr(web, "fetch", fake_fetch)
    monkeypatch.setenv("FARCODE_ALLOW_WEB", "1")
    out = tools.execute_tool("fetch_doc", {"query": "httpx"})
    assert out == "ENV OK"


def test_fetch_doc_schema_present_in_tool_schemas():
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "fetch_doc" in names


def test_fetch_doc_not_in_subagent_tools():
    """Subagent must not see fetch_doc — it is not in the read-only set."""
    from farcode import subagent
    names = {s["function"]["name"] for s in subagent.get_subagent_tools()}
    assert "fetch_doc" not in names
