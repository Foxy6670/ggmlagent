"""
Web search and browsing via SOCKS5 proxy (Tor at localhost:9050).
Returns plain-text summaries suitable for injection into the prompt.

Rate limits:
  search — 1 per 60 s  (DDG aggressively rate-limits Tor exit nodes)
  fetch  — 1 per 5 s   (polite minimum for arbitrary sites)
"""

import re
import time
import requests
from config import SOCKS5_PROXY

_MAX_PAGE_CHARS = 200_000  # pager in commands.py handles chunking; this is a safety cap
_SEARCH_URL = "https://html.duckduckgo.com/html/"

_SEARCH_INTERVAL = 60.0
_FETCH_INTERVAL  = 5.0

_last_search: float = 0.0
_last_fetch:  float = 0.0


def _proxy_session() -> requests.Session:
    s = requests.Session()
    s.proxies = {"http": SOCKS5_PROXY, "https": SOCKS5_PROXY}
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
    )
    return s


def _strip_html(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Drop lines that are blank or contain only whitespace/punctuation noise,
    # then collapse any resulting consecutive blank lines to a single blank.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    text  = "\n".join(lines)
    return text.strip()


def search(query: str) -> str:
    """
    DuckDuckGo HTML search (no JS, works over Tor).
    Hard rate-limited to 1 call per 60 s; returns an error if called too soon.
    """
    global _last_search
    elapsed = time.monotonic() - _last_search
    if elapsed < _SEARCH_INTERVAL:
        remaining = int(_SEARCH_INTERVAL - elapsed)
        return f"[web] Search rate-limited. Try again in {remaining}s."

    try:
        s = _proxy_session()
        resp = s.post(
            _SEARCH_URL,
            data={"q": query, "b": "", "kl": "us-en"},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"[web] Search failed: {e}"
    finally:
        _last_search = time.monotonic()

    html = resp.text
    results = []
    pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(.*?)</(?:span|div)>',
        re.DOTALL | re.IGNORECASE,
    )
    for i, m in enumerate(pattern.finditer(html), 1):
        url   = m.group(1)
        title = _strip_html(m.group(2)).strip()
        snip  = _strip_html(m.group(3)).strip()
        results.append(f"{i}. {title}\n   {url}\n   {snip}")
        if i >= 8:
            break

    if not results:
        return "[web] No results found (or DDG blocked the request)."
    return "[web:search]\n" + "\n\n".join(results)


def fetch(url: str) -> str:
    """
    Fetch a URL over Tor and return plain text.
    Soft rate-limited to 1 call per 5 s.
    """
    global _last_fetch
    elapsed = time.monotonic() - _last_fetch
    if elapsed < _FETCH_INTERVAL:
        remaining = _FETCH_INTERVAL - elapsed
        return f"[web] Fetch rate-limited. Try again in {remaining:.1f}s."

    try:
        s = _proxy_session()
        resp = s.get(url, timeout=20, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return f"[web] Fetch failed: {e}"
    finally:
        _last_fetch = time.monotonic()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type or "text/plain" in content_type:
        text = _strip_html(resp.text) if "html" in content_type else resp.text
    else:
        return f"[web] Unsupported content type: {content_type}"

    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS] + "\n\n[...content capped at safety limit]"

    return f"[web:fetch {url}]\n{text}"
