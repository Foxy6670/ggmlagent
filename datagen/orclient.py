#!/usr/bin/env python3
"""Shared paced OpenRouter client for the datagen generators.

High worker counts + unthrottled launches invite 429s. This serializes request
LAUNCHES through a global spacing gate (workers still overlap in flight — only
the starts are spaced) and retries transient HTTP failures with exponential
backoff. All generators funnel through chat() so the pacing is process-global.
"""
import json, time, threading, urllib.request, urllib.error

URL = "https://openrouter.ai/api/v1/chat/completions"
SPACING = 0.75          # seconds between request launches (~80/min ceiling)
TRIES = 4               # 429/5xx retries: 2s, 4s, 8s backoff

_lock = threading.Lock()
_next_slot = 0.0

def _pace():
    global _next_slot
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + SPACING
    wait = start - time.monotonic()
    if wait > 0:
        time.sleep(wait)

def chat(payload, key, timeout=180):
    """POST a chat completion with launch pacing + backoff. Returns parsed JSON."""
    last = None
    for attempt in range(TRIES):
        _pace()
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://localhost/frontier-boonie",
            "X-Title": "frontier-boonie datagen"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503) and attempt < TRIES - 1:
                time.sleep(2 * 2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < TRIES - 1:
                time.sleep(2 * 2 ** attempt)
                continue
            raise
    raise last
