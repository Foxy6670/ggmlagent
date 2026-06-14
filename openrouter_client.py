"""
OpenRouter API client.

Drop-in replacement for KoboldClient when OPENROUTER_API_KEY is set in .secrets.
Speaks the same OpenAI-compatible SSE format; agent.py needs no changes.

Extra behaviour vs KCPP:
  - OR_MIN_TURN_GAP: mandatory sleep between turns (reimpose the throttle that
    local inference gives for free — cloud is fast enough to burn 1M tok/day in
    minutes without it).
  - OR_DAILY_TOKEN_BUDGET: hard cap; raises RuntimeError when exceeded so the
    harness surfaces it cleanly rather than silently blowing the quota.
  - abort(): closes the active HTTP response (no server-side abort endpoint).
  - tokenize(): approximates at 4 chars/token (no OR tokenize endpoint).

Configuration (all settable in .secrets):
  OPENROUTER_API_KEY        — Bearer token (required)
  OPENROUTER_MODEL          — model string  (default: meta-llama/llama-3.1-70b-instruct:free)
  OR_MIN_TURN_GAP           — seconds between stream end and next call (default: 30)
  OR_DAILY_TOKEN_BUDGET     — total token cap for this harness run (default: 900000)
"""

import json
import queue as _queue
import threading as _threading
import time
import uuid
import requests
from typing import Callable, Iterator

from config import (
    OPENROUTER_MODEL,
    OR_MIN_TURN_GAP,
    OR_DAILY_TOKEN_BUDGET,
    OPENROUTER_API_KEY,
    CHAT_DEFAULTS,
)

_API_URL          = "https://openrouter.ai/api/v1/chat/completions"
_CONNECT_TIMEOUT  = 15    # s — cloud connect is fast; fail quickly on auth errors
_FIRST_TOKEN_TIMEOUT = 30  # s — cloud inference starts in seconds, not minutes
_INTER_TOKEN_TIMEOUT = 20  # s — streaming tokens arrive continuously once started


def _make_genkey() -> str:
    return "OR" + uuid.uuid4().hex[:4].upper()


class OpenRouterClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "X-Title": "ggmlagent",
        })
        self._last_stream_end: float = 0.0
        self._daily_tokens: int = 0
        self._active_resp: requests.Response | None = None
        self._resp_lock = _threading.Lock()

    # ------------------------------------------------------------------
    # Chat completions (primary generation path)
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict],
        genkey: str | None = None,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
        **overrides,
    ) -> tuple[str, "Iterator[str]"]:
        # Enforce inter-turn gap — reimpose the throttle local inference gives for free.
        gap = OR_MIN_TURN_GAP - (time.monotonic() - self._last_stream_end)
        if gap > 0:
            time.sleep(gap)

        if self._daily_tokens >= OR_DAILY_TOKEN_BUDGET:
            raise RuntimeError(
                f"[openrouter] Daily token budget exhausted "
                f"({self._daily_tokens:,} / {OR_DAILY_TOKEN_BUDGET:,} tokens used)"
            )

        if genkey is None:
            genkey = _make_genkey()

        payload = {
            **{k: v for k, v in CHAT_DEFAULTS.items() if k not in ("stream", "genkey")},
            **{k: v for k, v in overrides.items()},
            "model":   OPENROUTER_MODEL,
            "messages": messages,
            "stream":  True,
            "stream_options": {"include_usage": True},  # usage in final chunk
        }

        resp = self._session.post(
            _API_URL,
            json=payload,
            stream=True,
            timeout=(_CONNECT_TIMEOUT, None),
        )
        resp.raise_for_status()

        with self._resp_lock:
            self._active_resp = resp

        return genkey, self._iter_chat_tokens(resp, log_raw=log_raw, finish_info=finish_info)

    def chat_complete_sync(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> str:
        payload = {
            "model":       OPENROUTER_MODEL,
            "messages":    messages,
            "stream":      False,
            "max_tokens":  max_tokens,
            "temperature": 0.1,
            "top_p":       0.9,
        }
        resp = self._session.post(
            _API_URL,
            json=payload,
            timeout=(_CONNECT_TIMEOUT, timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        self._daily_tokens += data.get("usage", {}).get("total_tokens", 0)
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    # ------------------------------------------------------------------
    # Abort / tokenize
    # ------------------------------------------------------------------

    def abort(self, genkey: str) -> bool:
        # No server-side abort — close the response to unblock the reader thread.
        with self._resp_lock:
            if self._active_resp is not None:
                self._active_resp.close()
        return False

    def tokenize(self, text: str) -> int:
        # OpenRouter has no tokenize endpoint; 4 chars/token is a reasonable estimate.
        return len(text) // 4

    # ------------------------------------------------------------------
    # SSE parsing + timeout wrapper
    # ------------------------------------------------------------------

    def _iter_chat_tokens(
        self,
        response: requests.Response,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
    ) -> "Iterator[str]":
        """Wrap _parse_sse_stream with per-token timeouts and usage tracking."""
        token_q: "_queue.Queue[str | BaseException | object]" = _queue.Queue()
        usage_out: list[int] = []
        _DONE = object()

        def _reader() -> None:
            try:
                for tok in self._parse_sse_stream(
                    response, log_raw=log_raw, finish_info=finish_info,
                    usage_out=usage_out,
                ):
                    token_q.put(tok)
            except Exception as exc:  # noqa: BLE001
                token_q.put(exc)
            finally:
                token_q.put(_DONE)

        _threading.Thread(target=_reader, daemon=True).start()

        first = True
        try:
            while True:
                timeout = _FIRST_TOKEN_TIMEOUT if first else _INTER_TOKEN_TIMEOUT
                try:
                    item = token_q.get(timeout=timeout)
                except _queue.Empty:
                    response.close()
                    label = "first-token" if first else "inter-token"
                    raise TimeoutError(
                        f"[openrouter] {label} timeout ({timeout}s) — stream hung"
                    )
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                first = False
                yield item
        finally:
            if usage_out:
                self._daily_tokens += usage_out[0]
            self._last_stream_end = time.monotonic()
            with self._resp_lock:
                self._active_resp = None

    @staticmethod
    def _parse_sse_stream(
        response: requests.Response,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
        usage_out: "list[int] | None" = None,
    ) -> "Iterator[str]":
        """
        Parse OpenAI-compatible SSE stream — identical format to KCPP's chat path.

        Handles reasoning_content for models that expose chain-of-thought
        (e.g. Qwen3 on OpenRouter) by re-wrapping with <think>…</think> tags
        so agent.py's detection works unchanged.
        """
        in_reasoning = False

        for raw_line in response.iter_lines(decode_unicode=True):
            if log_raw:
                log_raw(repr(raw_line))

            if not raw_line.startswith("data:"):
                continue
            payload = raw_line[5:].strip()
            if payload == "[DONE]":
                break
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Usage chunk (stream_options include_usage=true) — no choices.
            if "usage" in data and not data.get("choices"):
                if usage_out is not None:
                    usage_out.append(data["usage"].get("total_tokens", 0))
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            reasoning = delta.get("reasoning_content") or ""
            token     = delta.get("content") or ""

            if reasoning:
                if not in_reasoning:
                    in_reasoning = True
                    yield "<think>"
                yield reasoning

            if token:
                if in_reasoning:
                    in_reasoning = False
                    yield "</think>"
                token = token.replace("<think>", "").replace("</think>", "")
                if token:
                    yield token

            fr = choices[0].get("finish_reason")
            if fr and finish_info is not None:
                finish_info.append(fr)
            # Don't break on finish_reason — the usage chunk arrives after it,
            # before [DONE].  Let [DONE] terminate the loop.

        if in_reasoning:
            yield "</think>"
