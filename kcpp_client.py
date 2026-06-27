"""
KoboldCPP API client.

Uses /v1/chat/completions (OpenAI-compatible) so the model's instruct
template is applied automatically — critical for instruction-tuned models.

The raw /api/extra/generate/stream endpoint is kept only for tokenize/abort.

When KCPP_BASE_URL is set to https://aihorde.net, requests are routed through
the AI Horde async job API instead of a local KCPP instance.
"""

import json
import queue as _queue
import threading as _threading
import time
import uuid
import requests
from typing import Callable, Iterator

from config import (
    KCPP_BASE_URL, KCPP_CHAT_URL, KCPP_ABORT_URL, KCPP_TOKENIZE_URL,
    SOCKS5_PROXY, CHAT_DEFAULTS, ABORT_COOLDOWN, KCPP_CONNECT_TIMEOUT,
    KCPP_FIRST_TOKEN_TIMEOUT, KCPP_PREFILL_RATE, KCPP_INTER_TOKEN_TIMEOUT,
    HORDE_API_KEY, HORDE_MODELS,
)

_USING_HORDE = "aihorde.net" in KCPP_BASE_URL

_HORDE_ASYNC_URL  = "https://aihorde.net/api/v2/generate/text/async"
_HORDE_CHECK_URL  = "https://aihorde.net/api/v2/generate/text/check/{}"
_HORDE_STATUS_URL = "https://aihorde.net/api/v2/generate/text/status/{}"
_HORDE_CANCEL_URL = "https://aihorde.net/api/v2/generate/text/status/{}"  # DELETE


def _make_genkey() -> str:
    return "KCPP" + uuid.uuid4().hex[:4].upper()


def _messages_to_chatml(messages: list[dict]) -> str:
    """Format a messages list into a ChatML/Qwen3 prompt string."""
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class KoboldClient:
    def __init__(self):
        self._session = requests.Session()
        if ".onion" in KCPP_BASE_URL:
            # .onion hostnames only resolve through the SOCKS5 proxy.
            self._session.proxies = {"http": SOCKS5_PROXY, "https": SOCKS5_PROXY}
        if _USING_HORDE and HORDE_API_KEY:
            self._session.headers.update({"apikey": HORDE_API_KEY})

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
        """
        Stream a chat completion.

        Returns (genkey, token_iterator).  For KCPP, genkey enables mid-stream
        abort.  For Horde, it carries the job ID so abort() can cancel the job.
        """
        if genkey is None:
            genkey = _make_genkey()

        if _USING_HORDE:
            return self._horde_chat_stream(messages, genkey, finish_info, **overrides)

        payload = {
            **CHAT_DEFAULTS,
            **overrides,
            "messages": messages,
            "genkey": genkey,       # KoboldCPP extension — enables abort()
        }

        resp = self._session.post(
            KCPP_CHAT_URL,
            json=payload,
            stream=True,
            timeout=(KCPP_CONNECT_TIMEOUT, None),  # connect fast-fail; per-token timeouts enforced in _iter_chat_tokens
        )
        resp.raise_for_status()

        # Prefill-aware first-token deadline: the first token can't arrive until
        # the whole prompt is prefilled, so budget for that.  ~4 chars/token is
        # a cheap estimate (good enough for a timeout — no tokenize round-trip).
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        first_token_timeout = KCPP_FIRST_TOKEN_TIMEOUT + (prompt_chars / 4) / KCPP_PREFILL_RATE

        return genkey, self._iter_chat_tokens(
            resp, log_raw=log_raw, finish_info=finish_info,
            first_token_timeout=first_token_timeout,
        )

    def _horde_chat_stream(
        self,
        messages: list[dict],
        genkey: str,
        finish_info: "list[str] | None" = None,
        **overrides,
    ) -> tuple[str, "Iterator[str]"]:
        """
        Submit to AI Horde, poll until done, return (job_id, word_iterator).
        The iterator yields the response word-by-word to simulate streaming.
        """
        max_tokens = overrides.get("max_tokens", CHAT_DEFAULTS.get("max_tokens", 512))
        prompt = _messages_to_chatml(messages)

        payload = {
            "prompt": prompt,
            "params": {
                "max_length":         min(max_tokens, 2048),  # Horde per-worker cap; workers may clamp further
                "max_context_length": 4096,
                "temperature":        overrides.get("temperature", CHAT_DEFAULTS.get("temperature", 0.7)),
                "top_p":              overrides.get("top_p",        CHAT_DEFAULTS.get("top_p", 0.9)),
                "stop_sequence":      ["</tool_call>"],
            },
            "models": HORDE_MODELS if HORDE_MODELS else [],
        }

        resp = self._session.post(
            _HORDE_ASYNC_URL,
            json=payload,
            timeout=(KCPP_CONNECT_TIMEOUT, 30),
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]

        def _poll_and_stream() -> Iterator[str]:
            check_url = _HORDE_CHECK_URL.format(job_id)
            while True:
                time.sleep(3)
                check = self._session.get(check_url, timeout=(KCPP_CONNECT_TIMEOUT, 10))
                check.raise_for_status()
                data = check.json()
                if data.get("faulted"):
                    raise RuntimeError(f"Horde job faulted: {data}")
                if data.get("done"):
                    break

            status = self._session.get(
                _HORDE_STATUS_URL.format(job_id),
                timeout=(KCPP_CONNECT_TIMEOUT, 30),
            )
            status.raise_for_status()
            text = status.json()["generations"][0]["text"]

            if finish_info is not None:
                finish_info.append("stop")

            # Yield word-by-word to simulate streaming for the agent loop.
            words = text.split(" ")
            for i, word in enumerate(words):
                yield word if i == len(words) - 1 else word + " "

        # Return job_id as the "genkey" so abort() can cancel the job.
        return job_id, _poll_and_stream()

    def chat_complete_sync(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> str:
        """
        Non-streaming chat completion. Returns the full response text.
        Used for compaction summaries where we want a single result without
        streaming overhead.
        """
        if _USING_HORDE:
            # Reuse the async path; collect all tokens into one string.
            _, token_iter = self._horde_chat_stream(
                messages, _make_genkey(), max_tokens=max_tokens
            )
            return "".join(token_iter).strip()

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False,
        }
        resp = self._session.post(
            KCPP_CHAT_URL,
            json=payload,
            timeout=(KCPP_CONNECT_TIMEOUT, timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def abort(self, genkey: str) -> bool:
        """
        Abort a running generation.
        For KCPP: POSTs to the abort endpoint and waits ABORT_COOLDOWN.
        For Horde: DELETEs the job (genkey is the job ID).
        """
        if _USING_HORDE:
            try:
                self._session.delete(
                    _HORDE_CANCEL_URL.format(genkey),
                    timeout=10,
                )
            except Exception:
                pass
            return False
        try:
            resp = self._session.post(
                KCPP_ABORT_URL,
                json={"genkey": genkey},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("success") in (True, "true")
        except Exception:
            return False
        finally:
            time.sleep(ABORT_COOLDOWN)

    # ------------------------------------------------------------------
    # Tokenize (used for context-length budget checks)
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> int:
        if _USING_HORDE:
            # Horde has no tokenize endpoint — approximate with char count.
            return len(text) // 4
        resp = self._session.post(
            KCPP_TOKENIZE_URL,
            json={"prompt": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["value"]

    # ------------------------------------------------------------------
    # SSE parsers (KCPP only)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sse_stream(
        response: requests.Response,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
    ) -> "Iterator[str]":
        """
        Parse OpenAI-compatible SSE stream — yields tokens with no timeout
        logic.  Called from _iter_chat_tokens which enforces per-token timeouts.

        Qwen3-series models send think-block tokens in "reasoning_content";
        we re-wrap with <think>…</think> so agent.py's detection works unchanged.
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
            if fr:
                if finish_info is not None:
                    finish_info.append(fr)
                break

        if in_reasoning:
            yield "</think>"

    @staticmethod
    def _iter_chat_tokens(
        response: requests.Response,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
        first_token_timeout: float = KCPP_FIRST_TOKEN_TIMEOUT,
    ) -> "Iterator[str]":
        """
        Wrap _parse_sse_stream with per-token timeouts using a thread+queue.

        Two separate deadlines:
          - first_token_timeout: max wait for the very first token.  Computed by
            the caller as KCPP_FIRST_TOKEN_TIMEOUT + prefill estimate, since the
            first token can't arrive until the whole prompt is prefilled and that
            scales with context size (a flat budget kills slow-but-honest prefill).
          - KCPP_INTER_TOKEN_TIMEOUT: max gap between any two consecutive tokens
            once generation has started (~15 s at 3 tok/s normal rate).

        On timeout, response.close() is called to unblock the reader thread,
        then TimeoutError is raised so the caller (agent.py) can abort+retry.
        """
        token_q: "_queue.Queue[str | BaseException | object]" = _queue.Queue()
        _DONE = object()

        def _reader() -> None:
            try:
                for tok in KoboldClient._parse_sse_stream(
                    response, log_raw=log_raw, finish_info=finish_info
                ):
                    token_q.put(tok)
            except Exception as exc:  # noqa: BLE001
                token_q.put(exc)
            finally:
                token_q.put(_DONE)

        _threading.Thread(target=_reader, daemon=True).start()

        first = True
        while True:
            timeout = first_token_timeout if first else KCPP_INTER_TOKEN_TIMEOUT
            try:
                item = token_q.get(timeout=timeout)
            except _queue.Empty:
                response.close()
                label = "first-token" if first else "inter-token"
                raise TimeoutError(
                    f"[kcpp] {label} timeout ({timeout:.0f}s) — generation hung"
                )
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            first = False
            yield item
