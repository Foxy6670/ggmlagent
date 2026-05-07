"""
KoboldCPP API client.

Uses /v1/chat/completions (OpenAI-compatible) so the model's instruct
template is applied automatically — critical for instruction-tuned models.

The raw /api/extra/generate/stream endpoint is kept only for tokenize/abort.

When KCPP_BASE_URL is set to https://aihorde.net, requests are routed through
the AI Horde async job API instead of a local KCPP instance.
"""

import json
import time
import uuid
import requests
from typing import Callable, Iterator

from config import (
    KCPP_BASE_URL, KCPP_CHAT_URL, KCPP_ABORT_URL, KCPP_TOKENIZE_URL,
    SOCKS5_PROXY, CHAT_DEFAULTS, ABORT_COOLDOWN, KCPP_CONNECT_TIMEOUT,
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
            timeout=(KCPP_CONNECT_TIMEOUT, 600),  # connect fast-fail; 10 min read for slow CPU prompt-eval
        )
        resp.raise_for_status()

        return genkey, self._iter_chat_tokens(resp, log_raw=log_raw, finish_info=finish_info)

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
            **CHAT_DEFAULTS,
            "messages": messages,
            "max_tokens": max_tokens,
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
    def _iter_chat_tokens(
        response: requests.Response,
        log_raw: "Callable[[str], None] | None" = None,
        finish_info: "list[str] | None" = None,
    ) -> Iterator[str]:
        """
        Parse OpenAI-compatible SSE stream.

        Each chunk looks like:
          data: {"choices": [{"delta": {"content": "tok"}, "finish_reason": null}]}
        Ends with:
          data: [DONE]

        Qwen3-series models (and others with native reasoning support) send
        think-block tokens in a separate "reasoning_content" field rather than
        "content".  We re-wrap those tokens with <think>…</think> tags so the
        existing in_think detection in agent.py works without any changes there.

        Every raw SSE line is forwarded to log_raw (if supplied) so that
        unexpected early terminations can be diagnosed from the session log.
        """
        in_reasoning = False   # True while we are inside a reasoning_content run

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
                # Qwen3-series models sometimes emit literal <think>/<think>
                # tags in the content field as a transition marker.  Strip them
                # here — we already inject synthetic tags above, so leaving
                # these through would cause visible stacking in the output.
                token = token.replace("<think>", "").replace("</think>", "")
                if token:
                    yield token

            fr = choices[0].get("finish_reason")
            if fr:
                if finish_info is not None:
                    finish_info.append(fr)
                break

        # If the stream ended while still inside a reasoning block, close it.
        if in_reasoning:
            yield "</think>"
