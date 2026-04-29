"""
KoboldCPP API client.

Uses /v1/chat/completions (OpenAI-compatible) so the model's instruct
template is applied automatically — critical for instruction-tuned models.

The raw /api/extra/generate/stream endpoint is kept only for tokenize/abort.
"""

import json
import time
import uuid
import requests
from typing import Callable, Iterator

from config import (
    KCPP_CHAT_URL, KCPP_ABORT_URL, KCPP_TOKENIZE_URL,
    CHAT_DEFAULTS, ABORT_COOLDOWN,
)


def _make_genkey() -> str:
    return "KCPP" + uuid.uuid4().hex[:4].upper()


class KoboldClient:
    def __init__(self):
        self._session = requests.Session()

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

        Returns (genkey, token_iterator).
        genkey is passed as a KoboldCPP extension so the generation can
        be aborted mid-stream via abort().

        log_raw, if provided, is called with the repr() of every raw SSE
        line received — useful for diagnosing unexpected stream endings.
        """
        if genkey is None:
            genkey = _make_genkey()

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
            timeout=180,            # 3 min — 32k token prompts can take ~2 min to process
        )
        resp.raise_for_status()

        return genkey, self._iter_chat_tokens(resp, log_raw=log_raw, finish_info=finish_info)

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
        payload = {
            **CHAT_DEFAULTS,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = self._session.post(
            KCPP_CHAT_URL,
            json=payload,
            timeout=timeout,
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
        Abort a running generation. Waits ABORT_COOLDOWN seconds before
        returning so the server is ready for the next request.
        """
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
        resp = self._session.post(
            KCPP_TOKENIZE_URL,
            json={"prompt": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["value"]

    # ------------------------------------------------------------------
    # SSE parsers
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
