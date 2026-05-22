"""
Telegram integration — sending and receiving.

Sending: calls the Telegram Bot API directly.
Receiving: reads unread entries from tg_chat_history.jsonl (written by telegram_poll.py).
Both directions are appended to tg_chat_history.jsonl for the agent to review.

Set env vars:
  TELEGRAM_BOT_TOKEN   — from @BotFather
  TELEGRAM_CHAT_ID     — Foxo's chat ID (run telegram_poll.py to find it)
"""

import json
import os
import time
import requests
from pathlib import Path
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_HISTORY, USE_TOR

_TOR = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}
_PROXIES = _TOR if USE_TOR else None
_HISTORY = Path(TELEGRAM_HISTORY)

# Chat ID of the most recently received message — replies go here.
# Falls back to TELEGRAM_CHAT_ID for Boonie-initiated messages.
_last_incoming_chat_id: int | None = None


def _append_history(entry: dict) -> None:
    _HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def send(message: str) -> str:
    """Send a message via Telegram. Replies to the last incoming sender;
    falls back to TELEGRAM_CHAT_ID for unprompted messages."""
    global _last_incoming_chat_id
    if not TELEGRAM_BOT_TOKEN:
        return "[telegram] BOT_TOKEN not set — set TELEGRAM_BOT_TOKEN env var."

    target = _last_incoming_chat_id or (int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None)
    if not target:
        return "[telegram] No chat target — TELEGRAM_CHAT_ID not set and no incoming message received."

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": target, "text": message},
            timeout=15,
            proxies=_PROXIES,
        )
        data = resp.json()
    except Exception as e:
        return f"[telegram] Send failed: {e}"

    if data.get("ok"):
        _append_history({"direction": "out", "from": "Boonie", "text": message,
                         "chat_id": target, "ts": time.time()})
        return "[telegram] Message sent."
    return f"[telegram] Send failed: {data.get('description', str(data))}"


def history() -> list[dict]:
    """Return all chat history entries (in and out, oldest first)."""
    if not _HISTORY.exists() or _HISTORY.stat().st_size == 0:
        return []
    out: list[dict] = []
    for line in _HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def drain_inbox() -> list[dict]:
    """
    Read unread incoming messages from tg_chat_history.jsonl.
    Marks them as read by rewriting the file with read=True on each entry.
    Returns a list of unread {"from": name, "text": text} dicts.
    """
    if not _HISTORY.exists() or _HISTORY.stat().st_size == 0:
        return []

    all_entries = []
    unread = []
    for line in _HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("direction", "in") == "in" and not entry.get("read"):
            unread.append(entry)
            entry = {**entry, "read": True}
        all_entries.append(entry)

    if unread:
        global _last_incoming_chat_id
        # Track the last sender so replies are routed back to them.
        for entry in reversed(unread):
            if entry.get("chat_id"):
                _last_incoming_chat_id = int(entry["chat_id"])
                break
        # Write to a temp file then atomically rename so telegram_poll's
        # concurrent appends are never lost to a truncating open("w").
        tmp = _HISTORY.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp, _HISTORY)

    return unread
