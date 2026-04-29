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
import time
import requests
from pathlib import Path
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_HISTORY

_TOR = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}
_HISTORY = Path(TELEGRAM_HISTORY)


def _append_history(entry: dict) -> None:
    _HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def send(message: str) -> str:
    """Send a message to Foxo via Telegram. Returns a result string."""
    if not TELEGRAM_BOT_TOKEN:
        return "[telegram] BOT_TOKEN not set — set TELEGRAM_BOT_TOKEN env var."
    if not TELEGRAM_CHAT_ID:
        return "[telegram] CHAT_ID not set — set TELEGRAM_CHAT_ID env var."

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": int(TELEGRAM_CHAT_ID), "text": message},
            timeout=15,
            proxies=_TOR,
        )
        data = resp.json()
    except Exception as e:
        return f"[telegram] Send failed: {e}"

    if data.get("ok"):
        _append_history({"direction": "out", "from": "Boonie", "text": message, "ts": time.time()})
        return "[telegram] Message sent to Foxo."
    return f"[telegram] Send failed: {data.get('description', str(data))}"


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
        with _HISTORY.open("w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")

    return unread
