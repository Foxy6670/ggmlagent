#!/usr/bin/env python3
"""
Telegram background poller — run this in a separate terminal alongside agent.py.

Usage:
    export TELEGRAM_BOT_TOKEN=123456:ABCdef...
    export TELEGRAM_CHAT_ID=987654321      # optional: only accept from this chat
    python3 telegram_poll.py

What it does:
  - Long-polls the Telegram Bot API for new messages
  - Writes each message to telegram_inbox.jsonl (read by the agent on each step)
  - Prints received messages to its own terminal for visibility
  - Sends a "bot online" message when it starts

Finding your chat ID:
  Run this script without TELEGRAM_CHAT_ID set. Send any message to your bot.
  The script will print the chat ID — copy it and set TELEGRAM_CHAT_ID.
"""

import json
import socket
import sys
import time
from pathlib import Path

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_HISTORY

BOT_TOKEN = TELEGRAM_BOT_TOKEN
CHAT_ID   = TELEGRAM_CHAT_ID
INBOX     = Path(TELEGRAM_HISTORY)

if not BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not set in .secrets or environment.", file=sys.stderr)
    sys.exit(1)

# Preflight: Tor SOCKS5 must be reachable, otherwise every request explodes
# with a 60-line urllib3/socks traceback.  Fail fast with a one-liner.
_TOR_HOST, _TOR_PORT = "127.0.0.1", 9050
try:
    with socket.create_connection((_TOR_HOST, _TOR_PORT), timeout=2):
        pass
except OSError as e:
    print(
        f"ERROR: Tor SOCKS5 proxy not reachable at {_TOR_HOST}:{_TOR_PORT} ({e}).\n"
        f"  Install:  sudo apt install tor\n"
        f"  Start:    sudo systemctl enable --now tor",
        file=sys.stderr,
    )
    sys.exit(1)

BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


_TOR = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}


def tg(method: str, **params) -> dict:
    resp = requests.get(f"{BASE}/{method}", params=params, timeout=40, proxies=_TOR)
    return resp.json()


def send(chat_id: int, text: str) -> None:
    requests.post(
        f"{BASE}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
        proxies=_TOR,
    )


def main():
    print(f"[telegram_poll] Starting. Inbox: {INBOX}")
    if CHAT_ID:
        print(f"[telegram_poll] Accepting messages from chat_id={CHAT_ID}")
        send(int(CHAT_ID), "Hi, Foxo!")
    else:
        print("[telegram_poll] TELEGRAM_CHAT_ID not set — will print all incoming chat IDs.")

    offset = 0
    while True:
        try:
            data = tg("getUpdates", offset=offset, timeout=30, allowed_updates="message")
            updates = data.get("result", [])
        except Exception as e:
            print(f"[telegram_poll] Poll error: {e} — retrying in 10s")
            time.sleep(10)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            if not msg:
                continue

            chat  = msg.get("chat", {})
            cid   = chat.get("id")
            text  = msg.get("text", "").strip()
            uname = msg.get("from", {}).get("username") or msg.get("from", {}).get("first_name", "?")

            if not text:
                continue

            if CHAT_ID and str(cid) != str(CHAT_ID):
                print(f"[telegram_poll] Ignored message from unknown chat_id={cid} (@{uname}): {text[:60]}")
                continue

            if not CHAT_ID:
                print(f"[telegram_poll] Message from chat_id={cid} (@{uname}): {text[:80]}")
                print("[telegram_poll] Set TELEGRAM_CHAT_ID to accept messages from this chat.")
                continue

            # Write to chat history. Use Telegram's authoritative `date` field
            # (Unix timestamp); fall back to local clock if missing. Without
            # this, every backlog message displays as "01 Jan 00:00" and the
            # agent can't distinguish a 2-day-old instruction from a fresh one.
            ts = msg.get("date") or time.time()
            entry = json.dumps({"direction": "in", "from": uname, "text": text,
                                "chat_id": cid, "ts": ts})
            INBOX.parent.mkdir(parents=True, exist_ok=True)
            with INBOX.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
            print(f"[telegram_poll] → inbox: [{uname}] {text[:80]}")

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[telegram_poll] Stopped.")
