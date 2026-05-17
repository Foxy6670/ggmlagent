#!/usr/bin/env python3
"""
Convert Telegram export into fine-tuning training examples.

Reads result.json from a Telegram Desktop export, strips automated startup
messages, groups Foxo→Boonie exchanges into conversation threads, and writes
telegram_training.jsonl in the same OpenAI chat format as the session files.

Each output example is one complete conversation thread: a run of messages
between two session-restart gaps, formatted as system + user/assistant pairs.

Usage:
    python3 merge_telegram_training.py result.json [output.jsonl]
    python3 merge_telegram_training.py result.json --stats
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Startup message patterns to strip
# ---------------------------------------------------------------------------
_STARTUP_RE = re.compile(
    r"^(Hi|Hello|Hey),?\s*Foxo[!.]?\s*$",
    re.IGNORECASE,
)

# Gap between messages that signals a new conversation thread (seconds).
# 4 hours: keeps related back-and-forth together while splitting across days.
_THREAD_GAP_SECS = 4 * 60 * 60

# If a Boonie message shares this fraction of words with the previous Boonie
# message, treat it as a duplicate and drop it.
_DEDUP_THRESHOLD = 0.75


def get_text(msg: dict) -> str:
    """Extract plain text from a Telegram message (handles entity lists)."""
    t = msg.get("text", "")
    if isinstance(t, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in t)
    return t


def is_startup(msg: dict) -> bool:
    return bool(_STARTUP_RE.match(get_text(msg).strip()))


def format_sender(msg: dict) -> str:
    """Return '[Name @ Telegram]' prefix matching the harness format."""
    name = msg.get("from", "Unknown")
    return f"[{name} @ Telegram]"


def load_messages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [m for m in data["messages"] if m.get("type") == "message"]


def split_threads(messages: list[dict]) -> list[list[dict]]:
    """
    Split the message list into conversation threads separated by long gaps.
    A new thread starts when there's a >90-minute silence.
    """
    if not messages:
        return []
    threads: list[list[dict]] = [[messages[0]]]
    for msg in messages[1:]:
        gap = int(msg["date_unixtime"]) - int(threads[-1][-1]["date_unixtime"])
        if gap > _THREAD_GAP_SECS:
            threads.append([])
        threads[-1].append(msg)
    return threads


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets — cheap near-duplicate check."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def thread_to_training_example(thread: list[dict]) -> dict | None:
    """
    Convert a conversation thread into a training example.
    Returns None if the thread has no real content after filtering.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Begin. Read your task file first."},
    ]

    assistant_turns = 0
    last_boonie_text = ""
    for msg in thread:
        text = get_text(msg).strip()
        if not text:
            continue
        if is_startup(msg):
            continue

        is_boonie = msg["from"] == "Boonie"

        if is_boonie:
            # Drop near-duplicate consecutive Boonie messages (session wrap-up spam)
            if _word_overlap(text, last_boonie_text) >= _DEDUP_THRESHOLD:
                continue
            last_boonie_text = text
            messages.append({"role": "assistant", "content": text})
            assistant_turns += 1
        else:
            prefix = format_sender(msg)
            messages.append({"role": "user", "content": f"{prefix}: {text}"})

    # Need at least one real Foxo message and one real Boonie response
    has_foxo = any(m["role"] == "user" and "@ Telegram]" in m["content"]
                   for m in messages)
    if not has_foxo or assistant_turns == 0:
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to result.json from Telegram export")
    parser.add_argument(
        "output",
        nargs="?",
        default="telegram_training.jsonl",
        help="Output .jsonl file (default: telegram_training.jsonl)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics only, don't write output",
    )
    args = parser.parse_args()

    all_msgs = load_messages(Path(args.input))

    boonie_total  = sum(1 for m in all_msgs if m["from"] == "Boonie")
    startup_count = sum(1 for m in all_msgs if is_startup(m))
    foxo_total    = sum(1 for m in all_msgs if m["from"] != "Boonie")

    print(f"Messages total:    {len(all_msgs)}")
    print(f"  From Boonie:     {boonie_total}  ({startup_count} startup, {boonie_total - startup_count} real)")
    print(f"  From Foxo:       {foxo_total}")

    threads = split_threads(all_msgs)
    print(f"Conversation threads (>{_THREAD_GAP_SECS//60}m gap): {len(threads)}")

    examples = []
    skipped  = 0
    for thread in threads:
        ex = thread_to_training_example(thread)
        if ex:
            examples.append(ex)
        else:
            skipped += 1

    print(f"Training examples: {len(examples)}  (skipped {skipped} startup-only threads)")

    if args.stats:
        # Sample a couple of threads
        print()
        for i, ex in enumerate(examples[:2]):
            print(f"--- Example {i} ({len(ex['messages'])} messages) ---")
            for m in ex["messages"]:
                if m["role"] == "system":
                    continue
                content = m["content"][:120].replace("\n", " ")
                print(f"  [{m['role']:9s}] {content}")
            print()
        return

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Count assistant turns across all examples
    asst_turns = sum(
        sum(1 for m in ex["messages"] if m["role"] == "assistant")
        for ex in examples
    )
    print(f"Total assistant turns: {asst_turns}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
