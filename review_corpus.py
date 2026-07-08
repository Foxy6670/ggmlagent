#!/usr/bin/env python3
"""
Build a review manifest for Boonie's pre-corpus approval pass (V3.3+).

Phase 1 of review mode: pure discovery/planning, zero LLM calls. Curates
and redacts every real session log the same way curate_training_data.py
already does, adds Telegram history as its own candidate, splits anything
too large to review in one pass into ordered segments (never mid-turn),
and orders everything largest-first so the highest-signal content gets
reviewed even if time runs out partway through.

This produces a manifest file for a separate review-runner script to work
through -- deliberately kept apart so the (zero-cost, fully auditable)
planning step can be checked before any tokens are spent on the actual
review conversation.

Usage:
    python3 review_corpus.py <logs_dir> [--telegram tg_chat_history.jsonl] [--out manifest.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from curate_training_data import curate_messages
from redact import redact

_CHARS_PER_TOKEN = 4       # rough estimate for planning, not exact
_SEGMENT_TOKEN_BUDGET = 8000  # leaves real headroom for the review prompt +
                               # carried-forward context + response, well
                               # inside a 32k ctx / 6144 response reserve


def render_session(messages: list[dict]) -> str:
    """Curated messages -> plain text, one block per turn."""
    return "\n".join(f"[{m['role']}]\n{m['content']}\n" for m in messages)


def chunk_by_turns(messages: list[dict], budget_chars: int) -> list[list[dict]]:
    """Split into ordered segments under budget_chars, on message
    boundaries only -- never cuts a single turn in half."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for m in messages:
        m_len = len(m["content"])
        if current and current_len + m_len > budget_chars:
            segments.append(current)
            current, current_len = [], 0
        current.append(m)
        current_len += m_len
    if current:
        segments.append(current)
    return segments


def build_telegram_candidate(tg_path: Path, budget_chars: int) -> dict | None:
    if not tg_path.exists():
        return None
    entries = []
    for line in tg_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not entries:
        return None

    lines = []
    for e in entries:
        who = e.get("from", "?") if e.get("direction") == "in" else "Boonie"
        lines.append(f"{who}: {redact(e.get('text', ''))}")

    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for ln in lines:
        if current and current_len + len(ln) > budget_chars:
            segments.append("\n".join(current))
            current, current_len = [], 0
        current.append(ln)
        current_len += len(ln)
    if current:
        segments.append("\n".join(current))

    text = "\n".join(lines)
    return {
        "source": "telegram_history", "kind": "telegram",
        "size": len(text), "n_segments": len(segments), "segments": segments,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs_dir", help="directory of raw session_*.train.jsonl files")
    ap.add_argument("--telegram", default=None, help="path to tg_chat_history.jsonl")
    ap.add_argument("--out", default="review_manifest.json")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    budget_chars = _SEGMENT_TOKEN_BUDGET * _CHARS_PER_TOKEN

    candidates = []
    dropped = 0
    for path in sorted(logs_dir.glob("*.train.jsonl")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        obj = json.loads(raw)
        curated, _stats = curate_messages(obj.get("messages", []))
        if curated is None:
            dropped += 1
            continue
        segs_msgs = chunk_by_turns(curated, budget_chars)
        segs_text = [render_session(s) for s in segs_msgs]
        candidates.append({
            "source": path.name, "kind": "session",
            "size": sum(len(s) for s in segs_text),
            "n_segments": len(segs_text), "segments": segs_text,
        })

    if args.telegram:
        tg = build_telegram_candidate(Path(args.telegram), budget_chars)
        if tg:
            candidates.append(tg)

    candidates.sort(key=lambda c: -c["size"])

    manifest = {
        "candidates": candidates,
        "total_candidates": len(candidates),
        "total_segments": sum(c["n_segments"] for c in candidates),
        "sessions_dropped_too_short": dropped,
    }
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"Candidates: {len(candidates)} "
          f"(sessions: {sum(1 for c in candidates if c['kind'] == 'session')}, "
          f"telegram: {sum(1 for c in candidates if c['kind'] == 'telegram')})")
    print(f"Sessions dropped (<3 good turns, same threshold as curate_training_data.py): {dropped}")
    print(f"Total review segments: {manifest['total_segments']}")
    print(f"Manifest written to {args.out}")
    print()
    print("Order (largest first):")
    for c in candidates:
        print(f"  {c['size']:>8,}b  {c['n_segments']:2d} seg(s)  {c['source']}")


if __name__ == "__main__":
    main()
