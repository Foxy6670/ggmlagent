#!/usr/bin/env python3
"""
Merge per-session training JSONL files into one dataset file.

Usage:
    python3 export_training.py                          # merge all in logs/
    python3 export_training.py logs/session_X.train.jsonl ...
    python3 export_training.py --out dataset.jsonl

Each input file is one session (one JSON line in OpenAI chat format).
The merged output is also JSONL — one session per line.

Stats printed to stderr so stdout can be piped cleanly:
    python3 export_training.py | wc -l   → number of sessions
"""

import argparse
import json
import sys
from pathlib import Path

from redact import redact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help=".train.jsonl files to merge")
    parser.add_argument("--out", default="-", help="output file (default: stdout)")
    parser.add_argument("--logs", default="moltbot/logs", help="log directory to scan")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        logs_dir = Path(args.logs)
        paths = sorted(logs_dir.glob("*.train.jsonl"))

    if not paths:
        print("No .train.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    out = open(args.out, "w", encoding="utf-8") if args.out != "-" else sys.stdout

    sessions = 0
    total_turns = 0
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skip {p}: {e}", file=sys.stderr)
                continue
            msgs = obj.get("messages", [])
            # Redact here too, defensively -- curate_training_data.py already
            # does this, but this script can also run directly against raw
            # .train.jsonl (uncurated), so don't rely on that step happening.
            for m in msgs:
                m["content"] = redact(m["content"])
            # count assistant turns as a proxy for training signal
            turns = sum(1 for m in msgs if m["role"] == "assistant")
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sessions += 1
            total_turns += turns

    if args.out != "-":
        out.close()

    print(f"Exported {sessions} sessions, {total_turns} assistant turns → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
