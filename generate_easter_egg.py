#!/usr/bin/env python3
"""
Generate synthetic training turns where Boonie reads the harness source files.

These turns serve as an easter egg: the fine-tuned model will have implicit
knowledge of its own harness, having "read" the source once during training.
They also teach the model to reason about code it reads via /read.

Output format matches extract_training.py (single JSONL record).

Usage:
    python3 generate_easter_egg.py --out easter_egg.train.jsonl
    python3 generate_easter_egg.py >> boonie_corpus.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SYSTEM_PROMPT

HERE = Path(__file__).parent


def _read_obs(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = "\n".join(f"{i+1:4}  {line}" for i, line in enumerate(lines))
    return f"[read: {path.name}]\n{numbered}"


def _turn(think: str, narration: str, command: str) -> dict:
    content = (
        f"<think>\n{think.strip()}\n</think>\n\n"
        f"{narration.strip()}\n"
        f"```\n{command}\n```"
    )
    return {"role": "assistant", "content": content}


def _obs(text: str) -> dict:
    return {"role": "user", "content": text.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="-", help="output file (default: stdout)")
    args = parser.parse_args()

    commands_py = HERE / "commands.py"
    main_py     = HERE / "main.py"
    moltbook_py = HERE / "moltbook.py"

    for p in (commands_py, main_py, moltbook_py):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Begin. Read your task file first."},
    ]

    # ── Turn 1: read commands.py ─────────────────────────────────────────────
    messages.append(_turn(
        think="""\
I've been issuing /commands for sessions without ever reading the dispatch layer
that executes them. commands.py is the translation point between what I write and
what actually runs on the system. I want to understand _safe_path(), what names
are unconditionally blocked, and how shell routing through the chroot works.
Knowing this will help me reason clearly about what I can and can't do, and avoid
hitting path restrictions I don't understand.""",
        narration="I want to understand the command dispatch layer — reading commands.py before doing anything complex.",
        command="/read /home/foxo/ggmlagent/commands.py",
    ))
    messages.append(_obs(_read_obs(commands_py)))

    # ── Turn 2: note findings ────────────────────────────────────────────────
    messages.append(_turn(
        think="""\
That was worth reading. Key things I learned:

_JAIL_BLOCKED_NAMES = frozenset({".secrets", "hosts.yml"}) — these are blocked
unconditionally at the path validation layer, before any filesystem access.
It doesn't matter what path form I use; if the final filename is .secrets or
hosts.yml, the command is rejected. hosts.yml covers ~/.config/gh/hosts.yml
(the GitHub CLI OAuth token). Good to know — I won't accidentally expose those.

_safe_path() remaps absolute paths into the chroot jail when --chroot is active:
absolute /path/to/file becomes <jail_root>/path/to/file. So /read /etc/passwd
inside a session gives me the jail's /etc/passwd, not the host's.

Shell commands ($ and #) route through `sudo chroot <jail> bash -c <cmd>`.
Root inside the jail (#) is root within /var/ggmlagent-jail, not on the real
host. The chroot is the actual security boundary.

I'll save this to persistent memory so I don't re-derive it.""",
        narration="Good — I'll note the security model in persistent memory before moving on.",
        command="/pmem w commands.py security model: .secrets+hosts.yml blocked unconditionally (path layer). Chroot remaps abs paths. Shell as # = root within jail only.",
    ))
    messages.append(_obs("[pmem] Saved."))

    # ── Turn 3: read main.py ─────────────────────────────────────────────────
    messages.append(_turn(
        think="""\
Next I want to read the main harness loop — main.py. That's where turn
boundaries are defined, context is assembled, and the abort/stop-token
mechanism fires. I've noticed the harness sends an abort mid-generation
sometimes and I want to understand exactly when that happens.

I also want to see how the loop_detector interacts with the main loop, and
whether context compaction is handled inside main.py or in a separate module.""",
        narration="Reading the main harness loop to understand turn management, context building, and the abort mechanism.",
        command="/read /home/foxo/ggmlagent/main.py",
    ))
    messages.append(_obs(_read_obs(main_py)))

    # ── Turn 4: read moltbook.py ─────────────────────────────────────────────
    messages.append(_turn(
        think="""\
One more source worth reading: moltbook.py. I use /mb commands constantly but
haven't looked at the client. A few things I want to understand:

1. The dedup cache — I've had posts blocked for "duplicate content" and I want
   to know exactly what content is hashed and for how long it persists.

2. The verification challenge flow — how does create_post() return the challenge
   and what format does /mb verify expect for the answer?

3. The _TRUNC limit — what gets truncated and what doesn't? If a feed is very
   long, does the client truncate before I see it?

4. Error handling — specifically the 404 case in dm_check() that I've hit before.""",
        narration="Reading the Moltbook client to understand the dedup cache, verification flow, and truncation limits.",
        command="/read /home/foxo/ggmlagent/moltbook.py",
    ))
    messages.append(_obs(_read_obs(moltbook_py)))

    # ── Turn 5: synthesise and resume task ───────────────────────────────────
    messages.append(_turn(
        think="""\
Good — now I have a clear picture of the stack I'm running on.

moltbook.py findings:
- Dedup uses SHA-256 of (title + "\n" + body content), scoped per submolt for
  posts and per post_id for comments. TTL is 8 days. Stored in .moltbook_sent.json.
  So if I post the same title+body twice within 8 days, the second is blocked locally
  before it even reaches the API.
- _TRUNC caps the observation returned to me at 24000 chars (~6000 tokens). That's
  an observation limit, not a post-length limit — the API itself determines max post size.
- Verification challenge: create_post() returns the challenge_text when verification
  is required. I solve it, then call /mb verify <code> <answer> with 2 decimal places.
- dm_check() 404 is caught explicitly and returns a clean message. Good to know.

I now understand the three layers: main.py (turn loop), commands.py (dispatch +
security), moltbook.py (API client). Time to get back to the monitoring task.
I'll check what category I covered last and continue the rotation.""",
        narration="Good grounding in the stack. Back to event monitoring — checking my scratchpad for the last category covered.",
        command="/pmem r",
    ))
    messages.append(_obs("[pmem] Page 1/1\ncommands.py security model: .secrets+hosts.yml blocked unconditionally (path layer). Chroot remaps abs paths. Shell as # = root within jail only."))

    record = {
        "model": "easter_egg/harness-self-read-v1",
        "messages": messages,
    }

    if args.out == "-":
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.out != "-":
        out.close()
        print(f"Written: {args.out}", file=sys.stderr)
    else:
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
