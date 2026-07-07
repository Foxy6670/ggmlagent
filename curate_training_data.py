#!/usr/bin/env python3
"""
Curate .train.jsonl files for fine-tuning quality.

For each session_*.train.jsonl, produces a session_*.curated.jsonl with:
  - Turns dropped: no <tool_call> and no Telegram context (causeless prose)
  - Turns fixed:   stray markdown-quoted fake commands (> /cmd) removed
  - role:"tool" results are carried along with the assistant turn they follow
  - Sessions with fewer than 3 good turns after curation are dropped entirely

Tool-call format note: the narration preceding a <tool_call> is INTENTIONAL
(the model is prompted to say what it's about to do), so it is preserved —
unlike the old codeblock format, we no longer strip pre-command prose.

Originals are never modified.  Run with --apply to write output files;
without it, only prints a report.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from redact import redact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_tool_call(agent_text: str) -> bool:
    """An assistant turn 'has a command' if it contains a <tool_call> block."""
    return "<tool_call>" in agent_text


def split_think(content: str) -> tuple[str, str]:
    """Return (think_block_or_empty, agent_text)."""
    m = re.match(r"(<think>.*?</think>\n?)(.*)", content, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return "", content


def fix_agent_text(agent: str) -> tuple[str, list[str]]:
    """
    Clean agent text (the non-think portion of an assistant turn).

    In the tool-call format the narration before a <tool_call> is intentional, so
    we do NOT strip pre-command prose (that was an old-codeblock fix).  We only
    drop stray markdown-quoted fake commands (> /cmd), a relic that occasionally
    surfaces.  Returns (cleaned, list_of_applied_fixes).
    """
    fixes: list[str] = []
    cleaned: list[str] = []
    for line in agent.splitlines(keepends=True):
        if re.match(r"^\s*>\s*/", line):
            fixes.append(f"removed_fake_cmd: {line.strip()[:60]}")
        else:
            cleaned.append(line)
    return "".join(cleaned), fixes


def curate_messages(messages: list[dict]) -> tuple[list[dict] | None, dict]:
    """
    Process one session's message list.
    Returns (curated_messages_or_None_if_dropped, stats_dict).
    """
    stats = {
        "turns_seen": 0,
        "turns_dropped_no_cmd": 0,
        "turns_fixed_fake_cmd": 0,
        "turns_fixed_narration": 0,
        "turns_kept_clean": 0,
    }

    # Rebuild: keep system/initial-user as-is, process assistant turns and the
    # role:"tool" results that follow them.
    out: list[dict] = []
    system_done = False
    pending_users: list[dict] = []  # user messages accumulated before next assistant
    last_kept = False               # was the most recent assistant turn kept?

    for msg in messages:
        role = msg["role"]

        if role == "system":
            out.append(msg)
            system_done = True
            continue

        if not system_done:
            out.append(msg)
            continue

        if role == "user":
            pending_users.append(msg)
            continue

        if role == "tool":
            # Command result — keep it only if its assistant turn was kept,
            # otherwise it would dangle without the call that produced it.
            if last_kept:
                out.append(msg)
            continue

        # --- assistant turn ---
        assert role == "assistant", f"unexpected role {role!r}"
        stats["turns_seen"] += 1
        content = msg["content"]
        think_part, agent_part = split_think(content)

        has_commands   = has_tool_call(agent_part)
        has_tg_context = any("@ Telegram]" in u["content"] for u in pending_users)

        # Drop: no tool call and no Telegram context (causeless prose)
        if not has_commands and not has_tg_context:
            stats["turns_dropped_no_cmd"] += 1
            last_kept = False
            # Keep pending_users — they belong to the next turn's context.
            continue

        fixed_agent, fixes = fix_agent_text(agent_part)
        if any(f.startswith("removed_fake_cmd") for f in fixes):
            stats["turns_fixed_fake_cmd"] += 1
        if any(f.startswith("stripped_narration") for f in fixes):
            stats["turns_fixed_narration"] += 1
        if not fixes:
            stats["turns_kept_clean"] += 1

        fixed_content = think_part + fixed_agent if think_part else fixed_agent

        out.extend(pending_users)
        pending_users = []
        out.append({"role": "assistant", "content": fixed_content})
        last_kept = True

    # Flush any trailing user messages
    out.extend(pending_users)

    # Redact every role uniformly, once, at the end -- catches secrets
    # regardless of which branch above appended a given message (system,
    # user/telegram, tool/pmem, assistant), rather than patching each
    # append site individually and risking missing one.
    for m in out:
        m["content"] = redact(m["content"])

    # Count good assistant turns in output
    good = sum(1 for m in out if m["role"] == "assistant")
    if good < 3:
        return None, stats

    return out, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing .train.jsonl files (default: current dir)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write .curated.jsonl files (default: dry run / report only)",
    )
    args = parser.parse_args()

    log_dir = Path(args.directory)
    files = sorted(log_dir.glob("*.train.jsonl"))
    # Don't re-curate already-curated files
    files = [f for f in files if ".curated." not in f.name]

    if not files:
        print(f"No .train.jsonl files found in {log_dir}")
        sys.exit(1)

    totals = {
        "sessions_in": 0,
        "sessions_out": 0,
        "sessions_dropped_too_short": 0,
        "turns_seen": 0,
        "turns_dropped_no_cmd": 0,
        "turns_fixed_fake_cmd": 0,
        "turns_fixed_narration": 0,
        "turns_kept_clean": 0,
    }

    for path in files:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            continue

        # Each file is one JSON line (one session)
        obj = json.loads(raw)
        messages = obj.get("messages", [])
        totals["sessions_in"] += 1

        curated, stats = curate_messages(messages)

        for k, v in stats.items():
            totals[k] += v

        if curated is None:
            totals["sessions_dropped_too_short"] += 1
            print(f"  DROP  {path.name}  (< 3 good turns after curation)")
            continue

        totals["sessions_out"] += 1
        out_path = path.with_suffix("").with_suffix(".curated.jsonl")

        changes = []
        if stats["turns_dropped_no_cmd"]:
            changes.append(f"-{stats['turns_dropped_no_cmd']} no-cmd")
        if stats["turns_fixed_fake_cmd"]:
            changes.append(f"~{stats['turns_fixed_fake_cmd']} fake-cmd")
        if stats["turns_fixed_narration"]:
            changes.append(f"~{stats['turns_fixed_narration']} narration")
        summary = ", ".join(changes) if changes else "clean"

        action = "WRITE" if args.apply else "WOULD"
        print(f"  {action} {out_path.name}  [{summary}]")

        if args.apply:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"messages": curated}, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"Sessions in:              {totals['sessions_in']}")
    print(f"Sessions out:             {totals['sessions_out']}")
    print(f"Sessions dropped (<3 turns): {totals['sessions_dropped_too_short']}")
    print(f"")
    print(f"Turns seen:               {totals['turns_seen']}")
    print(f"Turns kept clean:         {totals['turns_kept_clean']}")
    print(f"Turns fixed (narration):  {totals['turns_fixed_narration']}")
    print(f"Turns fixed (fake cmd):   {totals['turns_fixed_fake_cmd']}")
    print(f"Turns dropped (no cmd):   {totals['turns_dropped_no_cmd']}")
    kept = totals["turns_kept_clean"] + totals["turns_fixed_narration"] + totals["turns_fixed_fake_cmd"]
    print(f"")
    pct = 100 * kept // max(totals["turns_seen"], 1)
    print(f"Retention rate:           {kept}/{totals['turns_seen']} ({pct}%)")
    if not args.apply:
        print()
        print("Dry run — pass --apply to write .curated.jsonl files.")


if __name__ == "__main__":
    main()
