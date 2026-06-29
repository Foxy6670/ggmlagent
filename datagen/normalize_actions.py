#!/usr/bin/env python3
"""Deterministic harness-contract normalizer for generated resume turns.

Fixes the command/body-grammar slips the model makes (we own the contract):
  1. dropped-slash restoration on known command stems (+ clears bogus root, since
     ANY root value routes to the shell per agent._effective_command)
  2. body-fold for inline-only slash commands (body is otherwise ignored/lost)
  3. /mb post title extraction when the title landed in the body
Re-transcodes to the Qwen3 <tool_call> envelope and re-validates.

Reads frontier_resume_corpus_v3.jsonl, writes frontier_resume_corpus_clean.jsonl.
No API calls — reasoning is already generated and good; only actions are repaired.
"""
import json, re, sys

# Standalone reprocessing paths (importers just use the functions below):
#   python3 normalize_actions.py <src.jsonl> <dst.jsonl>
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/resume_raw.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/resume_seed_v1.jsonl"

# Real harness slash-command stems (from commands.py dispatch).
STEMS = {"append", "appendlines", "back", "cm", "cmem", "del", "dellines", "dir",
         "edit", "goto", "mb", "next", "patch", "pgdown", "pgup", "pm", "pmem",
         "pmew", "read", "search", "telegram", "tmp", "wallet"}

def is_body_ok(cmd):
    """Commands whose body the harness actually consumes (dispatch_block)."""
    if cmd.startswith(("/mb post", "/mb comment", "/mb reply", "/telegram",
                       "/patch", "/appendlines", "/edit")):
        return True
    if cmd.startswith("/cmem w") or cmd.startswith("/cm w"):
        return True
    return False

def normalize_action(cmd, body, root):
    notes = []
    cmd = (cmd or "").strip()
    body = body if (body and str(body).strip()) else None

    # 1. dropped-slash restoration (first token is a known stem, no slash)
    if cmd and not cmd.startswith("/") and not cmd.startswith(("$", "#")):
        first = cmd.split()[0] if cmd.split() else ""
        if first in STEMS:
            cmd = "/" + cmd
            root = None                       # it's a slash command, not shell
            notes.append("slash-restored")

    # 2. body-fold for inline-only slash commands (body would be ignored/lost)
    if cmd.startswith("/") and not is_body_ok(cmd) and body:
        if cmd.startswith("/search"):
            rest = cmd[len("/search"):].strip()
            cmd = f'/search "{body.strip()}"' if not rest else f"{cmd} {body.strip()}"
        else:
            cmd = f"{cmd.rstrip()} {body.strip()}"
        body = None
        notes.append("body-folded")

    # 3. /mb post title extraction (title landed in body)
    if cmd.startswith("/mb post"):
        toks = cmd.split()
        if len(toks) < 4 and body:            # /mb post <submolt> with no title
            lines = [l for l in body.splitlines() if l.strip()]
            if lines:
                title = lines[0].strip()
                cmd = f"{cmd.rstrip()} {title}"
                rest = body.split(title, 1)[-1].lstrip("\n ")
                body = rest.strip() or None
                notes.append("title-extracted")

    # slash commands must never carry root (it would force shell routing)
    if cmd.startswith("/"):
        root = None
    return cmd, body, root, notes

def transcode(cmd, body, root):
    args = {"command": cmd}
    if body and str(body).strip():
        args["body"] = body
    if isinstance(root, bool):
        args["root"] = root
    tc = json.dumps({"name": "run_command", "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{tc}\n</tool_call>"

def valid(cmd, body):
    if not cmd:
        return False, "empty"
    if cmd.startswith("/command"):
        return False, "literal-/command"
    if not cmd.startswith(("/", "$", "#")) and cmd.split()[0] in STEMS:
        return False, "missing-slash"
    if cmd.startswith(("/telegram", "/mb post", "/mb comment")) and not (body and body.strip()):
        return False, "body-missing"
    if cmd.startswith("/mb post") and len(cmd.split()) < 4:
        return False, "no-title"
    return True, "ok"

def main():
    changed = 0; still_bad = []
    recs = [json.loads(l) for l in open(SRC)]
    with open(DST, "w") as fout:
        for r in recs:
            c0, b0, r0 = r["command"], r.get("body"), r.get("root")
            c1, b1, r1, notes = normalize_action(c0, b0, r0)
            ok, why = valid(c1, b1)
            if notes:
                changed += 1
                print(f"[{r['scenario']} t{r['temp']}] {notes}")
                print(f"   before: cmd={c0!r} root={r0}")
                print(f"   after : cmd={c1!r} root={r1}")
            if not ok:
                still_bad.append((r["scenario"], r["temp"], why, c1))
            r["command"], r["body"], r["root"] = c1, b1, r1
            r["tool_call"] = transcode(c1, b1, r1)
            r["output"] = f"{r['reasoning']}\n\n{r['tool_call']}"
            r["valid"] = ok
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(recs)
    good = n - len(still_bad)
    print(f"\n=== normalized {changed} records | {good}/{n} contract-valid ===")
    if still_bad:
        print("STILL INVALID:")
        for s, t, why, c in still_bad:
            print(f"  [{s} t{t}] {why}: {c!r}")
    print(f"saved -> {DST}")

if __name__ == "__main__":
    main()
