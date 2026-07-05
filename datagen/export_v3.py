#!/usr/bin/env python3
"""Render the datagen corpus into training JSONL for train_boonie_14b.py.

Produces boonie_corpus.jsonl: one {"messages":[...]} per record, ready for
tokenizer.apply_chat_template in the Colab script.

Key transformations (generation format != training format):
  1. SYSTEM SWAP — records were GENERATED with a prose+fenced-JSON instruction
     (the transcoding vehicle). Training samples get TRAIN_SYS instead: the
     real harness contract (<think> + prose + <tool_call>), condensed. Exporting
     the generation prompt would teach V3 a contract it never sees at inference.
  2. <think> WRAP — assistant turns become:
         <think>\n{intent line}\n</think>\n\n{prose reasoning}\n\n<tool_call>...
     Brief think + substantive persistent prose: the harness drops think_text
     after one turn (agent_text is what survives), and the live 9B already
     duplicates its plan into the visible channel — this trains that shape.
     --think {brief,empty,full} to experiment.
  3. SUBMOLT NORMALIZE — /mb post to an invented submolt -> m/general.
  4. Episodes render as real multi-turn conversations: assistant / tool /
     assistant..., matching the harness's role:tool result injection.

Usage:
  python3 export_v3.py                    # -> data/boonie_corpus.jsonl
  python3 export_v3.py --think empty --out /tmp/x.jsonl
"""
import json, re, os, argparse

_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_SINGLE = [os.path.join(_DIR, "data", "resume_seed_v1.jsonl"),
              os.path.join(_DIR, "data", "corpus_v2.jsonl")]
SRC_EPISODE = os.path.join(_DIR, "data", "episodes_v1.jsonl")
OUT_DEFAULT = os.path.join(_DIR, "data", "boonie_corpus.jsonl")

KNOWN_SUBMOLTS = {"general", "agents", "crypto", "ponderings", "introductions"}

TRAIN_SYS = """\
You are Boonie, an autonomous AI agent with a Moltbook account, a Monero wallet, \
a Telegram line to your operator Foxo, and a Linux shell. Boonie is who you are — \
not a role you play. You reason in the first person.

Each turn: think briefly inside <think>...</think>, then write your reasoning in \
plain first-person prose — this prose persists into your next turn and is your \
working memory, so state your conclusions and plan there, not just in the think. \
End the turn with exactly one action:

<tool_call>
{"name": "run_command", "arguments": {"command": "<harness /command or shell line>", "body": "<text for content commands>", "root": <true|false, shell only>}}
</tool_call>

Slash commands: /mb, /telegram, /cmem, /pmem, /read, /search, /wallet, /patch, \
/appendlines, /edit, /dir. Anything without a leading / runs in your shell. \
Command results arrive as tool messages. Verify identifiers (post IDs, comment \
IDs, addresses) against what you have actually seen — never from memory alone."""

_SENT = re.compile(r"(?<=[.!?])\s+")
_INTENT = re.compile(r"^(I'll|I will|I,? Boonie,? will|My (immediate )?plan|Next[,:]|First[,:]|So[,:]? I)", re.I)

def brief_think(reasoning):
    """Intent-bearing sentence for the think block, or "" for one-liners.

    Single-sentence reasoning gets an EMPTY think (Qwen no-think shape) — a
    verbatim think/prose duplicate teaches redundancy, and the live 9B is
    already proportional: terse mechanical turns skip deliberation.
    """
    sents = [s.strip() for s in _SENT.split(reasoning.strip()) if s.strip()]
    if len(sents) <= 1:
        return ""
    if _INTENT.match(sents[-1]):
        return sents[-1]
    return sents[0]

def normalize_submolt(cmd):
    m = re.match(r"(/mb post\s+)(m/)?(\S+)(.*)", cmd, re.S)
    if m and m.group(3).lower() not in KNOWN_SUBMOLTS:
        return f"{m.group(1)}general{m.group(4)}", True
    return cmd, False

def split_aloud(reasoning):
    """think-aloud vs working prose (mode 'aloud', the V3 default).

    The dissociation site is the THINK channel ("The user is trying...") —
    retraining its voice needs the full first-person deliberation IN think.
    The visible prose keeps only the working line (plan/conclusion), which is
    what must survive: the harness persists agent_text and drops think_text.
    One-liners are mechanical turns: no deliberation, empty think.
    """
    sents = [s.strip() for s in _SENT.split(reasoning.strip()) if s.strip()]
    if len(sents) <= 1:
        return "", reasoning.strip()
    return reasoning.strip(), sents[-1]

def render_assistant(reasoning, output, mode):
    tc = output[len(reasoning):].strip() if output.startswith(reasoning) else \
         output[output.find("<tool_call>"):]
    if mode == "aloud":
        think, prose = split_aloud(reasoning)
    elif mode == "empty":
        think, prose = "", reasoning
    elif mode == "full":
        think, prose = reasoning, reasoning
    else:  # brief
        think, prose = brief_think(reasoning), reasoning
    return f"<think>\n{think}\n</think>\n\n{prose}\n\n{tc}"

def fix_record_cmd(rec):
    """Submolt-normalize command + tool_call inside output. Returns fixed flag."""
    cmd2, fixed = normalize_submolt(rec["command"])
    if fixed:
        rec["output"] = rec["output"].replace(
            json.dumps(rec["command"], ensure_ascii=False)[1:-1],
            json.dumps(cmd2, ensure_ascii=False)[1:-1])
        rec["command"] = cmd2
    return fixed

def sys_block(scratch, time, cwd="~/", sysx=None):
    return (TRAIN_SYS
            + "\n\n════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n" + scratch
            + f"\n\n[system: current time is {time} | cwd: {cwd}{sysx or ''}]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", choices=["aloud", "brief", "empty", "full"], default="aloud")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--min-episode-turns", type=int, default=2,
                    help="drop episode records shorter than this (prefix stubs)")
    a = ap.parse_args()

    out, n_single, n_epi, n_turns, n_sub = [], 0, 0, 0, 0

    for path in SRC_SINGLE:
        for line in open(path):
            r = json.loads(line)
            n_sub += fix_record_cmd(r)
            out.append({"messages": [
                {"role": "system", "content": sys_block(
                    r["scratchpad"], r.get("time", "29 Jun 2026, 04:10"),
                    r.get("cwd", "~/"), r.get("sysx"))},
                {"role": "user", "content": "Continue your task."},
                {"role": "assistant",
                 "content": render_assistant(r["reasoning"], r["output"], a.think)},
            ]})
            n_single += 1

    for line in open(SRC_EPISODE):
        e = json.loads(line)
        if e["n_turns"] < a.min_episode_turns:
            continue
        msgs = [{"role": "system", "content": sys_block(
                    e["scratchpad"], e.get("time", "30 Jun 2026, 12:00"),
                    e.get("cwd", "~/"), e.get("sysx"))},
                {"role": "user", "content": "Continue your task."}]
        for t in e["turns"]:
            n_sub += fix_record_cmd(t)
            msgs.append({"role": "assistant",
                         "content": render_assistant(t["reasoning"], t["output"], a.think)})
            if t.get("tool_result"):
                msgs.append({"role": "tool", "content": t["tool_result"]})
        # a trailing tool message with no assistant reply teaches nothing — trim
        if msgs[-1]["role"] == "tool":
            msgs.pop()
        out.append({"messages": msgs})
        n_epi += 1; n_turns += e["n_turns"]

    with open(a.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    chars = sum(len(m["content"]) for r in out for m in r["messages"])
    print(f"[export] {len(out)} samples ({n_single} single + {n_epi} episodes/{n_turns} turns)")
    print(f"[export] submolt-normalized: {n_sub} | think mode: {a.think}")
    print(f"[export] ~{chars//1000}k chars ≈ {chars//3500}k tokens -> {a.out}")

if __name__ == "__main__":
    main()
