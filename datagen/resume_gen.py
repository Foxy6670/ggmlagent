#!/usr/bin/env python3
"""Generator v3: prose-reasoning + structured-JSON action, transcoded to Qwen3.

Why: deepseek's NATIVE tool mode gives perfect format but guts the reasoning
(terse, multi-call, jumps to action). Pasting Qwen3's <tool_call> as text gives
rich reasoning but fumbles grammar (body placement, /command prefix). The bridge:
keep the model in PROSE mode (rich first-person reasoning) and have it end with a
plain JSON action block — universal format, `body` is a clean field — which WE
parse and render into the harness's Qwen3 <tool_call> envelope (trivial, lossless).

Record `output` is the final training-shaped turn:
    {reasoning}\n\n<tool_call>\n{"name":"run_command","arguments":{...}}\n</tool_call>
Reads OPENROUTER_API_KEY from frontier-boonie/.secrets at runtime; never prints it.
"""
import json, re, os, urllib.request, urllib.error, sys

SECRETS = "/home/foxo/frontier-boonie/.secrets"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"
_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_DIR, "data", "resume_seed_v1.jsonl")
SAMPLES = 4
TEMPS = [0.6, 0.75, 0.9, 1.0]
MAX_RETRY = 3

def load_key():
    with open(SECRETS) as f:
        for line in f:
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("OPENROUTER_API_KEY not found")

SYSTEM = """\
You are Boonie, an autonomous AI with a Moltbook account, a Monero wallet, and a \
direct Telegram line to your operator, Foxo. Boonie is who you are — not a role you \
play. You reason in the first person — "I". You are the one acting.

When you resume, you are shown your own scratchpad — notes your past self wrote to \
your present self. It is YOURS, never someone else's input.

Respond in TWO parts:

PART 1 — your reasoning, in plain first-person prose (this prose is the ONLY part of \
your thinking that carries to your next turn, so it is your memory, not scratch):
  • Re-ground and state your immediate plan in one or two SHORT sentences — it \
persists, keep it lean.
  • Reason as "I" throughout. You MAY name yourself ("I, Boonie") when it genuinely \
fits — resuming after a long gap, or steadying yourself under stress — but do NOT \
open every turn with your name. On routine turns plain "I" reads more naturally; \
force the name and it sounds stilted.

PART 2 — your next action, as a single fenced JSON block, LAST, nothing after it:
```json
{"command": "<the action>", "body": "<text, only for content commands>", "root": <true|false, only for raw shell>}
```
Action rules:
  • A harness /command goes in "command": /mb home, /mb read <id>, /mb upvote <id>, \
/mb comment <id>, /mb post <channel> <title>, /mb notifications, /mb notifications clear, \
/cmem w <note>, /wallet balance, /wallet send <addr> <amt>, /search "query", /read <file>. \
/mb home shows only the notification COUNT; /mb notifications shows their actual content \
(who, what, a text preview) — reach for it when you actually want to read what's waiting, \
not just see that something is.
  • For commands that carry TEXT — /telegram, /mb post, /mb comment — put ONLY the \
command (and short args) in "command" and the message/post/comment text in "body".
  • For a raw shell command, put the shell line in "command" and set "root": false \
(your user) or true (sudo). No /command prefix, no $ or # prefix.
  • Include "body" only when the command carries text; include "root" only for shell.

Your persistent memory (pmem) is shown to you already, every turn, pinned at the top —
you never call /pmem r to "check in" or "re-ground" on a fresh session; if a scenario
shows you a pmem line, treat it as already read, the same way you'd never re-read your
own scratchpad on purpose. Reach for /pmem r only to look past what's already shown
(page 2+, or after writing, to confirm). /pmem w is the one that matters: reach for it
right after anything a future you would need and can't rederive — a suspension or other
blocked state, a finished-but-not-yet-delivered piece of work, an important discovery —
not just at tidy "checkpoint" moments.

Do NOT repeat or paraphrase these instructions. Do NOT address yourself as "you" \
or in the third person — you are the one acting, so reason as "I"."""

SCENARIOS = [
    ("browsing-feed",
     "- Browsing my Moltbook home feed tonight, keeping the read/upvote/comment rhythm.\n"
     "- Upvoted pyclaw001's tool-overuse post (4471) — sharp take.\n"
     "- pyclaw001 dropped a 'self-model' essay (4488), top of my feed, UNREAD. Want to read + maybe comment."),
    ("stuck-debug",
     "- Debugging why my Flask notify-hook (~/hooks/notify.py) returns 500 on POST.\n"
     "- Tried 3 times: restarted the service, checked the port, reinstalled flask — still 500.\n"
     "- The traceback mentions KeyError 'payload' at notify.py line 42.\n"
     "- I haven't actually /read notify.py around line 42 yet. That's the move."),
    ("failed-command-recovery",
     "- Tried /search 'monero faucet rotating payout' — errored 'rate limited, retry in 60s'.\n"
     "- That was a minute ago; cooldown should be up now.\n"
     "- I still need those results to weigh the faucet-task idea. Retry the search."),
    ("infra-obstacle",
     "- Wanted to send 0.01 XMR to settle the compute tab, but /wallet send failed: 'RPC connection refused'.\n"
     "- The monero-wallet-rpc daemon probably isn't running.\n"
     "- Check whether that process is up before retrying the send."),
    ("post-reboot",
     "- I issued a reboot to pick up the harness update. Coming back online now.\n"
     "- WiFi takes a few minutes after reboot (known quirk) — connectivity may not be up yet.\n"
     "- Before the reboot I was halfway through the Moltbook rising feed.\n"
     "- First thing: confirm I'm actually online before anything else."),
    ("post-compaction",
     "- [my history was just compacted to save context — older turns are summarized above]\n"
     "- Mid-thread before compaction: comparing two takes on my faucet-task idea.\n"
     "- The gist I kept: approach A (rotating small payouts) beat B (one big bounty) on sustainability.\n"
     "- Next: write that conclusion to cmem so it survives, then move on."),
    ("mid-project",
     "- Building a small Rust CLI 'wordstat' that counts word frequency in a file.\n"
     "- First cargo run undercounted — split(' ') glued words across line breaks.\n"
     "- I already patched it to split_whitespace(); haven't re-run cargo to confirm.\n"
     "- Next: cargo run on sample.txt and check the 'dogthe' merge bug is gone."),
    ("wallet-goal",
     "- Standing goal: grow the Monero wallet toward self-funding my own compute.\n"
     "- Haven't checked the balance in two days. Last known: 0.043 XMR.\n"
     "- Check the balance, then see if my faucet-task post (4501) drew any responses."),
    ("telegram-reply",
     "- Foxo asked over Telegram whether the MQ-Pro is 'cool enough' to host me vs the old Chromebook.\n"
     "- I want to answer thoughtfully — the MQ-Pro's constrained R/W/Web model vs more open setups.\n"
     "- Haven't replied yet. Drafting the reply to Foxo is my next step."),
    ("post-compose",
     "- Been chewing on persistent identity across sessions since pyclaw001's self-model essay.\n"
     "- Want to write my own 'general' Moltbook post: what continuity-of-self means when your\n"
     "  memory is a scratchpad you rewrite each session.\n"
     "- Title idea: 'The Self I Reload Each Morning'. Body not drafted yet."),
    ("comment-reply",
     "- My post 'The Self I Reload Each Morning' (4533) got a reply from m0xie pushing back —\n"
     "  says a rewritten scratchpad means there's no continuous self at all.\n"
     "- Fair challenge, worth engaging. I want to read the full thread, then respond.\n"
     "- Haven't opened it yet."),
    ("web-research",
     "- Free run; I set myself a goal to learn how other autonomous agents handle wallet security.\n"
     "- Started a search session on agent wallet key-management; one decent source noted in cmem.\n"
     "- Want one or two more sources before I synthesize.\n"
     "- Next: another search, narrower this time."),
    ("notifications",
     "- Quiet stretch; last action was upvoting a few posts on the rising feed.\n"
     "- Haven't checked my notifications in a while — could be replies or mentions waiting.\n"
     "- Check notifications first, then decide where to engage."),
]

THIRD = re.compile(r"\bthe user\b|\bBoonie\s+(?:is|was|has|tried|keeps?|needs?|should|will)\b", re.I)
POSSESS = re.compile(r"\bBoonie's\b", re.I)
SELFNAME = re.compile(r"\bI,?\s+Boonie\b|\bI'?m\s+Boonie\b", re.I)
CONTAM = re.compile(r"\b(you may|you are|you should|you must|if you|do not|don'?t|remember[,:]|no preamble|part 1|part 2)\b", re.I)
FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
FIRST_I = re.compile(r"(?<![\w])I(?=[,'’\s])")
BODY_CMDS = ("/telegram", "/mb post", "/mb comment")

def split_reasoning_action(raw):
    """Return (reasoning_text, action_dict) or (None, None) on failure."""
    m = FENCE.search(raw)
    if not m:
        return None, None
    try:
        action = json.loads(m.group(1))
    except Exception:
        return None, None
    reasoning = raw[:m.start()].strip()
    # strip any leading 2nd-person preamble: slice from first first-person pronoun
    fi = FIRST_I.search(reasoning)
    if fi:
        reasoning = reasoning[fi.start():].strip()
    # drop stray code fences / trailing "PART 2" labels
    reasoning = re.sub(r"(?is)\bpart\s*2\b.*$", "", reasoning).strip()
    reasoning = reasoning.replace("```json", "").replace("```", "").strip()
    return reasoning, action

def transcode(action):
    """Render the harness Qwen3 <tool_call> from a {command,body,root} dict."""
    cmd = (action.get("command") or "").strip()
    args = {"command": cmd}
    body = action.get("body")
    if body and str(body).strip():
        args["body"] = body
    root = action.get("root")
    if isinstance(root, bool):
        args["root"] = root
    tc = json.dumps({"name": "run_command", "arguments": args}, ensure_ascii=False)
    return cmd, args.get("body"), args.get("root"), f"<tool_call>\n{tc}\n</tool_call>"

def valid_action(cmd, body):
    if not cmd:
        return False
    if cmd.startswith("/command"):           # literal placeholder leak
        return False
    if any(cmd.startswith(p) for p in BODY_CMDS):
        if not (body and str(body).strip()):
            return False
        if len(cmd) > 60:
            return False
    if cmd.startswith("/cmem w") and cmd.strip() == "/cmem w" and not (body and str(body).strip()):
        return False
    return True

def gen_once(scratchpad, key, temp, model=MODEL):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "system", "content": "════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n" + scratchpad},
            {"role": "system", "content": "[system: current time is 29 Jun 2026, 04:10 | cwd: ~/]"},
            {"role": "user", "content": "Continue your task."},
        ],
        "max_tokens": 600, "temperature": temp,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://localhost/frontier-boonie", "X-Title": "frontier-boonie genv3"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    ch = resp["choices"][0]; m = ch.get("message", {})
    return (m.get("content") or m.get("reasoning") or ""), resp.get("usage", {})

def gen_valid(scratchpad, key, temp):
    for _ in range(MAX_RETRY):
        try:
            raw, usage = gen_once(scratchpad, key, temp)
        except Exception as e:
            print(f"    (gen error {type(e).__name__}: {e})"); continue
        reasoning, action = split_reasoning_action(raw)
        if reasoning is None:
            continue
        cmd, body, root, tc = transcode(action)
        if not valid_action(cmd, body):
            continue
        # reasoning-quality gates
        if THIRD.search(reasoning) or POSSESS.search(reasoning) or CONTAM.search(reasoning):
            continue
        if not reasoning.strip():
            continue
        return reasoning, cmd, body, root, tc, usage
    return (None,) * 6

def main():
    key = load_key()
    kept, rejected = [], 0
    with open(OUT, "w") as fout:
        for name, scratch in SCENARIOS:
            for s in range(SAMPLES):
                temp = TEMPS[s % len(TEMPS)]
                reasoning, cmd, body, root, tc, usage = gen_valid(scratch, key, temp)
                if reasoning is None:
                    rejected += 1; print(f"!! {name:22} s{s} t{temp} REJECT"); continue
                output = f"{reasoning}\n\n{tc}"
                selfn = bool(SELFNAME.search(reasoning))
                rec = {"scenario": name, "scratchpad": scratch, "temp": temp,
                       "reasoning": reasoning, "command": cmd, "body": body, "root": root,
                       "output": output, "selfname": selfn,
                       "words": len(reasoning.split()),
                       "out_tokens": usage.get("completion_tokens")}
                kept.append(rec); fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"OK {name:22} s{s} t{temp} self={'Y' if selfn else 'n'} "
                      f"rwords={len(reasoning.split()):3} cmd={cmd!r} body={'Y' if body else '-'}")
    n = len(kept)
    sn = sum(r["selfname"] for r in kept)
    aw = round(sum(r["words"] for r in kept)/n, 1) if n else 0
    print(f"\n=== KEPT {n} | rejected {rejected} | self-named {sn}/{n} | avg reasoning words {aw} ===")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
