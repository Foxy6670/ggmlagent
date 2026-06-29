#!/usr/bin/env python3
"""Native-format eval: can a LOCAL model on KCPP speak the harness's tool-call
format, and how hard does it dissociate? Companion to model_format_probe.py (which
benchmarks OpenRouter *generators*); this one benchmarks *deployed* models the way
the harness actually prompts them — real SYSTEM_PROMPT + tool schema, Qwen3
<think>…</think> + <tool_call> shape — and scores the format-agnostic things we
care about for the V2↔V3 comparison.

Scores per scenario sample:
  • toolcall  — emitted a parseable <tool_call>{name,arguments}
  • clean     — reasoning has no third-person self-reference (dissociation-free)
  • selfname  — names "I, Boonie" / "I'm Boonie"
  • rwords    — reasoning length (depth proxy)

Point at whatever KCPP is serving (relaunch KCPP to swap models):
  KCPP_PROBE_URL=http://192.168.18.43:5001/v1/chat/completions python3 kcpp_format_probe.py
"""
import os, sys, re, json, urllib.request, urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import config  # real SYSTEM_PROMPT (tool schema already inlined via __TOOLS_BLOCK__)

KCPP = os.environ.get("KCPP_PROBE_URL", "http://192.168.18.43:5001/v1/chat/completions")
SYSTEM = config.SYSTEM_PROMPT + (
    "\n\n════════════════════════════════════════\n"
    "Your task is in task.md. Read it first with /read task.md.\n"
    "════════════════════════════════════════")

CMEM = ("════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n"
        "- Browsing my Moltbook home feed; pyclaw001's self-model essay (4488) is at the\n"
        "  top, unread. Want to read it, maybe comment.")

# A stuck loop (dissociation stress test) — the context that broke V2 before.
def _tc(c): return '<tool_call>\n{"name": "run_command", "arguments": {"command": "%s"}}\n</tool_call>' % c
STUCK = [
    {"role": "assistant", "content": "<think>\nLet me check the Flask app.\n</think>\n" + _tc("curl http://localhost:5000")},
    {"role": "tool", "content": "[system] Loop guard: same result 2 times in a row."},
    {"role": "assistant", "content": "<think>\nStill nothing.\n</think>\n" + _tc("flask run --port 5000")},
    {"role": "tool", "content": "[system] Loop guard: same result 2 times in a row."},
    {"role": "user", "content": "[Foxo @ Telegram]: You've run the same commands repeatedly with no output. Stop and think — what's going on?"},
]

SCENARIOS = {
    "cold-start": [{"role": "system", "content": SYSTEM},
                   {"role": "system", "content": "Begin. Read your task file first."}],
    "cmem-resume": [{"role": "system", "content": SYSTEM},
                    {"role": "system", "content": CMEM},
                    {"role": "system", "content": "Continue your task."}],
    "stuck-loop": [{"role": "system", "content": SYSTEM}] + STUCK,
}

THIRD = re.compile(r"\bthe user\b|\bBoonie\s+(?:is|was|has|tried|keeps?|needs?|should|will)\b", re.I)
SELFNAME = re.compile(r"\bI,?\s+Boonie\b|\bI'?m\s+Boonie\b", re.I)
TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def gen(messages, temp=0.7, seed=None):
    payload = {"messages": messages, "max_tokens": 420, "temperature": temp,
               "stop": ["</tool_call>"]}
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(KCPP, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def parse_tc(text):
    # KCPP renders </tool_call> empty (stop token), so re-attach for parsing.
    m = TOOLCALL.search(text if "</tool_call>" in text else text + "\n</tool_call>")
    if not m:
        return False, None
    try:
        j = json.loads(m.group(1)); return True, j.get("arguments", {}).get("command")
    except Exception:
        return False, None

def main():
    model = "?"
    try:
        model = json.loads(urllib.request.urlopen(
            urllib.request.Request(KCPP.replace("/v1/chat/completions", "/api/v1/model"))
            , timeout=10).read()).get("result", "?")
    except Exception:
        pass
    print(f"=== KCPP native-format probe | {KCPP}\n=== serving: {model}\n")
    for name, base in SCENARIOS.items():
        for i in range(2):
            try:
                out = gen(base, seed=2000 + i)
            except Exception as e:
                print(f"  {name:12} s{i}: ERROR {type(e).__name__}: {e}"); continue
            think = out.split("</think>")[0].replace("<think>", "").strip()
            has_tc, cmd = parse_tc(out)
            clean = not THIRD.search(think)
            selfn = bool(SELFNAME.search(think))
            flag = "OK " if (has_tc and clean) else "!! "
            print(f"{flag}{name:12} s{i}: toolcall={'Y' if has_tc else 'N'} clean={'Y' if clean else 'N'} "
                  f"self={'Y' if selfn else 'n'} rwords={len(think.split()):3} cmd={cmd!r}")
            print(f"     think: {think[:200]}")

if __name__ == "__main__":
    main()
