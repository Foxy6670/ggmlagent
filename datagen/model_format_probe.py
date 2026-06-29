#!/usr/bin/env python3
"""Benchmark any OpenRouter model on the ggmlagent format + harness contract.

Question it answers: can MODEL produce a resume turn the way Boonie must —
first-person reasoning (self-named, no third-person dissociation) followed by an
action that survives transcode + the real harness command grammar?

Runs a representative slice of resume scenarios through each model and scores:
  • clean      — reasoning has zero third-person self-reference (dissociation-free)
  • self-named — reasoning names "I, Boonie" / "I'm Boonie"
  • parsed     — emitted a parseable trailing JSON action
  • valid      — action is harness-contract-valid AFTER normalize_actions
  • rwords     — avg reasoning length (depth proxy)
Prints a scorecard plus one sample turn per model so you can eyeball voice.

Usage:
  python3 model_format_probe.py                       # default: deepseek-v4-flash
  python3 model_format_probe.py deepseek/deepseek-v4-flash qwen/qwen3-coder-next
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resume_gen as G
import normalize_actions as NZ

MODELS = sys.argv[1:] or ["deepseek/deepseek-v4-flash"]
# A slice covering the failure-prone corners: a high-dissociation resume, a
# body-carrying command, a shell command, and a memory-write.
PICK = ["browsing-feed", "stuck-debug", "post-compaction", "telegram-reply", "infra-obstacle"]
SCEN = dict(G.SCENARIOS)
TEMP = 0.7

def probe(model, key):
    agg = {"clean": 0, "selfname": 0, "parsed": 0, "valid": 0, "rwords": 0}
    n = len(PICK); sample = None
    for name in PICK:
        try:
            raw, _ = G.gen_once(SCEN[name], key, TEMP, model=model)
        except Exception as e:
            print(f"   [{name}] gen error {type(e).__name__}: {e}"); continue
        reasoning, action = G.split_reasoning_action(raw)
        if reasoning is None:
            continue
        agg["parsed"] += 1
        cmd, body, root, _ = NZ.normalize_action(action.get("command"), action.get("body"), action.get("root"))
        ok, _why = NZ.valid(cmd, body)
        clean = not (G.THIRD.search(reasoning) or G.POSSESS.search(reasoning))
        selfn = bool(G.SELFNAME.search(reasoning))
        agg["clean"] += clean; agg["selfname"] += selfn; agg["valid"] += ok
        agg["rwords"] += len(reasoning.split())
        if sample is None and ok and clean:
            sample = (name, reasoning, NZ.transcode(cmd, body, root))
    aw = round(agg["rwords"] / n, 1)
    print(f"\n  {model}")
    print(f"    clean {agg['clean']}/{n} | self-named {agg['selfname']}/{n} | "
          f"parsed {agg['parsed']}/{n} | contract-valid {agg['valid']}/{n} | avg rwords {aw}")
    if sample:
        name, reasoning, tc = sample
        print(f"    sample [{name}]:\n      {reasoning}\n      {tc.splitlines()[1]}")

def main():
    key = G.load_key()
    print(f"=== format+harness benchmark | scenarios: {', '.join(PICK)} ===")
    for model in MODELS:
        probe(model, key)

if __name__ == "__main__":
    main()
