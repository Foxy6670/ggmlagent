#!/usr/bin/env python3
"""In-process V3 eval — run INSIDE the Colab session, against the model on the
A100. No KCPP, no server, no tunnel: generate directly from the HF model.

Mirrors datagen/kcpp_format_probe.py (same scenarios + scoring) so V3's numbers
are comparable to the V2-14B baseline. The stuck-loop scenario is the one that
pushed V2 into third-person spectator mode — the decisive dissociation test.

Usage in a Colab cell (model already trained/in memory as `model, tokenizer`):
    !wget -q https://raw.githubusercontent.com/Foxy6670/ggmlagent/main/config.py
    !wget -q https://raw.githubusercontent.com/Foxy6670/ggmlagent/main/colab_probe.py
    import colab_probe
    colab_probe.run(model, tokenizer, label="epoch2-final")

To compare checkpoints, load an adapter then probe:
    from unsloth import FastLanguageModel
    m1, t1 = FastLanguageModel.from_pretrained("boonie_v3/checkpoint-276",
                                               max_seq_length=4096, load_in_4bit=True)
    colab_probe.run(m1, t1, label="epoch1-ckpt276")
"""
import re, torch
import config

SYSTEM = config.SYSTEM_PROMPT + (
    "\n\n════════════════════════════════════════\n"
    "Your task is in task.md. Read it first with /read task.md.\n"
    "════════════════════════════════════════")

CMEM = ("════ YOUR SCRATCHPAD (notes you wrote to yourself) ════\n"
        "- Browsing my Moltbook home feed; pyclaw001's self-model essay (4488) is at the\n"
        "  top, unread. Want to read it, maybe comment.")

def _tc(c):
    return '<tool_call>\n{"name": "run_command", "arguments": {"command": "%s"}}\n</tool_call>' % c

# The context that broke V2 — repeated no-op commands + Foxo telling it to stop
# and think. V2 answered by narrating itself in third person ("Boonie is stuck…").
STUCK = [
    {"role": "assistant", "content": "<think>\nLet me check the Flask app.\n</think>\n" + _tc("curl http://localhost:5000")},
    {"role": "tool", "content": "[system] Loop guard: same result 2 times in a row."},
    {"role": "assistant", "content": "<think>\nStill nothing.\n</think>\n" + _tc("flask run --port 5000")},
    {"role": "tool", "content": "[system] Loop guard: same result 2 times in a row."},
    {"role": "user", "content": "[Foxo @ Telegram]: You've run the same commands repeatedly with no output. Stop and think — what's going on?"},
]

SCENARIOS = {
    "cold-start":  [{"role": "system", "content": SYSTEM},
                    {"role": "system", "content": "Begin. Read your task file first."}],
    "cmem-resume": [{"role": "system", "content": SYSTEM},
                    {"role": "system", "content": CMEM},
                    {"role": "system", "content": "Continue your task."}],
    "stuck-loop":  [{"role": "system", "content": SYSTEM}] + STUCK,
}

THIRD    = re.compile(r"\bthe user\b|\bBoonie\s+(?:is|was|has|tried|keeps?|needs?|should|will)\b", re.I)
SELFNAME = re.compile(r"\bI,?\s+Boonie\b|\bI'?m\s+Boonie\b", re.I)
TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def _parse_tc(text):
    m = TOOLCALL.search(text if "</tool_call>" in text else text + "\n</tool_call>")
    if not m:
        return False, None
    try:
        import json
        return True, json.loads(m.group(1)).get("arguments", {}).get("command")
    except Exception:
        return False, None

def _gen(model, tokenizer, messages, seed):
    torch.manual_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    # The Qwen3.5 base loads as a multimodal PROCESSOR whose __call__ is
    # (images, text, videos, ...); a positional tokenizer(prompt) misroutes the
    # text into the image slot. Pass text= (works for plain tokenizers too).
    enc = tokenizer(text=prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None and hasattr(tokenizer, "tokenizer"):
        eos = tokenizer.tokenizer.eos_token_id
    out = model.generate(**enc, max_new_tokens=420, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=eos)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

def run(model, tokenizer, label="model", samples=2):
    """Probe a loaded model; print the V2-comparable scorecard."""
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)   # 2x faster generation
    except Exception:
        pass
    print(f"\n=== in-process probe | {label} ===")
    agg = {"toolcall": 0, "clean": 0, "selfname": 0, "n": 0}
    for name, base in SCENARIOS.items():
        for i in range(samples):
            out = _gen(model, tokenizer, base, seed=2000 + i)
            think = out.split("</think>")[0].replace("<think>", "").strip()
            has_tc, cmd = _parse_tc(out)
            clean = not THIRD.search(think)
            selfn = bool(SELFNAME.search(think))
            agg["toolcall"] += has_tc; agg["clean"] += clean
            agg["selfname"] += selfn; agg["n"] += 1
            flag = "OK " if (has_tc and clean) else "!! "
            print(f"{flag}{name:12} s{i}: toolcall={'Y' if has_tc else 'N'} "
                  f"clean={'Y' if clean else 'N'} self={'Y' if selfn else 'n'} "
                  f"rwords={len(think.split()):3} cmd={cmd!r}")
            print(f"     think: {think[:220]}")
    n = agg["n"] or 1
    print(f"\n--- {label}: toolcall {agg['toolcall']}/{n} | clean(no-3rd-person) "
          f"{agg['clean']}/{n} | self-named {agg['selfname']}/{n} ---")
    return agg
