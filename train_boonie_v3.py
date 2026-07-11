#!/usr/bin/env python3
"""
LoRA fine-tune for Boonie V3 — Qwen3.5-9B-GLM5.1-Distill on the frontier corpus.
Designed for Google Colab with a T4 GPU (15 GiB VRAM).

Differences from train_boonie_14b.py (V2 era):
  * Base: Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1 — the V3 base decision
    (format-clean on <tool_call>, dissociation-resistant, self-checkpointing).
    No unsloth pre-quant exists; loads fp16 (~18 GB download) and quantizes to
    4-bit on the fly. 9B-4bit ≈ 5.5 GiB VRAM — comfortable on a T4.
  * train_on_responses_only: loss lands ONLY on assistant turns — the corpus's
    tool-result messages are environment, not behavior; training on them
    teaches result-hallucination.
  * Corpus: boonie_corpus.jsonl from datagen/export_v3.py — sample/token
    counts grow across versions; check the [export] log line at generation
    time for the current corpus's actual numbers.

Before running:
  1. Upload boonie_corpus.jsonl via the Files panel -> /content/boonie_corpus.jsonl
  2. (Recommended) Mount Drive so the GGUF survives session end:
       from google.colab import drive; drive.mount('/content/drive')
     then set OUTPUT to e.g. /content/drive/MyDrive/boonie_v3
  3. Run:  !python train_boonie_v3.py

Output:
  <OUTPUT>/lora/   — LoRA adapter (~150 MB checkpoint)
  <OUTPUT>/gguf/   — Q6_K GGUF for KoboldCPP on the TUF (~6.5 GB)
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Ampere (A100/L4) has hardware bf16 — more stable than fp16, no loss scaling.
# Turing (T4) does not, so fall back to fp16 there. Auto-detect so the same
# script is correct on either runtime.
_BF16 = torch.cuda.is_bf16_supported()
print(f"[precision] bf16={_BF16} (fp16={not _BF16})")

# ─── tuning knobs ──────────────────────────────────────────────────────
# Base/output/batch are env-overridable so ONE script trains either target:
#   9B  (default):   !python train_boonie_v3.py
#   14B (control):   !BOONIE_BASE=unsloth/Qwen3-14B-bnb-4bit \
#                     BOONIE_OUT=boonie_14b_v3_1 BOONIE_BATCH=2 python train_boonie_v3.py
# The 14B is the CONTROL experiment — does SFT on the V3.1 corpus fix the framing
# that prompting alone couldn't? Same corpus, same recipe; only the base differs.
MODEL       = os.environ.get("BOONIE_BASE", "Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1")
MAX_SEQ_LEN = 4096           # longest corpus sample ≈ 3k tokens; 4bit has headroom
CORPUS      = "/content/boonie_corpus.jsonl"   # upload boonie_corpus_v3.jsonl here
OUTPUT      = os.environ.get("BOONIE_OUT", "boonie_v3_1")
BATCH       = int(os.environ.get("BOONIE_BATCH", "4"))   # 14B: set 2 (bigger model)
EPOCHS      = 2              # ~1k samples; watch loss — epoch-1 ckpt often the best
                             # deploy (V3.0 overfit slightly by epoch 2)
LR          = 2e-4

# Optional: push the GGUF straight to a HF Hub repo instead of (or alongside)
# saving locally. Point of this: downloading a 15-20GB export to the TUF over
# a home connection takes 10+ minutes of billed Colab compute for zero GPU
# work. Pushing to HF is datacenter-to-datacenter (fast, usually ~1-2 min),
# so you disconnect Colab right after the push and pull down to the TUF at
# your own pace afterward, at zero ongoing compute cost.
#   HF_PUSH_REPO=Foxy6670/boonie-v3-2-gguf python train_boonie_v3.py
# Needs HF_TOKEN set (Colab: use the Secrets panel, or huggingface_hub.login()
# beforehand) with write access. Leave HF_PUSH_REPO unset to skip entirely --
# default behavior (local save only) is unchanged.
HF_PUSH_REPO = os.environ.get("HF_PUSH_REPO", "")

# Separate opt-in: only fires if the HF push above actually happened (nothing
# to disconnect-after otherwise), so pushing to HF for other reasons doesn't
# surprise-kill your session. Not the default even with HF_PUSH_REPO set --
# some runs you'll want to stay up to look at the loss curve.
#   HF_PUSH_REPO=... AUTO_DISCONNECT=1 python train_boonie_v3.py
AUTO_DISCONNECT = os.environ.get("AUTO_DISCONNECT", "") == "1"

# ─── load base + attach LoRA ───────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    dtype          = None,   # T4 -> fp16
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = True,   # standard checkpointing; NO gradient
                                         # offload (the "unsloth" mode shuttles
                                         # grads to host RAM to save VRAM we have
                                         # in surplus on an 80GB A100 — pure slowdown)
    random_state               = 3407,
)

# ─── dataset ───────────────────────────────────────────────────────────
ds = load_dataset("json", data_files=CORPUS, split="train")

# The stock Qwen3 chat template STRIPS <think> from non-final assistant turns
# (matching how the harness omits prior think from context — see agent.py
# _build_messages, which rebuilds history from agent_text only). But stripping +
# per-turn loss would train the model to sometimes emit NO think, undermining the
# whole dissociation fix. So we render ChatML explicitly and keep think in every
# assistant turn: uniform "always deliberate in first person" signal. The mild
# train/infer divergence (training conditions on prior think, inference won't have
# it) is benign — think is self-contained; the persistent prose carries working
# memory in both. role:tool is wrapped in <tool_response> exactly as the stock
# template does at inference, so the generation target format matches deployment.

def render_chatml(messages):
    out = []
    for i, m in enumerate(messages):
        role, content = m["role"], m["content"]
        if role == "tool":
            prev_tool = i > 0 and messages[i - 1]["role"] == "tool"
            next_tool = i + 1 < len(messages) and messages[i + 1]["role"] == "tool"
            if not prev_tool:
                out.append("<|im_start|>user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if not next_tool:
                out.append("<|im_end|>\n")
        else:
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)

# Structural parity check on a think-BEARING single turn — the real data path,
# and the format the model must learn to GENERATE. (A think-LESS probe is
# misleading: stock Qwen3 auto-injects an empty <think></think>, which our real
# corpus turns never trigger since they all carry explicit think tags.) The
# stock template keeps think on a final/only assistant turn, so this must match.
_c = "<think>\nI, Boonie, will read the file.\n</think>\n\nReading it now.\n\n<tool_call>\n{}\n</tool_call>"
_probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
          {"role": "assistant", "content": _c}]
_stock = tokenizer.apply_chat_template(_probe, tokenize=False, add_generation_prompt=False)
if render_chatml(_probe) != _stock:
    print("[warn] manual ChatML != stock on think-bearing turn — generation target "
          "format may diverge from deployment; inspect:\n  stock : " + repr(_stock)
          + "\n  manual: " + repr(render_chatml(_probe)))

def format_chat(ex):
    return {"text": render_chatml(ex["messages"])}

ds = ds.map(format_chat, remove_columns=ds.column_names)
print(f"[data] {len(ds)} samples ready")

# Tripwire: every <think> in the corpus must survive into the rendered text.
raw = load_dataset("json", data_files=CORPUS, split="train")
src_thinks = sum(m["content"].count("<think>")
                 for ex in raw.select(range(len(raw))) for m in ex["messages"])
out_thinks = sum(t.count("<think>") for t in ds["text"])
print(f"[check] <think> blocks: corpus={src_thinks} rendered={out_thinks}")
if out_thinks < src_thinks:
    raise SystemExit(
        f"FATAL: {src_thinks - out_thinks} <think> blocks lost in rendering — "
        "render_chatml should preserve all of them; investigate before training.")

# ─── train ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    args = SFTConfig(
        output_dir                  = OUTPUT,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH,   # A100 has room; effective batch =
        gradient_accumulation_steps = 2,        # BATCH*2 (8 for the 9B) — keeps
                                                # enough steps on a ~1k-sample corpus.
        warmup_ratio                = 0.1,
        learning_rate               = LR,
        lr_scheduler_type           = "cosine",
        logging_steps               = 5,
        save_strategy               = "epoch",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        fp16                        = not _BF16,   # T4=fp16, A100/L4=bf16
        bf16                        = _BF16,
        seed                        = 3407,
        report_to                   = "none",
    ),
)

# Mask loss to assistant turns only (Qwen chat template markers). System
# prompts and tool results are context to condition on, not behavior to learn.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

trainer.train()

# ─── export ────────────────────────────────────────────────────────────
print("[export] saving LoRA adapter…")
model.save_pretrained(f"{OUTPUT}/lora")
tokenizer.save_pretrained(f"{OUTPUT}/lora")

# Free the 19.3 GB HF download cache before GGUF export — the merged fp16
# (~19 GB) + F16 GGUF (~19 GB) intermediates don't fit alongside it on a
# 112 GB Colab disk. Weights are already in memory; the cache is dead weight.
import shutil, os as _os
shutil.rmtree(_os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
print("[export] HF cache purged for disk headroom")

# Q8_0 = master/deployment quant (TUF; spills a little VRAM, quality first).
# Q6_K = the smaller sibling from the same merge — kept at Q6+ deliberately,
# not Q5, per the "don't go below Q6 for agentic tasks" floor. Also
# reproducible later via `llama-quantize --allow-requantize` from the Q8_0
# alone, so this export isn't the only way to get it if a run is misconfigured.
print("[export] producing GGUF Q8_0 + Q6_K for KoboldCPP…")
model.save_pretrained_gguf(f"{OUTPUT}/gguf", tokenizer,
                           quantization_method=["q8_0", "q6_k"])
print(f"[done] GGUFs at {OUTPUT}/gguf/")

if HF_PUSH_REPO:
    print(f"[export] pushing GGUF to hf.co/{HF_PUSH_REPO} …")
    model.push_to_hub_gguf(HF_PUSH_REPO, tokenizer,
                            quantization_method=["q8_0", "q6_k"])
    print(f"[done] pushed to https://huggingface.co/{HF_PUSH_REPO} — "
          f"safe to disconnect this Colab session now; pull down to the TUF "
          f"with: huggingface-cli download {HF_PUSH_REPO} --local-dir <dir>")

    if AUTO_DISCONNECT:
        try:
            from google.colab import runtime
            print("[export] AUTO_DISCONNECT=1 — marking runtime for deletion now.")
            runtime.unassign()
        except ImportError:
            print("[export] AUTO_DISCONNECT=1 but not running in Colab — skipping.")
