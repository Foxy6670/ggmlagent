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
  * Corpus: boonie_corpus.jsonl from datagen/export_v3.py (1101 samples,
    ~714k tokens: <think> first-person deliberation + persistent working
    prose + <tool_call>).

Before running:
  1. Upload boonie_corpus.jsonl via the Files panel -> /content/boonie_corpus.jsonl
  2. (Recommended) Mount Drive so the GGUF survives session end:
       from google.colab import drive; drive.mount('/content/drive')
     then set OUTPUT to e.g. /content/drive/MyDrive/boonie_v3
  3. Run:  !python train_boonie_v3.py

Output:
  <OUTPUT>/lora/   — LoRA adapter (~150 MB checkpoint)
  <OUTPUT>/gguf/   — Q5_K_M GGUF for KoboldCPP on the TUF (~6.5 GB)
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ─── tuning knobs ──────────────────────────────────────────────────────
MODEL       = "Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1"
MAX_SEQ_LEN = 4096           # longest corpus sample ≈ 3k tokens; 9B-4bit on T4
                             # has headroom to raise to 8192 if ever needed
CORPUS      = "/content/boonie_corpus.jsonl"
OUTPUT      = "boonie_v3"
EPOCHS      = 2              # 1101 samples; watch loss — if still falling hard
                             # at epoch 2's end, a 3rd epoch is cheap
LR          = 2e-4

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
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

# ─── dataset ───────────────────────────────────────────────────────────
ds = load_dataset("json", data_files=CORPUS, split="train")

def format_chat(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False,
    )}

ds = ds.map(format_chat, remove_columns=ds.column_names)
print(f"[data] {len(ds)} samples ready")

# ─── template tripwire ─────────────────────────────────────────────────
# Qwen-family chat templates often STRIP <think> from non-final assistant
# turns. Our corpus's entire point is first-person deliberation inside think —
# if the template drops it, training silently loses the dissociation fix.
raw = load_dataset("json", data_files=CORPUS, split="train")
src_thinks = sum(m["content"].count("<think>")
                 for ex in raw.select(range(len(raw))) for m in ex["messages"])
out_thinks = sum(t.count("<think>") for t in ds["text"])
print(f"[check] <think> blocks: corpus={src_thinks} rendered={out_thinks}")
if out_thinks < src_thinks:
    raise SystemExit(
        f"FATAL: chat template stripped {src_thinks - out_thinks} <think> blocks "
        "from rendered text. Fix: render with a template that preserves think in "
        "prior turns (e.g. patch tokenizer.chat_template to drop the strip branch) "
        "before training.")

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
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_ratio                = 0.1,
        learning_rate               = LR,
        lr_scheduler_type           = "cosine",
        logging_steps               = 5,
        save_strategy               = "epoch",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        fp16                        = True,   # T4 = Turing, no hardware bf16
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

# Q8_0 = Boonie's deployment quant (TUF; spills a little VRAM, quality first).
# Q5_K_M = the smaller sibling (fallback / faster option) from the same merge.
print("[export] producing GGUF Q8_0 + Q5_K_M for KoboldCPP…")
model.save_pretrained_gguf(f"{OUTPUT}/gguf", tokenizer,
                           quantization_method=["q8_0", "q5_k_m"])
print(f"[done] GGUFs at {OUTPUT}/gguf/")
