#!/usr/bin/env python3
"""
LoRA fine-tune Qwen3-14B on Boonie's corpus.
Designed for Google Colab with a T4 GPU (15 GiB VRAM).

Before running:
  1. Upload boonie_corpus.jsonl via the Files panel — lands at /content/boonie_corpus.jsonl
  2. (Recommended) Mount Google Drive so the GGUF survives session end:
       from google.colab import drive; drive.mount('/content/drive')
     then set OUTPUT below to a Drive path, e.g. /content/drive/MyDrive/boonie_14b
  3. Run:  !python train_boonie_14b.py

Output:
  <OUTPUT>/lora/         — LoRA adapter only (~200 MB, fast checkpoint)
  <OUTPUT>/gguf/         — Q5_K_M GGUF ready for KoboldCPP on TUF (~10 GB)
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ─── tuning knobs ──────────────────────────────────────────────────────
MODEL       = "unsloth/Qwen3-14B"
MAX_SEQ_LEN = 4096           # T4 (15 GiB) handles 14B-4bit at 4096 comfortably
                             # raise to 8192 if nvidia-smi shows >4 GiB free after load
CORPUS      = "/content/boonie_corpus.jsonl"   # default Colab upload path
OUTPUT      = "boonie_14b"                     # or a Drive path — see header
EPOCHS      = 2
LR          = 2e-4

# ─── load base + attach LoRA ───────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    dtype          = None,   # auto-detect; T4 will pick fp16
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
print(f"[data] {len(ds)} sessions ready")

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

trainer.train()

# ─── export ────────────────────────────────────────────────────────────
# Save LoRA adapter first — small, fast, survives if GGUF export runs long
print("[export] saving LoRA adapter…")
model.save_pretrained(f"{OUTPUT}/lora")
tokenizer.save_pretrained(f"{OUTPUT}/lora")

# Q5_K_M GGUF — merges adapter + quantizes in one step (~10 GB output)
# If this OOMs, try quantization_method="q4_k_m" instead
print("[export] producing GGUF Q5_K_M for KoboldCPP…")
model.save_pretrained_gguf(f"{OUTPUT}/gguf", tokenizer, quantization_method="q5_k_m")
print(f"[done] GGUF at {OUTPUT}/gguf/")
