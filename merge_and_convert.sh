#!/usr/bin/env bash
# Convert a LoRA adapter OR a pre-merged HF model into GGUF, then quantize.
#
# If INPUT contains config.json + model.safetensors (sharded or not), it's
# treated as a pre-merged model and the merge step is skipped.
# Otherwise INPUT is treated as a LoRA adapter dir and merged into BASE_MODEL.
#
# Usage:
#   ./merge_and_convert.sh [INPUT] [OUT_NAME]
#
# Env overrides:
#   BASE_MODEL   HF id or local path of the base (default: unsloth/Qwen3.5-4B)
#   LLAMA_CPP    path to llama.cpp checkout       (default: ~/llama.cpp)
#   VENV         python venv to activate          (default: ~/unsloth-venv)
#   QUANT        quant type for llama-quantize    (default: Q5_K_M)
#   WORK         scratch dir for the merged model (default: ./boonie-merged)

set -euo pipefail

INPUT="${1:-./boonie-lora}"
OUT_NAME="${2:-boonie-qwen3.5-4b}"

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.5-4B}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
VENV="${VENV:-$HOME/unsloth-venv}"
QUANT="${QUANT:-Q5_K_M}"
WORK="${WORK:-./boonie-merged}"

F16_GGUF="${OUT_NAME}-f16.gguf"
QUANT_GGUF="${OUT_NAME}-${QUANT}.gguf"

if [[ ! -d "$INPUT" ]]; then
    echo "error: INPUT '$INPUT' does not exist" >&2
    exit 1
fi
if [[ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]]; then
    echo "error: convert_hf_to_gguf.py not found under '$LLAMA_CPP'" >&2
    echo "       set LLAMA_CPP=/path/to/llama.cpp" >&2
    exit 1
fi

# Detect: pre-merged model (has config.json + safetensors) vs LoRA adapter.
shopt -s nullglob
SAFETENSORS=("$INPUT"/model*.safetensors)
shopt -u nullglob
if [[ -f "$INPUT/config.json" && ${#SAFETENSORS[@]} -gt 0 ]]; then
    PREMERGED=1
    MERGED_DIR="$INPUT"
    echo "detected pre-merged model at $INPUT — skipping merge step"
else
    PREMERGED=0
    MERGED_DIR="$WORK"
    echo "detected LoRA adapter at $INPUT — will merge into $BASE_MODEL"
fi

# llama-quantize binary moved around between llama.cpp versions; try both.
QUANT_BIN=""
for cand in "$LLAMA_CPP/build/bin/llama-quantize" "$LLAMA_CPP/llama-quantize" "$LLAMA_CPP/build/bin/quantize"; do
    if [[ -x "$cand" ]]; then QUANT_BIN="$cand"; break; fi
done
if [[ -z "$QUANT_BIN" ]]; then
    echo "error: llama-quantize binary not found under '$LLAMA_CPP'" >&2
    echo "       build llama.cpp first (cmake -B build && cmake --build build -j)" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

if [[ "$PREMERGED" -eq 0 ]]; then
    echo "=== [1/3] merging LoRA adapter into base ==="
    echo "    base    : $BASE_MODEL"
    echo "    adapter : $INPUT"
    echo "    output  : $WORK"
    python3 - <<PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "${BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print("loaded base")
m = PeftModel.from_pretrained(base, "${INPUT}")
print("loaded adapter, merging...")
m = m.merge_and_unload()
m.save_pretrained("${WORK}", safe_serialization=True)
AutoTokenizer.from_pretrained("${BASE_MODEL}").save_pretrained("${WORK}")
print("merged model written to ${WORK}")
PYEOF
else
    echo "=== [1/3] skipped — input already merged ==="
fi

echo
echo "=== [2/3] converting to GGUF F16 ==="
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16

echo
echo "=== [3/3] quantizing to $QUANT ==="
"$QUANT_BIN" "$F16_GGUF" "$QUANT_GGUF" "$QUANT"

echo
echo "=== done ==="
ls -lh "$F16_GGUF" "$QUANT_GGUF"
echo
echo "ship to Gateway:"
echo "  scp $QUANT_GGUF gateway:/path/to/models/"
