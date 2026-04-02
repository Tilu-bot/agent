#!/usr/bin/env bash
# export_gguf.sh — Convert a fine-tuned HuggingFace model to GGUF and
#                  optionally import it into Ollama.
#
# Usage:
#   bash scripts/export_gguf.sh <model_dir> [quant_type] [model_name]
#
# Arguments:
#   model_dir   Path to the saved HuggingFace model (output of fine_tune.py).
#               Default: ./ft-model
#   quant_type  llama.cpp quantisation type. Default: Q4_K_M
#               Other options: Q2_K  Q3_K_M  Q5_K_M  Q8_0  F16
#   model_name  Name to use when creating the Ollama model.
#               Default: agentic-ft
#
# Requirements:
#   - llama.cpp must be cloned and built alongside this repo, OR
#     the LLAMA_CPP_DIR environment variable must point to its directory.
#   - Python ≥ 3.10 with the 'gguf' package: pip install gguf
#   - Ollama installed (optional; only needed for the Ollama import step).
#
# Example:
#   LLAMA_CPP_DIR=~/llama.cpp bash scripts/export_gguf.sh ./ft-model Q4_K_M my-agent

set -euo pipefail

MODEL_DIR="${1:-./ft-model}"
QUANT_TYPE="${2:-Q4_K_M}"
MODEL_NAME="${3:-agentic-ft}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Locate llama.cpp ──────────────────────────────────────────────────────────
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/../llama.cpp}"
if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    echo "ERROR: llama.cpp not found at ${LLAMA_CPP_DIR}"
    echo "Clone it with:"
    echo "  git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP_DIR}"
    echo "  cd ${LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release -j"
    echo "Or set the LLAMA_CPP_DIR environment variable."
    exit 1
fi

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    # Older llama.cpp used a different script name
    CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert.py"
fi
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: conversion script not found in ${LLAMA_CPP_DIR}"
    exit 1
fi

QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
if [[ ! -f "${QUANTIZE_BIN}" ]]; then
    QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/quantize"
fi
if [[ ! -f "${QUANTIZE_BIN}" ]]; then
    echo "ERROR: quantize binary not found. Build llama.cpp first:"
    echo "  cd ${LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release -j"
    exit 1
fi

# ── Validate input ────────────────────────────────────────────────────────────
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: model directory not found: ${MODEL_DIR}"
    exit 1
fi

GGUF_F16="${MODEL_DIR}/model-f16.gguf"
GGUF_QUANT="${MODEL_DIR}/model-${QUANT_TYPE}.gguf"

echo "=== Step 1: Convert HuggingFace model to F16 GGUF ==="
python3 "${CONVERT_SCRIPT}" \
    --outfile "${GGUF_F16}" \
    --outtype f16 \
    "${MODEL_DIR}"
echo "F16 GGUF saved to: ${GGUF_F16}"

echo ""
echo "=== Step 2: Quantise to ${QUANT_TYPE} ==="
"${QUANTIZE_BIN}" "${GGUF_F16}" "${GGUF_QUANT}" "${QUANT_TYPE}"
echo "Quantised GGUF saved to: ${GGUF_QUANT}"

# ── Optional: import into Ollama ──────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    echo ""
    echo "=== Step 3: Import into Ollama as '${MODEL_NAME}' ==="

    MODELFILE="${MODEL_DIR}/Modelfile"
    cat > "${MODELFILE}" <<MODELFILE_EOF
FROM ${GGUF_QUANT}

SYSTEM """
You are a helpful AI agent that can plan and execute tasks using tools.
You produce structured JSON plans and tool calls.
"""

PARAMETER temperature 0.3
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
MODELFILE_EOF

    ollama create "${MODEL_NAME}" -f "${MODELFILE}"
    echo "Model '${MODEL_NAME}' created in Ollama."
    echo ""
    echo "To use it, update config.yaml:"
    echo "  models:"
    echo "    fast: ${MODEL_NAME}"
    echo "    reasoning: ${MODEL_NAME}"
else
    echo ""
    echo "Ollama not found — skipping import step."
    echo "To import manually:"
    echo "  cat > Modelfile <<EOF"
    echo "  FROM ${GGUF_QUANT}"
    echo "  EOF"
    echo "  ollama create ${MODEL_NAME} -f Modelfile"
fi

echo ""
echo "=== Done ==="
echo "GGUF file: ${GGUF_QUANT}"
