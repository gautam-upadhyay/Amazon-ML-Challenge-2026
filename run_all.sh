#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline: raw TSVs → normalisation → blocking → features →
# LightGBM models → cross-encoder reranking → blending → stacking → outputs.
#
# Reproduces the best submission (macro F0.5 = 0.985906).
#
# REQUIREMENTS:
#   - Python 3.11 with all packages from requirements.txt installed.
#   - An NVIDIA GPU with CUDA for steps 8a and 8b (cross-encoder fine-tuning).
#     A consumer GPU (RTX 4050/4060, 8 GB VRAM) is sufficient.
#   - All other steps run on CPU only.
#
# DIRECTORY LAYOUT (override with env vars):
#   ER_DATA       → challenge data root        (default: ../../../dataset)
#   ER_WORK       → intermediate files          (default: ../../../work)
#   ER_OUTPUT     → final TSV output            (default: ../../../output)
#   ER_JOBS       → thread count                (default: cpu count)
#   RERANK_DATA   → reranker input parquets     (default: $ER_WORK/rerank_data)
#   RERANK_MINILM_DIR → MiniLM output parquets  (default: $ER_WORK/rerank_out/minilm)
#   RERANK_XLMR_DIR   → XLM-R output parquets   (default: $ER_WORK/rerank_out/xlmr)
#
# TOTAL RUNTIME: ~6 h CPU (Apple M4, 10 cores) + ~2 h GPU (RTX 4050)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/src"

# ── Resolve all directory paths ──────────────────────────────────────────────
WORK="${ER_WORK:-../../../work}"
OUTPUT="${ER_OUTPUT:-../../../output}"
export RERANK_DATA="${RERANK_DATA:-${WORK}/rerank_data}"
export RERANK_MINILM_DIR="${RERANK_MINILM_DIR:-${WORK}/rerank_out/minilm}"
export RERANK_XLMR_DIR="${RERANK_XLMR_DIR:-${WORK}/rerank_out/xlmr}"

echo ""
echo "Pipeline directories:"
echo "  ER_DATA    = ${ER_DATA:-../../../dataset}"
echo "  ER_WORK    = ${WORK}"
echo "  ER_OUTPUT  = ${OUTPUT}"
echo "  RERANK_DATA       = ${RERANK_DATA}"
echo "  RERANK_MINILM_DIR = ${RERANK_MINILM_DIR}"
echo "  RERANK_XLMR_DIR   = ${RERANK_XLMR_DIR}"
echo ""

# ── CPU steps 1–6 ───────────────────────────────────────────────────────────

echo "=== Step 1: Indic→Latin token dictionaries from TRAIN pairs ==="
python3 s01_mine_translit.py

echo "=== Step 2: Normalise all six source files ==="
python3 s02_prepare.py

echo "=== Step 3: Candidate generation (train + test) ==="
python3 s03_block.py train
python3 s03_block.py test

echo "=== Step 4a: Pair similarity + competition features ==="
python3 s04_features.py train
python3 s04_features.py test

echo "=== Step 4b: Structured difference features ==="
python3 s04b_diff_features.py train
python3 s04b_diff_features.py test

echo "=== Step 4c: Learned token log-odds ==="
python3 s04c_token_features.py train
python3 s04c_token_features.py test

echo "=== Step 4d: Namesake counts ==="
python3 s04d_name_stats.py train
python3 s04d_name_stats.py test

echo "=== Step 5: Stage-1 cross-fit, OOF scores, stage-2, decision tuning ==="
python3 s05_train.py all

echo "=== Step 4e: Edit-type features (requires stage-1 OOF scores from step 5) ==="
python3 s04e_edit_features.py train
python3 s04e_edit_features.py test

echo "=== Step 6: Cascade filter + stage-2 scoring → LightGBM-only outputs ==="
python3 s06_predict.py

# ── GPU steps 7–10 ──────────────────────────────────────────────────────────
# Steps 8a and 8b require an NVIDIA GPU with CUDA.
# The rerank scripts read from $RERANK_DATA and write to $RERANK_OUT.

echo "=== Step 7: Export uncertain pairs for cross-encoders ==="
mkdir -p "${RERANK_DATA}"
python3 s07_export_rerank.py "${RERANK_DATA}"

echo "=== Step 8a: MiniLM cross-encoder fine-tuning + scoring (GPU, ~1 h) ==="
mkdir -p "${RERANK_MINILM_DIR}"
export RERANK_OUT="${RERANK_MINILM_DIR}"
python3 rerank/rerank.py all

echo "=== Step 8b: XLM-R cross-encoder fine-tuning + scoring (GPU, ~1 h) ==="
mkdir -p "${RERANK_XLMR_DIR}"
export RERANK_OUT="${RERANK_XLMR_DIR}"
python3 rerank/rerank2.py run --with_iu

echo "=== Step 9: Blend both cross-encoders with LightGBM scores ==="
python3 s08b_blend2.py "${RERANK_MINILM_DIR}" "${RERANK_XLMR_DIR}" --write

echo "=== Step 10: Learned stacker for India/US → final matching_results.tsv ==="
python3 s09_stacker.py
python3 s09_stacker.py --write "${OUTPUT}" "${OUTPUT}"

echo ""
echo "=== Done. Final outputs in ${OUTPUT}/ ==="
echo "    matching_results.tsv"
echo "    candidate_pairs.tsv"
echo ""
echo "To validate, run the challenge's validate_submission.py (from student_resource/):"
echo "  python3 validate_submission.py \\"
echo "      --matching  ${OUTPUT}/matching_results.tsv \\"
echo "      --candidate ${OUTPUT}/candidate_pairs.tsv \\"
echo "      --test-dir  \${ER_DATA:-../../../dataset}/test"
