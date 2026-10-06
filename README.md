# Business Entity Resolution — Reproducible Pipeline

Matches every Source 1 entity to its Source 2 / Source 3 records.

Final outputs:
- `output/matching_results.tsv` — final entity-resolution predictions
- `output/candidate_pairs.tsv` — full candidate set scored by the final model

**Best-submission leaderboard score: macro F0.5 = 0.985906**

---

## Environment

- **Python 3.11**, CPU only for all LightGBM steps.
- **CUDA GPU** required for the cross-encoder fine-tuning steps (steps 8a/8b).
  A consumer GPU (RTX 4050/4060, 8 GB VRAM) is sufficient; ~1 hour per encoder.
- Developed on an Apple M4 (10 cores, 16 GB RAM) for CPU steps and an RTX 4050 for GPU steps.
- No external data, no APIs. Learned artefacts: LightGBM models, a transliteration
  dictionary mined from training ground truth, and two fine-tuned cross-encoders loaded
  from public Hugging Face base checkpoints.

### Installation

```bash
# 1. Install core pipeline (CPU):
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (required for steps 8a/8b only).
#    Choose the wheel matching your CUDA driver, e.g. for CUDA 12.1:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Directory Layout

The pipeline expects the following layout (all paths are overridable via env vars):

```
student_resource/
├── dataset/{train,test}/...            # challenge data (unchanged)
├── work/                               # intermediates (auto-created)
│   ├── rerank_data/                    # cross-encoder input parquets (step 7)
│   │   ├── train_pairs.parquet
│   │   ├── heldout_pairs.parquet
│   │   ├── test_pairs.parquet
│   │   └── stress_pairs.parquet
│   └── rerank_out/                     # cross-encoder output parquets (steps 8a/8b)
│       ├── minilm/                     # MiniLM outputs
│       │   ├── ce_heldout.parquet
│       │   ├── ce_test.parquet
│       │   └── ce_stress.parquet
│       └── xlmr/                       # XLM-R outputs
│           ├── ce_heldout.parquet
│           ├── ce_testfr.parquet
│           ├── ce_testiu.parquet
│           └── ce_stress.parquet
├── output/                             # final TSVs (auto-created)
└── code/business_entity_resolution/    # this folder
    ├── src/
    │   ├── rerank/
    │   │   ├── rerank.py               # MiniLM cross-encoder
    │   │   └── rerank2.py              # XLM-R cross-encoder
    │   └── ...
    ├── README.md
    ├── requirements.txt
    └── run_all.sh
```

**Environment variable overrides:**

| Variable | Default | Purpose |
|---|---|---|
| `ER_DATA` | `../../../dataset` | Challenge data root |
| `ER_WORK` | `../../../work` | Intermediate files |
| `ER_OUTPUT` | `../../../output` | Final TSV output |
| `ER_JOBS` | CPU count | Thread count |
| `RERANK_DATA` | `$ER_WORK/rerank_data` | Cross-encoder input parquets (written by step 7) |
| `RERANK_MINILM_DIR` | `$ER_WORK/rerank_out/minilm` | MiniLM cross-encoder output dir |
| `RERANK_XLMR_DIR` | `$ER_WORK/rerank_out/xlmr` | XLM-R cross-encoder output dir |

---

## Run End-to-End

```bash
./run_all.sh
```

This script runs the entire pipeline including the GPU cross-encoder steps.
An NVIDIA GPU with CUDA is required for steps 8a/8b. All other steps are CPU-only.

Or run step by step from `src/` (see detailed table below).

**Total time:** ~6 hours CPU (M4, 10 cores) + ~2 hours GPU (RTX 4050) for the two cross-encoders.
**Peak memory:** ~12 GB (blocking the largest country).
**Disk:** ~30 GB for intermediates.

---

## Pipeline Steps

All scripts run from `src/`. Times measured on Apple M4 (10 cores).

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `s01_mine_translit.py` | Mines Indic-script → Latin token dictionaries from **training** ground-truth pairs | 10 s |
| 2 | `s02_prepare.py` | Normalises names and addresses for all six source files | 3 min |
| 3a | `s03_block.py train` | Candidate generation (train): 6 TF-IDF top-k retrievers per country, forward + reverse | ~1 h |
| 3b | `s03_block.py test` | Candidate generation (test) | ~1 h |
| 4a | `s04_features.py train` | ~60 similarity, blocking-score and competition features per candidate | ~10 min |
| 4a | `s04_features.py test` | | ~10 min |
| 4b | `s04b_diff_features.py train` | ~55 structured difference features (name alignment, legal forms, number diffs) | ~15 min |
| 4b | `s04b_diff_features.py test` | | ~15 min |
| 4c | `s04c_token_features.py train` | Learned token log-odds (tables from reserved folds 9, 15, 19) | ~7 min |
| 4c | `s04c_token_features.py test` | | ~7 min |
| 4d | `s04d_name_stats.py train` | Namesake counts (how many S1/S2+S3 records share each exact core name) | ~2 min |
| 4d | `s04d_name_stats.py test` | | ~2 min |
| 5 | `s05_train.py all` | Stage-1 LightGBM (cross-fitted), out-of-fold scores, stage-2 LightGBM, decision tuning, held-out report | ~1 h |
| 4e | `s04e_edit_features.py train` | Edit-type features (uses stage-1 OOF scores from step 5) | ~10 min |
| 4e | `s04e_edit_features.py test` | | ~10 min |
| 6 | `s06_predict.py` | Cascade filter + stage-2 scoring on test; writes `candidate_pairs.tsv` and LightGBM-only `matching_results.tsv` | ~25 min |
| 7 | `s07_export_rerank.py` | Exports uncertain pairs (LightGBM probability 0.003–0.997) as text for cross-encoders | ~5 min |
| 8a | `rerank/rerank.py all` | **GPU** MiniLM cross-encoder fine-tuning + scoring (~40 min main + ~20 min stress) | ~1 h GPU |
| 8b | `rerank/rerank2.py run --with_iu` | **GPU** XLM-R cross-encoder fine-tuning + scoring | ~1 h GPU |
| 9 | `s08b_blend2.py` | Blends both cross-encoders with LightGBM; picks variant/weight on tuning folds; writes `matching_results.tsv` | ~5 min |
| 10 | `s09_stacker.py` | Trains learned stacker (LightGBM meta-model) for India/US and overwrites India/US rows in `matching_results.tsv` | ~5 min |

**Step 4e must run after step 5** (it needs the stage-1 out-of-fold parquets written by `s05_train.py`).

---

### Cross-encoder steps in detail (steps 7–10)

The `run_all.sh` script handles all of this automatically. If running manually from `src/`:

**Step 7:** Export uncertain pairs to `$RERANK_DATA`.
```bash
export RERANK_DATA="${RERANK_DATA:-../../../work/rerank_data}"
python3 s07_export_rerank.py "${RERANK_DATA}"
```
This writes `${RERANK_DATA}/{train,heldout,test,stress}_pairs.parquet`.

**Step 8a:** MiniLM cross-encoder (GPU required).
```bash
export RERANK_OUT="${RERANK_MINILM_DIR:-../../../work/rerank_out/minilm}"
python3 rerank/rerank.py all
# Reads from $RERANK_DATA, writes to $RERANK_OUT:
#   ce_heldout.parquet, ce_test.parquet, ce_stress.parquet
```

**Step 8b:** XLM-R cross-encoder (GPU required).
```bash
export RERANK_OUT="${RERANK_XLMR_DIR:-../../../work/rerank_out/xlmr}"
python3 rerank/rerank2.py run --with_iu
# Reads from $RERANK_DATA, writes to $RERANK_OUT:
#   ce_heldout.parquet, ce_testfr.parquet, ce_testiu.parquet, ce_stress.parquet
```

**Step 9:** Blend cross-encoders.
```bash
python3 s08b_blend2.py "${RERANK_MINILM_DIR}" "${RERANK_XLMR_DIR}" --write
```

**Step 10:** Apply stacker for India/US.
```bash
python3 s09_stacker.py
python3 s09_stacker.py --write "${ER_OUTPUT:-../../../output}" "${ER_OUTPUT:-../../../output}"
```

---

## Source Files

### Core pipeline

| File | Role |
|---|---|
| `config.py` | Paths and shared settings |
| `textnorm.py` | Normalisation (legal forms, abbreviations, alias splitting, leetspeak, URL/phone stripping, Indic transliteration, state canonicalisation) |
| `s01_mine_translit.py` | Indic→Latin dictionaries from training ground-truth pairs |
| `s02_prepare.py` | Normalise all six source files |
| `s03_block.py` | Candidate generation: six TF-IDF top-k retrievers per country, forward + reverse |
| `s04_features.py` | Similarity, blocking-score and competition features |
| `s04b_diff_features.py` | Structured difference features |
| `s04c_token_features.py` | Learned token log-odds; self-training for low-coverage vocabularies |
| `s04d_name_stats.py` | Namesake counts |
| `s04e_edit_features.py` | Edit-type features (substitution vs. insertion/deletion/transposition) |
| `s05_train.py` | Stage-1 cross-fit, out-of-fold scores, stage-2 LightGBM, decision tuning, held-out report |
| `s06_predict.py` | Cascade filter (stage B), stage-2 scoring, writes `candidate_pairs.tsv` |
| `s07_export_rerank.py` | Exports uncertain pairs as text for the cross-encoders |
| `rerank/rerank.py` | MiniLM cross-encoder fine-tuning and scoring (GPU) |
| `rerank/rerank2.py` | XLM-R cross-encoder fine-tuning and scoring (GPU) |
| `s08_blend.py` | Blend one cross-encoder score with LightGBM (v7 step; superseded by s08b for the best submission) |
| `s08b_blend2.py` | Blend both cross-encoders; pick variant/weight automatically; writes `matching_results.tsv` |
| `s09_stacker.py` | Learned stacker for India/US over stage-2 + both cross-encoders; final `matching_results.tsv` |
| `stage2.py` | Context features built from stage-1 probabilities (competition, cluster consensus) |
| `decide.py` | One-to-one constraint and expected-F0.5 set selection |
| `tokstats.py` | Token log-odds tables and features |
| `metrics.py` | Macro F0.5 exactly as defined by the challenge |

### Experimental / ablation scripts

These files are retained for auditability but are **not** required for reproducing the submitted 0.985906 result.

| File | Purpose |
|---|---|
| `s05b_stage3.py` | Ablation: cross-fitted stage-2 → stage-3 (not used in final submission) |
| `exp_block_recall.py` | Blocking recall measurement experiment |
| `exp_errors.py` | Held-out error decomposition (blocking misses / FP / FN) |
| `exp_proxy_recall.py` | Label-free blocking check used for France |
| `exp_country_check.py` | Country coverage diagnostics |
| `exp_expected_f.py` | Expected-F0.5 decision rule ablations |
| `exp_selftrain.py` | Self-training token tables experiment |
| `exp_selftrain_model.py` | Self-training model experiment |
| `exp_xgb_stage2.py` | XGBoost stage-2 comparison |
| `make_variant.py` | Synthetic variant generation |
| `make_fr_variants.py` | France variant generation |
| `make_fr_variants2.py` | France variant generation (v2) |
| `predict_country.py` | Country-specific prediction ablation |

---

## Held-out Performance

Entities never used for fitting or tuning (folds 8, 18):

| Segment | Macro F0.5 |
|---|---|
| India | 0.9908 |
| US | 0.9905 |
| **Overall** | **0.9907** |

Leaderboard (test set): **0.985906**

Candidate set: **4.52 per Source 1 entity** after the cascade filter in `s06_predict.py` (`CAND_P1 = 0.001`).
Blocking recall: India 98.64%, US 99.11%, France (label-free proxy) 99.89%.

---

## Model Licences

| Model | Source | Licence | Download |
|---|---|---|---|
| LightGBM | Microsoft | MIT | `pip install lightgbm` |
| `paraphrase-multilingual-MiniLM-L12-v2` | sentence-transformers / Hugging Face | Apache-2.0 | automatic on first run |
| `xlm-roberta-base` | Meta AI / Hugging Face | MIT | automatic on first run |

No other pretrained models are used. No external data or lookup services are accessed at runtime.
