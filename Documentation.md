# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Definitely_Not_Overfit
**Team Members:** Karthik Aljapur, SATHI Kaushik Kumar Reddy, Gautam Prasad Upadhyay, Bhanu Prakash Kusha — SRM Institute of Science and Technology (SRMIST), Kattankulathur, Chennai
**Submission Date:** 28 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 records to Source 1 entities in five parts:

1. Country-agnostic text normalisation.
2. Bidirectional TF-IDF blocking, including a name×address "cross-feature" retriever.
3. A **cascade**: a stage-1 LightGBM scorer acts as a candidate filter, cutting the candidate set from ~51 to **4.5 records per Source 1 entity** while preserving all true matches that were present in the blocked candidate set.
4. The final matcher: a stage-2 LightGBM that reasons over the whole candidate graph, blended with two fine-tuned multilingual cross-encoder transformers on the pairs LightGBM is unsure about.
5. A decision layer that enforces the one-to-one structure of the data and picks each entity's match set by maximising expected F0.5.

On held-out training entities never used for fitting or tuning, the final pipeline scores **macro F0.5 = 0.9907** (India 0.9908, US 0.9905). Leaderboard: 0.9837 → 0.9849 → 0.9858 → **0.985906**.

---

## 2. Methodology

### 2.1 Problem Analysis

These findings, from profiling the 2.2M training entities and 7.6M labelled matches, shaped the design:

- **Each S2/S3 record belongs to at most one S1 entity** (0 violations in 7.6M assignments). "Which S1 entity claims this record most strongly?" is therefore a first-class signal, used both as a feature and as a hard constraint at decision time.
- **Singletons are 5.6%** of S1 entities. Matched entities have 1–11 records (mode 3).
- **Distractors are near-copies of real entities** with one or two deliberate edits:
  - a changed house number (46→51);
  - a changed legal form (Private Limited → Limited / Public Limited);
  - one substituted name word (Buoyant *Computing* → Buoyant *Steels*);
  - a phonetic or ending mutation of the distinctive word (Yamuika→Yamuica, Aditin→Adityx).

  True matches carry different noise: added filler words ("Services", "Center"), dropped or misspelt legal words, random typos, reordered address components and extra prefix numbers ("Door No 31").
- **Distractors are isolated one-off copies.** A deviating house number is shared by another record attached to the same entity only 9% of the time for distractors, versus 63% for true matches. The cluster of true records agrees with itself even when the S1 record is noisy.
- **Several real S1 entities share the exact same name** at different addresses, and ~38% of singletons have exact-name twins in S2/S3. The address is usually the decisive field.
- **Indian names are extremely generic** ("Sun Services Pvt Ltd" appears on hundreds of records) while addresses are often partial ("Flat 701, Thane"). Neither field alone is distinctive; only their combination is.
- **7% of names and 9% of addresses contain Indic scripts** (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia, Gurmukhi). The test set uses exactly the same 1,518-word Indic vocabulary as training. Some Indic industry words ("मोटर्स", Motors) occur *only* in distractor records.
- **Rebrands** appear as "InventedName formerly/aka/dba/t/a RealName", or as an invented name alone at the entity's address.
- **Empty addresses** (3.3% of S2/S3) are almost always true matches (97.8%), but are genuinely ambiguous when several S1 entities share the name.
- **France** (15% of test entities) is absent from training. It brings French legal forms (SARL, SAS, EURL), street abbreviations (R, AV, IMP), and records that alternate between a region ("Hauts-de-France") and a department ("Nord") in the address.

### 2.2 Solution Strategy

**Approach Type:** Blocking + cascade filter + two-stage gradient-boosted classifier blended with transformer cross-encoders + graph-aware decision layer.
**Core Innovation:**
1. Name×address cross-feature blocking and reverse-direction blocking, followed by a model-based cascade filter that keeps 4.5 candidates per entity.
2. "Structured difference" features that describe *what kind* of edit separates two records, targeting the data's distractor recipe, plus cluster-consensus features that test whether a record agrees with the other records already attached to the same entity.
3. A second-stage model and a decision layer that exploit the one-to-one structure of the matching.
4. Multilingual cross-encoders that read both records' text, applied only to the uncertain pairs, which also carry the model to the unseen country.

Pipeline: `normalise → block (6 retrievers, both directions, ~51/entity) → ~165 pair features → stage-1 LightGBM (cross-fitted) = cascade filter (p1 ≥ 0.001, 4.5/entity) → graph + cluster context → stage-2 LightGBM → blend with cross-encoders on uncertain pairs → one-to-one + expected-F0.5 selection`.

---

## 3. Candidate Generation (Blocking)

**Normalisation (`textnorm.py`).**
- Unicode NFKC and accent folding.
- Legal forms and street types are canonicalised to *short* tokens (limited→ltd, street→st, saint→st, rue/r→rue), so the same rules behave sensibly for any country.
- Alias splitting on formerly / aka / dba / t/a / trading as.
- URL, hashtag and phone-number stripping.
- Leetspeak repair only inside mixed letter/digit tokens (at1antic→atlantic, 5terling→sterling).
- US and Indian state names/codes, and French regions and their departments (Nord → Hauts-de-France, Gironde → Nouvelle-Aquitaine, Loire-Atlantique → Pays de la Loire, …), are canonicalised to one token when they form a whole address component. Old Indian city names are mapped (Madras→Chennai).
- Indic tokens are mapped with a dictionary **mined from training ground-truth pairs**: positional alignment of equal-length name pairs, and lift-based alignment for state names. Romanisation is the fallback.

**Stage A: retrievers (`s03_block.py`).** All are TF-IDF cosine top-k with multithreaded sparse matrix products, run per country, so the work grows with the number of records, not with pairs of records:

| Retriever | Representation | Direction | k |
|---|---|---|---|
| Xf / Xr | cross features: every (name token × address token) pair | S1→S2/3 / S2/3→S1 | 15 / 3 |
| Af / Ar | name character 3-grams + address words | both | 25 / 3 |
| Br | address words | S2/3→S1 | 3 |
| Es | space-free name 3-grams, only for records with ≤4 address tokens | S2/3→S1 | 10 |

- **Cross features** stay rare even when both tokens are common, so "sun services + thane + 701" separates one business from hundreds of namesakes. They alone retrieve 91% of true pairs with 15 candidates.
- **Reverse retrieval** exploits the one-to-one structure. A record's best few S1 entities are strong candidates even when the S1 entity's own list is crowded by near-duplicate distractors. It raised recall by ~1.3 points.
- Very frequent features are pruned per retriever for speed. Reverse-only candidates are capped at 30 per S1 entity, which stops "hub" entities (addresses made only of common tokens) from collecting tens of thousands of candidates.

**Stage B: cascade filter (`s06_predict.py`, `CAND_P1`).** The stage-1 LightGBM scores the ~51 blocked pairs per entity, and only pairs with probability ≥ 0.001 go on to the final matcher. These surviving pairs are exactly the set written to `candidate_pairs.tsv` and the only pairs the final model scores.

| Measured on held-out entities | Before filter | After filter |
|---|---|---|
| Candidates per S1 entity | 48.7 | **4.49** |
| True matches kept | 98.92% | 98.92% |
| Final macro F0.5 | 0.98948 | 0.98948 |

On the test set the filter keeps **7.84M pairs (4.52 per S1 entity)** out of 88.0M blocked pairs.

- **Blocking keys used:** TF-IDF over name char-3grams, address words, and name×address cross tokens, with forward and reverse top-k; then a model-based filter.
- **Candidate pairs generated:** test 7.84M after the cascade (4.52 per S1 entity); 88.0M before it.
- **How you ensured true matches were not lost:**
  - Recall was measured against ground truth after every change. The blocker retrieves **98.64% (India) and 99.11% (US)** of true pairs, and the cascade filter preserved all true matches that were present in the blocked candidate set on the held-out evaluation.
  - The blocking misses (≈1–1.4%) are dominated by records with an empty address and a generic name; these were not recoverable at the blocking stage.
  - For France, which has no labels, we used a label-free proxy: pairs sharing an exact core name unique to one S1 entity plus the same house number. The proxy is 98–99.8% precise on train, and the blocker retrieves 99.89% of these proxy pairs in France, on par with India (99.91%) and the US (100%).

---

## 4. Matching Model

**Features used (~165):**
- *Name:* rapidfuzz ratio / token-sort / token-set / partial ratio and Jaro–Winkler on the core name, the full name and the space-free name; alias- and domain-aware variants; token counts and overlaps; IDF-weighted word and char-3gram TF-IDF cosines; legal-form agreement. "Novelty" is the share of name tokens never used by any S1 name, which flags invented rebrand names and distractor-only vocabulary.
- *Address:* the same string metrics; word and char-3gram TF-IDF cosines; house-number agreement; number-set intersections.
- *Structured differences (`s04b`):*
  - token-by-token name alignment: exact / close / missing counts on each side, closeness of the worst close pair, whether the close pair is phonetically equal, whether the change sits at the word start or end, the IDF of the most distinctive missing word, and a "one word substituted" flag;
  - typo-tolerant legal-form indicators per side (prvate→private);
  - number-level address diffs: S1 numbers missing from the record, extra numbers, string and numeric closeness of changed numbers, and whether each side's first number appears on the other side;
  - address word coverage.
- *Edit types (`s04e`):* for the closest non-identical word pairs, counts of substitutions vs. insertions/deletions/transpositions and anagram flags (true-match typos scramble letters; distractor mutations substitute them at the same length); for house numbers, whether a changed number is a truncation or subsequence of the original (noise) or a same-length digit change (distractor).
- *Learned token log-odds (`s04c`):* for tokens present on only one side of a pair, the log-odds of the pair being a match, learned only from training folds reserved for this purpose. Filler words score positive; words substituted into distractors ("pvt", "group", "holdings") score negative. For a country whose vocabulary the tables do not cover (France: 47% coverage), the tables are instead learned from the pipeline's own stage-1 scores as soft labels, and the pairs are rescored with them. The model is trained with the token features randomly masked for 30% of entities so it also has a fallback.
- *Namesake counts (`s04d`):* how many S1 entities and S2/S3 records share each side's exact core name, computed from unlabeled source files.
- *Blocking and competition context:* retriever scores; each pair's rank and gap among the S1 entity's candidates, and among all S1 entities claiming the same record.
- *Stage-2 graph context (`stage2.py`):* from stage-1 probabilities p1, the rank / max / sum of p1 within the S1 entity, the rank / sum / max of p1 over S1 entities claiming the same record, and the gap to the best competing claim.
- *Cluster consensus (`stage2.py`):* for each record, the probability-weighted share of the entity's other confidently attached records (plus the S1 record) that agree with it on house number, number set, name and legal form.

**Model type:**
- **LightGBM (MIT licence)**, binary log-loss, 255 leaves. Stage 1 is cross-fitted: two models are trained on disjoint entity folds (40% of entities each, easy negatives subsampled with inverse-probability weights), and each scores the other half, so stage 2 trains on honest out-of-fold probabilities. At test time the two stage-1 models are averaged.
- **Two cross-encoder transformers** re-judge the pairs LightGBM is unsure about (probability between 0.003 and 0.997; about 2% of candidate pairs). Each reads both records' "name | address" text together and outputs a match probability:
  - `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters), fine-tuned for one epoch on 900K labelled uncertain training pairs;
  - `xlm-roberta-base` (MIT, 278M parameters), fine-tuned for one epoch on 420K of the same pairs.

  For India/US, a small LightGBM **stacker** (`s09_stacker.py`) combines the stage-2 probability, both transformer scores and a few pair features (namesake counts, house-number and edit-type flags, name/address similarity) to decide per pair how much to trust each transformer; it is trained on the tuning folds only. For France, the stage-2 probability is blended in logit space with the average of both transformers at 50/50. The weights were chosen on tuning folds (India/US) and on a cross-country stress test (France, see §5), and verified on report folds. Both models are multilingual, so they read French directly, which LightGBM's statistics could not. Training ran on a single consumer GPU (RTX 4050/4060) in about an hour per model. All models are far below 8B parameters; no external data or lookups are used; pretrained weights are only the public base checkpoints.

**Threshold selection method:** there is no global threshold.
1. *One-to-one:* each record is kept only for the S1 entity with the highest probability (within a 0.05 margin).
2. *Expected-F0.5 set selection:* per entity, F0.5 = 1.25·TP / (|predicted| + 0.25·|true|). For each prefix of the probability-sorted candidates we estimate E[F] and compare it with predicting nothing, which scores 1 exactly when no candidate is a match (probability ∏(1−p)).

The parameters (margin, minimum probability, empty-set bias) are tuned on dedicated folds. For the unseen country a stronger empty-set bias (3.0) is used, as supported by the stress test.

---

## 5. Results & Error Analysis

**Validation protocol:** S1 entities are hashed into 20 folds, with separate folds for stage-1 training, stage-2 training, early stopping, token statistics, decision tuning, and a final **report** set never used by any fitting or tuning step (221,701 entities).

| Configuration (report folds) | Macro F0.5 |
|---|---|
| Stage 1, generic similarity features, threshold 0.7 + one-to-one (India) | 0.9765 |
| + structured difference features (India, stage 1) | 0.9858 |
| + stage-2 graph context + expected-F0.5 decision (all) | 0.9880 |
| + learned token log-odds with masking fallback (all) | 0.9883 |
| + namesake counts (all) | 0.9885 |
| + cluster-consensus features (all) | 0.9888 |
| + stage 1 trained on 2.7× more entities with weighted subsampling (all) | 0.9894 |
| + edit-type features (all) | 0.9895 |
| + cascade filter, 4.5 candidates per entity (all) | 0.9895 (unchanged) |
| + MiniLM cross-encoder blend (all) | 0.9903 |
| + XLM-R cross-encoder blend | 0.9903 |
| + learned stacker over both cross-encoders (India/US) — **final** | **0.9907** |

Final model by segment: India 0.9908, US 0.9905 (overall 0.99066).

**Leaderboard:**

| Submission | Score |
|---|---|
| Two-stage LightGBM + cluster consensus + larger stage-1 training set | 0.9837 |
| + edit features, French region normalisation, MiniLM blend | 0.9849 |
| + XLM-R blend, learned stacker, cascade candidate set | 0.9858 |
| + France blend weight tuned (60% LightGBM) — **final** | **0.985906** |

- **F_0.5 Score (macro):** 0.9907 on held-out entities.
- **Common false positives (wrong merges):** near-duplicate distractors with a single substituted name word at the same address, or with a changed house number and an otherwise identical name. Pair precision is ~99.75%.
- **Common false negatives (missed matches):**
  - Empty-address records whose name is shared by several S1 entities, where the available fields provide insufficient evidence to reliably distinguish between multiple candidate entities. These are about 60% of the model's misses; neither formatting nor cluster evidence separates them consistently, so they are correctly left unmatched under F0.5.
  - Blocking misses: generic names with empty addresses, invented rebrand names with fragmentary addresses, and acronym-style names ("upmedia" for "Urban Media"). These are about a quarter of the remaining loss.
  - Records with addresses whose house number carries noise, which are hard to separate from number-changed distractors.

**Generalisation to an unseen country.** With no French labels, we built a stress test: models trained on the US only, scored on India, with token features blanked.
- Stage-1 LightGBM alone: 0.940.
- + MiniLM blend: 0.947.
- + MiniLM and XLM-R averaged, weight tuned: 0.950.

The main cause of the gap is that the meaning of a house-number difference is country-specific (noise in India, a different business in the US). Leaderboard deltas indicate France scores about 0.95–0.96, against ~0.99 for India/US, and it remains the main source of lost score. The transformers are the only component that measurably improved it.

---

## 6. Conclusion

The largest gains came from understanding how the data was generated. The first was exploiting the one-to-one structure: reverse blocking, competition features, stage-2 graph context, cluster consensus and the one-to-one decision. The second was describing *what kind* of edit separates two records, not only how similar they are. The third was letting multilingual transformers read the uncertain pairs, which added information no hand-built feature captured.

A model-based cascade reduced the candidate set to 4.5 records per Source 1 entity while preserving all true matches that were present in the blocked candidate set on the held-out evaluation. Blocking and filtering scale with the number of records rather than pairs, which suits a billion-record setting. The remaining loss is dominated by genuinely ambiguous records, the blocking ceiling, and the unseen country.

---

## Appendix

### A. Code Artefacts

The code is in `code/business_entity_resolution/`. Run `./run_all.sh`; details are in that folder's README.

| Script | Role |
|---|---|
| `s01_mine_translit.py` | Indic→Latin dictionaries from training pairs |
| `s02_prepare.py` | normalisation of all six files (`textnorm.py`) |
| `s03_block.py` | candidate generation, stage A (blocking) |
| `s04_features.py` | similarity, blocking-score and competition features |
| `s04b_diff_features.py` | structured difference features |
| `s04c_token_features.py`, `tokstats.py` | learned token log-odds, coverage rule, self-trained tables for uncovered vocabularies |
| `s04d_name_stats.py` | namesake counts |
| `s04e_edit_features.py` | edit-type features |
| `s05_train.py`, `stage2.py` | stage-1 cross-fit, out-of-fold scores, stage 2 with graph and cluster features, decision tuning, held-out report, cross-country test |
| `s06_predict.py`, `decide.py` | cascade filter (candidate generation, stage B), stage-2 scoring, writes `candidate_pairs.tsv` |
| `s07_export_rerank.py` | exports the uncertain pairs as text for the cross-encoders |
| `rerank/rerank.py`, `rerank/rerank2.py` | cross-encoder fine-tuning and scoring (GPU) |
| `s08_blend.py`, `s08b_blend2.py` | blend cross-encoder scores, verify on held-out and stress test, write `matching_results.tsv` |
| `s09_stacker.py` | learned stacker for India/US over stage 2 and both cross-encoders |
| `metrics.py` | macro F0.5 exactly as defined by the challenge |

### B. Reproducibility

The pipeline is self-contained inside `code/business_entity_resolution/`. All paths are configurable via environment variables; no absolute paths or user-specific directories are required. Cross-encoder output directories default to `$ER_WORK/rerank_out/minilm` and `$ER_WORK/rerank_out/xlmr`. Detailed reproduction instructions including the GPU steps are in `code/business_entity_resolution/README.md`. The two fine-tuned transformer cross-encoders are trained from public Hugging Face checkpoints during the reproduction run; they are not bundled as binary weights.

Runtime: approximately 6 hours end-to-end on an Apple M4 (10 cores, 16 GB) for the LightGBM pipeline, plus approximately 2 hours on one RTX 4050/4060 for the two cross-encoders.

### B. Additional Results

- Blocking recall on train: India 98.64%, US 99.11%. France label-free proxy: 99.89%.
- Candidate set on test: 4.52 per S1 entity after the cascade (50.8 after blocking alone).
- Held-out loss decomposition (LightGBM only): 50% model false negatives (about half genuinely ambiguous), 25% blocking misses, 19% false positives.
