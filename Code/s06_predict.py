"""Score test candidates, apply the decision layer, and write both submission files.

Countries whose vocabulary the token tables do not cover (the token features were left
missing by s04c) get a self-training pass: token tables are learned from the pipeline's
own stage-1 scores, the pairs are rescored with them, and a stricter decision rule is
applied (both validated on the train-one-country/score-the-other stress test).
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import decide
import tokstats as K
from config import N_JOBS, OUTPUT_DIR, WORK_DIR, feat_path, norm_path
from s05_train import to_x
from stage2 import add_cluster_features, add_stage2_features

BATCH = 3_000_000
# Cascade: stage 1 acts as a candidate filter. Only pairs with stage-1 probability >= CAND_P1
# reach the final model (stage 2 + decision); they ARE the candidate set written to
# candidate_pairs.tsv. On held-out entities this keeps 4.5 candidates per S1 entity instead
# of 48.7 with no loss of true matches or final F0.5.
CAND_P1 = 0.001
# Stress-test-validated (train one country, score the other): the empty-set bias is
# what helps; min_p above the default added nothing, so keep the safer value.
LOWCOV_CFG = {"margin": 0.0, "miss": 0.0, "none_bias": 3.0, "min_p": 0.15}


def _boosters(names):
    return [lgb.Booster(model_file=str(WORK_DIR / f"{n}.txt")) for n in names]


def _batches(country, tk):
    off = 0
    for rb in pq.ParquetFile(feat_path("test", country)).iter_batches(batch_size=BATCH):
        F = pl.from_arrow(rb)
        if tk is not None:
            F = pl.concat([F.drop([c for c in F.columns if c.startswith("tk_")]),
                           tk[off:off + F.height]], how="horizontal")
        off += F.height
        yield F


def stage1_scores(country, tk=None):
    ms = _boosters(["lgb_s1_A", "lgb_s1_B"])
    feats = ms[0].feature_name()
    out = []
    for F in _batches(country, tk):
        X = to_x(F, feats)
        p = np.mean([m.predict(X, num_threads=N_JOBS) for m in ms], axis=0)
        out.append(F.select("s1_id", "r_id").with_columns(p1=pl.Series(p, dtype=pl.Float32)))
    return pl.concat(out)


def is_lowcov(country):
    col = pl.scan_parquet(feat_path("test", country)).select("tk_n2_sum").head(200_000).collect()
    return col["tk_n2_sum"].null_count() == col.height


def self_tk(country, P1):
    s1, r = K.token_frames("test", country)
    tables = K.soft_tables(P1.rename({"p1": "p"}), s1, r)
    J = P1.select("s1_id", "r_id").join(s1, on="s1_id", how="left").join(r, on="r_id", how="left")
    return K.token_features(J, tables)


def score_country(country, selftrain=True):
    lowcov = is_lowcov(country) and selftrain
    P = stage1_scores(country)
    tk = None
    if lowcov:
        tk = self_tk(country, P)
        P = stage1_scores(country, tk=tk)
        print(f"  {country}: low token coverage -> self-trained tables and rescored", flush=True)
    keep = (P["p1"] >= CAND_P1).to_numpy()
    print(f"  {country}: cascade keeps {keep.sum():,} of {len(keep):,} blocked pairs", flush=True)
    ctx = add_cluster_features(add_stage2_features(P.filter(pl.Series(keep))), "test", country)
    m = _boosters(["lgb_stage2"])[0]
    feats = m.feature_name()
    out, off, coff = [], 0, 0
    for F in _batches(country, tk):
        k = keep[off:off + F.height]
        off += F.height
        F = F.filter(pl.Series(k))
        c = ctx[coff:coff + F.height]
        coff += F.height
        assert (c["r_id"] == F["r_id"]).all()
        F = pl.concat([F, c.drop("s1_id", "r_id")], how="horizontal")
        out.append(F.select("s1_id", "r_id").with_columns(p=m.predict(to_x(F, feats), num_threads=N_JOBS)))
    return pl.concat(out), lowcov


def write_lists(pairs, s1_ids, col, path):
    lists = pairs.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    with open(WORK_DIR / "decision.json") as f:
        cfg_default = json.load(f)
    s1 = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"])
    scored, preds = [], []
    for country in sorted(s1["country"].unique().to_list()):
        P, lowcov = score_country(country)
        cfg = LOWCOV_CFG if lowcov else cfg_default
        pred = decide.select(decide.one_to_one(P, margin=cfg["margin"]), miss=cfg["miss"],
                             none_bias=cfg["none_bias"], min_p=cfg["min_p"])
        scored.append(P)
        preds.append(pred)
        print(f"{country}: {P.height} pairs, {pred.height} matches (cfg={'lowcov' if lowcov else 'default'})", flush=True)
    P = pl.concat(scored)
    P.write_parquet(WORK_DIR / "test_scores.parquet")
    ids = s1["entity_id"]
    m = write_lists(pl.concat(preds), ids, "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    write_lists(P.select("s1_id", "r_id"), ids, "candidate_entity_ids", OUTPUT_DIR / "candidate_pairs.tsv")
    print(f"matched pairs: {pl.concat(preds).height}; entities with matches: {(m['matched_entity_ids'] != '').sum()} "
          f"of {m.height}; candidate pairs: {P.height}", flush=True)


if __name__ == "__main__":
    main()
