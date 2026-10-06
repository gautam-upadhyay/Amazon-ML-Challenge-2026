"""Learned blend (stacker) for India/US: a small LightGBM decides how much to trust each
cross-encoder per pair, using the stage-2 probability, both transformer scores and a few
pair features. Trained on tuning folds 7/17, verified on report folds 8/18.

usage: python s09_stacker.py            # evaluate
       python s09_stacker.py --write <base_dir> <out_dir>   # rescore India/US test rows of base
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
# pyrefly: ignore [missing-import]
import polars as pl

import decide
import s05_train as S
from config import WORK_DIR, feat_path, norm_path
from metrics import macro_f05

import os as _os
# Paths to the MiniLM (M1) and XLM-R (M2) cross-encoder output directories.
# Defaults point to subdirectories of WORK_DIR so everything stays inside the
# submission package when the pipeline is run with the standard layout.
# Override with env vars RERANK_MINILM_DIR and RERANK_XLMR_DIR if needed.
from config import WORK_DIR as _WORK_DIR
M1 = _os.environ.get("RERANK_MINILM_DIR", str(_WORK_DIR / "rerank_out" / "minilm"))
M2 = _os.environ.get("RERANK_XLMR_DIR",   str(_WORK_DIR / "rerank_out" / "xlmr"))
PF = ["ns_r_in_s1", "ns_q_in_s1", "house_state", "hn_samelen_2p", "hn_samelen_1", "ed_samelen_sub", "subst",
      "n_tset", "a_tset", "legal_eq"]
FEATS = ["lp", "l1", "l2", "has1", "has2"] + PF + ["us"]


def lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def build(P, ce1, ce2, split):
    """P: s1_id, r_id, p (+country). Returns frame with FEATS for pairs where any transformer scored."""
    P = (P.join(ce1.rename({"ce": "c1"}), on=["s1_id", "r_id"], how="left")
          .join(ce2.rename({"ce": "c2"}), on=["s1_id", "r_id"], how="left")
          .filter(pl.col("c1").is_not_null() | pl.col("c2").is_not_null()))
    parts = []
    for c in ("India", "US"):
        Pc = P.filter(pl.col("country") == c)
        f = pl.scan_parquet(feat_path(split, c)).select(["s1_id", "r_id"] + PF).join(Pc.lazy().select("s1_id", "r_id"), on=["s1_id", "r_id"]).collect()
        parts.append(Pc.join(f, on=["s1_id", "r_id"], how="left"))
    P = pl.concat(parts)
    return P.with_columns(lp=lg("p"), l1=lg("c1"), l2=lg("c2"), has1=pl.col("c1").is_not_null().cast(pl.Int8),
                          has2=pl.col("c2").is_not_null().cast(pl.Int8), us=(pl.col("country") == "US").cast(pl.Int8))


def X(D):
    return D.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()


def f05(P, ids, cfg, col):
    sel = decide.select(decide.one_to_one(P, col=col, margin=cfg["margin"]), col=col, miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    return macro_f05(sel, S.truth(), ids)[0]


def train():
    ev = pl.read_parquet(WORK_DIR / "eval_pred.parquet").select("s1_id", "r_id", "fold", "p")
    ctry = pl.read_parquet(norm_path("train", "1"), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    ev = ev.join(ctry, on="s1_id")
    ce1, ce2 = pl.read_parquet(f"{M1}/ce_heldout.parquet"), pl.read_parquet(f"{M2}/ce_heldout.parquet")
    D = build(ev, ce1, ce2, "train")
    D = D.join(S.truth().with_columns(y=pl.lit(1)), on=["s1_id", "r_id"], how="left").with_columns(pl.col("y").fill_null(0))
    tr = D.filter(pl.col("fold").is_in(list(S.TUNE)))
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=S.SEED),
                  lgb.Dataset(X(tr), tr["y"].to_numpy(), feature_name=FEATS), 300)
    return m, ev, D


def apply(m, P, D):
    """Replace p by the stacker's probability on pairs in D; others keep p."""
    s = D.select("s1_id", "r_id").with_columns(ps=pl.Series(m.predict(X(D))))
    return P.join(s, on=["s1_id", "r_id"], how="left").with_columns(pb=pl.coalesce("ps", "p")).drop("ps")


def evaluate():
    m, ev, D = train()
    cfg = json.load(open(WORK_DIR / "decision.json"))
    rid = S.s1_ids_for(S.COUNTRIES, S.REPORT)
    rep = ev.filter(pl.col("fold").is_in(list(S.REPORT)))
    from s08_blend import blend
    from s08b_blend2 import rd
    v8 = blend(rep, pl.read_parquet(f"{M2}/ce_heldout.parquet").select("s1_id", "r_id", "ce"), 0.7)
    best_cfg = {"margin": 0.05, "miss": 0.1, "none_bias": 1.5, "min_p": 0.05}
    a = f05(v8, rid, best_cfg, "pb")
    st = apply(m, rep, D.filter(pl.col("fold").is_in(list(S.REPORT))))
    b = f05(st, rid, best_cfg, "pb")
    print(f"REPORT: v8c India/US (XLM-R @0.7) {a:.5f} | stacker {b:.5f} ({b - a:+.5f})")
    for c in S.COUNTRIES:
        ids = S.s1_ids_for([c], S.REPORT)
        print(f"  {c}: {f05(v8, ids, best_cfg, 'pb'):.5f} -> {f05(st, ids, best_cfg, 'pb'):.5f}")
    m.save_model(str(WORK_DIR / "lgb_stacker.txt"))
    return b > a


def write(base_dir, out_dir, nb=1.5, mp=0.05, k=1.0):
    m = lgb.Booster(model_file=str(WORK_DIR / "lgb_stacker.txt"))
    P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    ctry = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    P = P.join(ctry, on="s1_id").filter(pl.col("country") != "France")
    ce1 = pl.read_parquet(f"{M1}/ce_test.parquet").select("s1_id", "r_id", "ce")
    ce2 = pl.read_parquet(f"{M2}/ce_testiu.parquet").select("s1_id", "r_id", "ce")
    D = build(P, ce1, ce2, "test")
    Pb = apply(m, P, D)
    if k != 1.0:  # calibration shift: odds * k (k < 1 = more cautious)
        Pb = Pb.with_columns(pb=pl.col("pb") * k / (1 - pl.col("pb") + pl.col("pb") * k))
    cfg = {"margin": 0.05, "miss": 0.1, "none_bias": nb, "min_p": mp}
    sel = decide.select(decide.one_to_one(Pb, col="pb", margin=cfg["margin"]), col="pb", miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    lists = sel.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias("new"))
    iu = P.select("s1_id").unique()["s1_id"]
    base = pl.read_csv(os.path.join(base_dir, "matching_results.tsv"), separator="\t", quote_char=None, infer_schema=False).fill_null("")
    iu_ids = ctry.filter(pl.col("country") != "France")["s1_id"]
    out = (base.join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
               .with_columns(pl.when(pl.col("source1_entity_id").is_in(iu_ids.implode()))
                               .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                               .alias("matched_entity_ids")).drop("new"))
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    print(f"India/US: {sel.height} matches (stacker) -> {out_dir}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        i = sys.argv.index("--write")
        extra = [float(x) for x in sys.argv[i + 3:i + 6]]
        write(sys.argv[i + 1], sys.argv[i + 2], *extra)
    else:
        evaluate()
