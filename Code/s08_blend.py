"""Blend the GPU cross-encoder scores into the LightGBM probabilities on the uncertain band.

p_blend = sigmoid(w * logit(p_lgbm) + (1 - w) * logit(p_ce)) for pairs the cross-encoder
scored; all other pairs keep p_lgbm. The weight w and decision parameters are tuned on
folds 7/17 and reported on 8/18. The stress set (US-only models scored on India) decides
whether the blend is also applied to countries unseen in training (France).

usage: python s08_blend.py <gpu_out_dir> [--write]
"""
import itertools
import json
import sys

import numpy as np
import polars as pl

import decide
import s05_train as S
from config import OUTPUT_DIR, WORK_DIR, norm_path
from metrics import macro_f05
from s06_predict import LOWCOV_CFG, write_lists

WEIGHTS = (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def logit(x):
    x = pl.col(x).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def blend(P, ce, w, col="p"):
    P = P.join(ce, on=["s1_id", "r_id"], how="left")
    z = w * logit(col) + (1 - w) * logit("ce")
    return P.with_columns(pl.when(pl.col("ce").is_null()).then(pl.col(col)).otherwise(1 / (1 + (-z).exp())).alias("pb"))


def f05(P, ids, cfg, col="pb"):
    sel = decide.select(decide.one_to_one(P, col=col, margin=cfg["margin"]), col=col, miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    return macro_f05(sel, S.truth(), ids)[0]


def main(gpu_dir, write):
    with open(WORK_DIR / "decision.json") as f:
        cfg = json.load(f)
    ev = pl.read_parquet(WORK_DIR / "eval_pred.parquet").select("s1_id", "r_id", "fold", "p")
    ce_h = pl.read_parquet(f"{gpu_dir}/ce_heldout.parquet")
    tune = ev.filter(pl.col("fold").is_in(list(S.TUNE)))
    rep = ev.filter(pl.col("fold").is_in(list(S.REPORT)))
    tid, rid = S.s1_ids_for(S.COUNTRIES, S.TUNE), S.s1_ids_for(S.COUNTRIES, S.REPORT)
    res = {w: f05(blend(tune, ce_h, w), tid, cfg) for w in WEIGHTS}
    best_w = max(res, key=lambda w: (round(res[w], 5), w))
    print("tuning folds by weight on LightGBM:", {w: round(v, 5) for w, v in res.items()})
    # re-tune the decision rule for the chosen weight
    best = (-1, cfg)
    Tb = blend(tune, ce_h, best_w)
    for margin, miss, nb, mp in itertools.product((0.0, 0.05), (0.0, 0.1), (1.0, 1.2, 1.5), (0.0, 0.05, 0.15)):
        c = dict(margin=margin, miss=miss, none_bias=nb, min_p=mp)
        f = f05(Tb, tid, c)
        if f > best[0]:
            best = (f, c)
    bcfg = best[1]
    base = f05(rep.with_columns(pb=pl.col("p")), rid, cfg)
    new = f05(blend(rep, ce_h, best_w), rid, bcfg)
    print(f"REPORT folds: LightGBM alone {base:.5f} | blended (w={best_w}) {new:.5f} | decision {bcfg}")
    for c in S.COUNTRIES:
        ids = S.s1_ids_for([c], S.REPORT)
        print(f"  {c}: {f05(rep.with_columns(pb=pl.col('p')), ids, cfg):.5f} -> {f05(blend(rep, ce_h, best_w), ids, bcfg):.5f}")

    apply_unseen = False
    try:
        xs = pl.read_parquet(WORK_DIR / "xcountry_US_to_India.parquet").select("s1_id", "r_id", "p")
        ce_s = pl.read_parquet(f"{gpu_dir}/ce_stress.parquet")
        sid = S.s1_ids_for(["India"], S.REPORT)
        s_base = f05(xs.with_columns(pb=pl.col("p")), sid, cfg)
        s_res = {w: f05(blend(xs, ce_s, w), sid, cfg) for w in WEIGHTS}
        print(f"STRESS (US-only models on India): alone {s_base:.5f} | by weight {({w: round(v, 5) for w, v in s_res.items()})}")
        apply_unseen = s_res[best_w] > s_base + 0.0005
    except FileNotFoundError:
        print("no stress scores: blend will NOT be applied to unseen countries")
    print(f"apply blend to unseen-vocabulary countries (France): {apply_unseen}")

    if not write:
        return
    if new <= base:
        print("blend does not beat LightGBM on REPORT folds: nothing written")
        return
    P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    ce_t = pl.read_parquet(f"{gpu_dir}/ce_test.parquet")
    s1 = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"])
    ctry = s1.rename({"entity_id": "s1_id"})
    preds = []
    for country in sorted(s1["country"].unique().to_list()):
        Pc = P.join(ctry.filter(pl.col("country") == country), on="s1_id").drop("country")
        unseen = country not in S.COUNTRIES
        if unseen and not apply_unseen:
            Pc = Pc.with_columns(pb=pl.col("p"))
        else:
            Pc = blend(Pc, ce_t, best_w)
        c = LOWCOV_CFG if unseen else bcfg
        preds.append(decide.select(decide.one_to_one(Pc, col="pb", margin=c["margin"]), col="pb", miss=c["miss"],
                                   none_bias=c["none_bias"], min_p=c["min_p"]))
        print(f"{country}: {preds[-1].height} matches ({'blended' if not unseen or apply_unseen else 'LightGBM only'})")
    write_lists(pl.concat(preds), s1["entity_id"], "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    print("wrote output/matching_results.tsv (candidate_pairs.tsv unchanged: same candidate set)")


if __name__ == "__main__":
    main(sys.argv[1], "--write" in sys.argv[2:])
