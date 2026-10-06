"""Blend two cross-encoders (MiniLM run + XLM-R run) into the LightGBM scores.

Variants per pair: minilm, xlmr, ens (mean of the two where both exist, else whichever
exists). India/US: variant and weight chosen on tuning folds, confirmed on report folds;
the India/US test side only changes if XLM-R test scores for India/US exist.
France: variant and weight chosen on the stress set (US-only models scored on India),
used only if it beats the current v7 France choice (MiniLM, weight 0.7) there.

usage: python s08b_blend2.py <minilm_out_dir> <xlmr_out_dir> [--write]
"""
import itertools
import json
import os
import sys

import polars as pl

import decide
import s05_train as S
from config import OUTPUT_DIR, WORK_DIR, norm_path
from metrics import macro_f05
from s06_predict import LOWCOV_CFG, write_lists
from s08_blend import blend

WEIGHTS = (0.8, 0.7, 0.6, 0.5, 0.4)
VARIANTS = ("minilm", "xlmr", "ens")


def rd(d, n):
    f = os.path.join(d, f"ce_{n}.parquet")
    return pl.read_parquet(f).select("s1_id", "r_id", "ce") if os.path.exists(f) else None


def combine(a, b, variant):
    if variant == "minilm" or b is None:
        return a
    if variant == "xlmr" or a is None:
        return b
    return (pl.concat([a, b]).group_by("s1_id", "r_id").agg(pl.col("ce").mean()))


def f05(P, ids, cfg):
    sel = decide.select(decide.one_to_one(P, col="pb", margin=cfg["margin"]), col="pb", miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    return macro_f05(sel, S.truth(), ids)[0]


def tune_cfg(P, ids):
    best = (-1, None)
    for margin, miss, nb, mp in itertools.product((0.0, 0.05), (0.0, 0.1), (1.0, 1.2, 1.5), (0.0, 0.05, 0.15)):
        c = dict(margin=margin, miss=miss, none_bias=nb, min_p=mp)
        f = f05(P, ids, c)
        if f > best[0]:
            best = (f, c)
    return best[1]


def main(d1, d2, write):
    cfg = json.load(open(WORK_DIR / "decision.json"))
    ev = pl.read_parquet(WORK_DIR / "eval_pred.parquet").select("s1_id", "r_id", "fold", "p")
    tid, rid = S.s1_ids_for(S.COUNTRIES, S.TUNE), S.s1_ids_for(S.COUNTRIES, S.REPORT)
    tune, rep = ev.filter(pl.col("fold").is_in(list(S.TUNE))), ev.filter(pl.col("fold").is_in(list(S.REPORT)))
    h1, h2 = rd(d1, "heldout"), rd(d2, "heldout")

    # India/US: pick variant + weight on tuning folds
    res = {}
    for v, w in itertools.product(VARIANTS if h2 is not None else ("minilm",), WEIGHTS):
        res[(v, w)] = f05(blend(tune, combine(h1, h2, v), w), tid, cfg)
    base_key = max((k for k in res if k[0] == "minilm"), key=lambda k: (round(res[k], 5), k[1]))
    best_key = max(res, key=lambda k: (round(res[k], 5), k[0] == "minilm", k[1]))
    print("tuning folds:", {f"{k[0]}@{k[1]}": round(v, 5) for k, v in res.items()})
    rep_scores = {}
    for key in {base_key, best_key}:
        c = tune_cfg(blend(tune, combine(h1, h2, key[0]), key[1]), tid)
        rep_scores[key] = (f05(blend(rep, combine(h1, h2, key[0]), key[1]), rid, c), c)
        print(f"REPORT {key[0]}@{key[1]}: {rep_scores[key][0]:.5f}  cfg {c}")
    iu_key = best_key if rep_scores[best_key][0] > rep_scores[base_key][0] else base_key

    # France: pick variant + weight on the stress set
    xs = pl.read_parquet(WORK_DIR / "xcountry_US_to_India.parquet").select("s1_id", "r_id", "p")
    sid = S.s1_ids_for(["India"], S.REPORT)
    s1_, s2_ = rd(d1, "stress"), rd(d2, "stress")
    sres = {}
    for v, w in itertools.product(VARIANTS if s2_ is not None else ("minilm",), WEIGHTS):
        sres[(v, w)] = f05(blend(xs, combine(s1_, s2_, v), w), sid, cfg)
    print("stress:", {f"{k[0]}@{k[1]}": round(v, 5) for k, v in sres.items()})
    fr_key = max(sres, key=lambda k: (round(sres[k], 5), k[0] == "minilm", k[1]))
    v7_fr = sres[("minilm", 0.7)]
    if sres[fr_key] <= v7_fr + 0.0003:
        fr_key = ("minilm", 0.7)
    print(f"CHOICE India/US: {iu_key[0]}@{iu_key[1]} (report {rep_scores[iu_key][0]:.5f} vs MiniLM {rep_scores[base_key][0]:.5f})")
    print(f"CHOICE France:   {fr_key[0]}@{fr_key[1]} (stress {sres[fr_key]:.5f} vs v7 France {v7_fr:.5f})")
    if not write:
        return

    P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    t1 = rd(d1, "test")
    t2_fr, t2_iu = rd(d2, "testfr"), rd(d2, "testiu")
    s1 = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"])
    ctry = s1.rename({"entity_id": "s1_id"})
    preds = []
    for country in sorted(s1["country"].unique().to_list()):
        Pc = P.join(ctry.filter(pl.col("country") == country), on="s1_id").drop("country")
        if country in S.COUNTRIES:
            v, w = iu_key if t2_iu is not None else base_key
            ce = combine(t1, t2_iu, v)
            c = rep_scores[(v, w)][1] if (v, w) in rep_scores else rep_scores[base_key][1]
        else:
            v, w = fr_key
            ce = combine(t1, t2_fr, v)
            c = LOWCOV_CFG
        Pc = blend(Pc, ce, w)
        preds.append(decide.select(decide.one_to_one(Pc, col="pb", margin=c["margin"]), col="pb", miss=c["miss"],
                                   none_bias=c["none_bias"], min_p=c["min_p"]))
        print(f"{country}: {preds[-1].height} matches ({v}@{w})")
    write_lists(pl.concat(preds), s1["entity_id"], "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    print("wrote output/matching_results.tsv")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--write"]
    main(a[0], a[1], "--write" in sys.argv)
