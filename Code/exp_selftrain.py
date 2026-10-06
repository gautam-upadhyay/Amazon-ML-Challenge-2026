"""Can token log-odds tables be learned WITHOUT labels for an unseen country?

Train stage 1 on a source country (its token tables come from its labels), score the
target country with token features blanked, learn target token tables from those
scores as soft labels, then rescore the target with the self-trained tables. Tables
are learned on target folds 9/15/19 and evaluated on report folds 8/18.
"""
import sys

import numpy as np
import polars as pl

import decide
import s05_train as S
import tokstats as K
from config import N_JOBS, feat_path
from metrics import macro_f05


def soft_tables(P, s1, r):
    P = P.join(s1, on="s1_id").join(r, on="r_id").with_row_index("_i")
    tables = {}
    for side in K.SIDES:
        e = K.one_sided(P.rename({"p": "label"}), side)
        t = e.group_by("t").agg(pos=pl.col("label").sum(), n=pl.len()).filter(pl.col("n") >= K.MIN_COUNT)
        prior = t["pos"].sum() / t["n"].sum()
        tables[side] = t.with_columns(
            lo=(np.log((pl.col("pos") + 1) / (pl.col("n") - pl.col("pos") + 1)) - np.log(prior / (1 - prior))).cast(pl.Float32)
        ).select("t", "lo")
    return tables


def score(m, feats, F):
    return F.select("s1_id", "r_id").with_columns(p=m.predict(S.to_x(F, feats), num_threads=N_JOBS))


def with_tk(F, tables, s1, r):
    J = F.select("s1_id", "r_id").join(s1, on="s1_id", how="left").join(r, on="r_id", how="left")
    return pl.concat([F.drop([c for c in F.columns if c.startswith("tk_")]), K.token_features(J, tables)], how="horizontal")


def blank_tk(F):
    return F.with_columns([pl.lit(None, pl.Float32).alias(c) for c in F.columns if c.startswith("tk_")])


def evaluate(P, ids, tag):
    sel = decide.select(decide.one_to_one(P))
    print(f"{tag}: macro F0.5 {macro_f05(sel, S.truth(), ids)[0]:.5f}", flush=True)


def main(src, tgt, rounds=2):
    Ftr, Fes = S.load([src], S.A_TR), S.load([src], S.A_ES)
    feats = S.feature_cols(Ftr)
    m = S.fit(Ftr, Fes, feats)
    del Ftr, Fes
    s1, r = K.token_frames("train", tgt)
    stat = S.load([tgt], K.STAT_FOLDS)
    rep = S.load([tgt], S.REPORT)
    ids = S.s1_ids_for([tgt], S.REPORT)
    evaluate(score(m, feats, blank_tk(rep)), ids, f"{src}->{tgt} token features blanked")
    P = score(m, feats, blank_tk(stat))
    for it in range(rounds):
        tables = soft_tables(P, s1, r)
        evaluate(score(m, feats, with_tk(rep, tables, s1, r)), ids, f"{src}->{tgt} self-trained tables, round {it + 1}")
        P = score(m, feats, with_tk(stat, tables, s1, r))
    true_tables = K.load_tables()
    evaluate(score(m, feats, with_tk(rep, true_tables, s1, r)), ids, f"{src}->{tgt} label-learned tables (upper bound)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
