"""Full self-training for an unseen country: train on the source country, turn the
pipeline's own decisions on the target country into pseudo-labels, retrain on
source labels + target pseudo-labels, repeat. Token features are blanked on the
target throughout. Pseudo-labels come from target folds 0-2; evaluation on 8/18."""
import sys

import numpy as np
import polars as pl

import decide
import s05_train as S
from config import N_JOBS
from metrics import macro_f05

POOL = (0, 1, 2)
CONF_NEG = 0.05


def blank_tk(F):
    return F.with_columns([pl.lit(None, pl.Float32).alias(c) for c in F.columns if c.startswith("tk_")])


def fit_weighted(parts, Fes, feats):
    import lightgbm as lgb
    X = np.vstack([S.to_x(F, feats) for F, _ in parts])
    y = np.concatenate([F["label"].to_numpy() for F, _ in parts])
    w = np.concatenate([np.full(F.height, wt, dtype=np.float32) for F, wt in parts])
    d = lgb.Dataset(X, y, weight=w, feature_name=feats, free_raw_data=True)
    dv = lgb.Dataset(S.to_x(Fes, feats), Fes["label"].to_numpy(), reference=d)
    return lgb.train(S.PARAMS, d, 4000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])


def pseudo(pool, p):
    P = pool.select("s1_id", "r_id").with_columns(p=p)
    sel = decide.select(decide.one_to_one(P)).with_columns(pl.lit(1, pl.Int8).alias("pl"))
    P = P.join(sel, on=["s1_id", "r_id"], how="left").with_columns(pl.col("pl").fill_null(0))
    keep = P.select((pl.col("pl") == 1) | (pl.col("p") < CONF_NEG)).to_series()
    return pool.with_columns(label=P["pl"]).filter(keep)


def main(src, tgt, rounds=2):
    src_tr, src_es = S.load([src], S.A_TR), S.load([src], S.A_ES)
    pool = blank_tk(S.load([tgt], POOL))
    rep = blank_tk(S.load([tgt], S.REPORT))
    ids = S.s1_ids_for([tgt], S.REPORT)
    feats = S.feature_cols(src_tr)
    m = S.fit(src_tr, src_es, feats)
    for it in range(rounds + 1):
        P = rep.select("s1_id", "r_id").with_columns(p=m.predict(S.to_x(rep, feats), num_threads=N_JOBS))
        f = macro_f05(decide.select(decide.one_to_one(P)), S.truth(), ids)[0]
        print(f"{src}->{tgt} round {it}: macro F0.5 {f:.5f}", flush=True)
        if it == rounds:
            break
        pl_pool = pseudo(pool, m.predict(S.to_x(pool, feats), num_threads=N_JOBS))
        true_rate = pl_pool.join(S.truth().with_columns(t=pl.lit(1)), on=["s1_id", "r_id"], how="left")
        acc = (true_rate["label"] == true_rate["t"].fill_null(0)).mean()
        print(f"  pseudo-labels: {pl_pool.height} pairs, positives {pl_pool['label'].sum()}, agreement with truth {acc:.4f}", flush=True)
        m = fit_weighted([(src_tr, 1.0), (pl_pool, 1.0)], src_es, feats)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
