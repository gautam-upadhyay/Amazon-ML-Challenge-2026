"""Stage 3: cross-fitted stage 2, then graph features recomputed from stage-2 scores.

Stage-2 models A2 (folds 0,1,4,5; early-stops on 6) and B2 (folds 10,11,14; early-stops
on 16) each score the other half, giving honest stage-2 probabilities p2 for every
training pair. Cluster-consensus and competition features are recomputed from p2
(sharper membership estimates than p1) and stage 3 is trained on the same folds with
p2 from the model that did not see them. Tuning on folds 7,17; report on 8,18.

usage: python s05b_stage3.py [fit2|oof2|fit3|all]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import s05_train as S
from config import N_JOBS, WORK_DIR, feat_path
from stage2 import add_cluster_features, add_context

A2_TR, A2_ES, B2_TR, B2_ES = (0, 1, 4, 5), (6,), (10, 11, 14), (16,)
S3_TR, S3_ES = (0, 1, 4, 5, 10, 11, 14), (6, 16)


def oof_path(c):
    return WORK_DIR / f"oof_train_{c}.parquet"


def oof2_path(c):
    return WORK_DIR / f"oof2_train_{c}.parquet"


def fit2():
    extra = {c: oof_path(c) for c in S.COUNTRIES}
    for name, tr, es in (("A", A2_TR, A2_ES), ("B", B2_TR, B2_ES)):
        Ftr = S.subsample(S.load(S.COUNTRIES, tr, extra), "s2")
        Fes = S.subsample(S.load(S.COUNTRIES, es, extra), "s2")
        feats = S.feature_cols(Ftr) + ["p1"]
        print(f"stage2 {name}: train {Ftr.shape}", flush=True)
        m = S.fit(Ftr, Fes, feats)
        m.save_model(str(WORK_DIR / f"lgb_s2_{name}.txt"), num_iteration=m.best_iteration)
        del Ftr, Fes


def stream_p2(split, country, ctx, models, route):
    """Stream the feature parquet with the aligned stage-2 context; returns s1_id, r_id, p2."""
    bs = {k: lgb.Booster(model_file=str(WORK_DIR / f"lgb_s2_{k}.txt")) for k in models}
    feats = next(iter(bs.values())).feature_name()
    base = [f for f in feats if f not in ctx.columns]
    out, off = [], 0
    for rb in pq.ParquetFile(feat_path(split, country)).iter_batches(batch_size=3_000_000, columns=["s1_id", "r_id"] + base):
        F = pl.from_arrow(rb)
        c = ctx[off:off + F.height]
        off += F.height
        assert (c["r_id"] == F["r_id"]).all()
        F = pl.concat([F, c.drop("s1_id", "r_id")], how="horizontal")
        if split == "train":
            F = S.apply_token_mask(F)
        X = S.to_x(F, feats)
        if route:
            hi = (F.select(S.fold_expr())["fold"] >= 10).to_numpy()
            p = np.empty(F.height)
            p[hi] = bs["A"].predict(X[hi], num_threads=N_JOBS)
            p[~hi] = bs["B"].predict(X[~hi], num_threads=N_JOBS)
        else:
            p = np.mean([b.predict(X, num_threads=N_JOBS) for b in bs.values()], axis=0)
        out.append(F.select("s1_id", "r_id").with_columns(p2=pl.Series(p, dtype=pl.Float32)))
    return pl.concat(out)


def stage3_ctx(P2, split, country):
    P2 = add_context(P2, "p2", "p2_")
    return add_cluster_features(P2, split, country, col="p2", prefix="cc3_")


def oof2():
    for c in S.COUNTRIES:
        t = time.time()
        ctx = pl.read_parquet(oof_path(c))
        P2 = stream_p2("train", c, ctx, ("A", "B"), route=True)
        del ctx
        stage3_ctx(P2, "train", c).write_parquet(oof2_path(c))
        print(f"oof2 {c}: {P2.height} pairs in {time.time() - t:.0f}s", flush=True)


def load3(folds):
    parts = []
    for c in S.COUNTRIES:
        F = S.load([c], folds, {c: oof_path(c)})
        ex = (pl.scan_parquet(oof2_path(c)).with_columns(S.fold_expr())
                .filter(pl.col("fold").is_in(list(folds))).drop("fold").collect())
        parts.append(F.join(ex, on=["s1_id", "r_id"], how="left"))
    return pl.concat(parts, how="diagonal_relaxed")


def sub3(F):
    hard = (pl.col("label") == 1) | (pl.col("p2") >= 0.001)
    u = ((pl.col("s1_id") + pl.col("r_id")).hash(seed=S.SEED + 5) % 1000) < 100
    return F.with_columns(w=pl.when(hard).then(1.0).otherwise(10.0).cast(pl.Float32)).filter(hard | u)


def fit3():
    Ftr, Fes = sub3(load3(S3_TR)), sub3(load3(S3_ES))
    feats = S.feature_cols(Ftr) + ["p1"]
    feats = [f for f in feats if f != "p2"] + ["p2"]
    print(f"stage3: train {Ftr.shape}", flush=True)
    m = S.fit(Ftr, Fes, feats)
    m.save_model(str(WORK_DIR / "lgb_stage3.txt"), num_iteration=m.best_iteration)
    del Ftr, Fes
    ev = load3(S.TUNE + S.REPORT)
    ev = ev.select("s1_id", "r_id", "fold", "p1", "p2").with_columns(p=m.predict(S.to_x(ev, feats), num_threads=N_JOBS))
    ev.write_parquet(WORK_DIR / "eval_pred3.parquet")
    rep_ids = S.s1_ids_for(S.COUNTRIES, S.REPORT)
    rep = ev.filter(pl.col("fold").is_in(list(S.REPORT)))
    S.evaluate(rep, rep_ids, "p2", "cross-fitted stage2 REPORT:")
    S.evaluate(rep, rep_ids, "p", "stage3 REPORT:")
    cfg = S.tune_decision(ev.filter(pl.col("fold").is_in(list(S.TUNE))), S.s1_ids_for(S.COUNTRIES, S.TUNE))
    import decide
    from metrics import macro_f05
    sel = decide.select(decide.one_to_one(rep, margin=cfg["margin"]), miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    f, _ = macro_f05(sel, S.truth(), rep_ids)
    print(f"FINAL STAGE3 REPORT macro F0.5 = {f:.5f}", flush=True)
    for c in S.COUNTRIES:
        print(f"  {c}: {macro_f05(sel, S.truth(), S.s1_ids_for([c], S.REPORT))[0]:.5f}", flush=True)
    mk = pl.DataFrame({"s1_id": rep_ids}).with_columns(masked=S.mask_expr())
    for flag, name in ((False, "with token features"), (True, "without token features")):
        print(f"  {name}: {macro_f05(sel, S.truth(), mk.filter(pl.col('masked') == flag)['s1_id'])[0]:.5f}", flush=True)
    with open(WORK_DIR / "decision3.json", "w") as fh:
        json.dump(cfg, fh)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("fit2", "all"):
        fit2()
    if step in ("oof2", "all"):
        oof2()
    if step in ("fit3", "all"):
        fit3()


if __name__ == "__main__":
    main()
