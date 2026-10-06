"""Two-stage LightGBM training with cross-fitting, plus decision tuning.

S1 entities are assigned to 20 folds by a hash of their id:
  stage-1 model A trains on folds 0-2 (early-stops on 3) and scores folds 10-19;
  stage-1 model B trains on folds 10-12 (early-stops on 13) and scores folds 0-9;
so every training pair gets an out-of-fold stage-1 probability p1. Stage 2 adds
context features computed from p1 over the whole candidate graph (stage2.py),
trains on folds 4,5,14 (early-stops on 6,16), decision rules are tuned on folds
7,17, and the final score is reported on the untouched folds 8,18. Folds 9,15,19
are reserved for learning the token log-odds tables (tokstats.py).

usage: python s05_train.py [stage1|oof|stage2|all|xcountry]
"""
import itertools
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import decide
from config import N_JOBS, SEED, WORK_DIR, feat_path, norm_path
from metrics import load_truth_pairs, macro_f05
from stage2 import add_cluster_features, add_stage2_features

COUNTRIES = ["India", "US"]
N_FOLDS = 20
A_TR, A_ES, B_TR, B_ES = (0, 1, 2, 4, 5, 6, 7, 8), (3,), (10, 11, 12, 14, 16, 17, 18), (13,)
S2_TR, S2_ES = (0, 1, 4, 5, 10, 11, 14), (6, 16)
TUNE, REPORT = (7, 17), (8, 18)
ID_COLS = {"s1_id", "r_id", "label", "fold", "p1", "w"}
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=N_JOBS, seed=SEED, verbose=-1)


TOKEN_MASK_SHARE = 3  # out of 10 entities train and score without token features


def mask_expr():
    return (pl.col("s1_id").hash(seed=SEED + 7) % 10) < TOKEN_MASK_SHARE


def apply_token_mask(F):
    """Blank token log-odds features for a fixed 30% of training entities so the models
    learn a fallback for vocabularies the token tables do not cover."""
    tk = [c for c in F.columns if c.startswith("tk_")]
    if not tk:
        return F
    m = mask_expr()
    return F.with_columns([pl.when(m).then(None).otherwise(pl.col(c)).alias(c) for c in tk])


def fold_expr():
    return (pl.col("s1_id").hash(seed=SEED) % N_FOLDS).cast(pl.Int8).alias("fold")


_TRUTH = None


def truth():
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = load_truth_pairs()[1]
    return _TRUTH


def label(F):
    lab = truth().select("s1_id", "r_id").with_columns(label=pl.lit(1, pl.Int8))
    return F.join(lab, on=["s1_id", "r_id"], how="left").with_columns(pl.col("label").fill_null(0))


def subsample(F, kind):
    """Keep positives and hard negatives, downsample easy negatives with weight 1/rate.
    Unbiased in expectation, so probability calibration is preserved."""
    if kind == "s1":
        hard = (pl.col("label") == 1) | (pl.col("q_rank") <= 6) | (pl.col("r_rank") <= 2)
        rate = 0.15
    else:
        hard = (pl.col("label") == 1) | (pl.col("p1") >= 0.001)
        rate = 0.10
    u = ((pl.col("s1_id") + pl.col("r_id")).hash(seed=SEED + 3) % 1000) < int(rate * 1000)
    F = F.with_columns(w=pl.when(hard).then(1.0).otherwise(1.0 / rate).cast(pl.Float32))
    return F.filter(hard | u)


def load(countries, folds, extra=None, sub=None):
    parts = []
    for c in countries:
        lf = pl.scan_parquet(feat_path("train", c)).with_columns(fold_expr()).filter(pl.col("fold").is_in(list(folds)))
        F = lf.collect()
        if extra is not None:
            ex = (pl.scan_parquet(extra[c]).with_columns(fold_expr())
                    .filter(pl.col("fold").is_in(list(folds))).drop("fold").collect())
            F = F.join(ex, on=["s1_id", "r_id"], how="left")
        F = apply_token_mask(F)
        if sub is not None:
            F = subsample(label(F), sub)
        parts.append(F)
    F = pl.concat(parts, how="diagonal_relaxed")
    return F if sub is not None else label(F)


def feature_cols(F):
    return [c for c in F.columns if c not in ID_COLS]


def to_x(F, feats):
    return F.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


def fit(Ftr, Fes, feats, rounds=4000):
    wtr = Ftr["w"].to_numpy() if "w" in Ftr.columns else None
    wes = Fes["w"].to_numpy() if "w" in Fes.columns else None
    dtr = lgb.Dataset(to_x(Ftr, feats), Ftr["label"].to_numpy(), weight=wtr, feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(to_x(Fes, feats), Fes["label"].to_numpy(), weight=wes, reference=dtr)
    t = time.time()
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[des],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    print(f"  trained {m.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
    print("  top features:", [f for f, _ in imp[:20]], flush=True)
    return m


def s1_ids_for(countries, folds):
    s = pl.read_parquet(norm_path("train", "1"), columns=["entity_id", "country"])
    s = s.filter(pl.col("country").is_in(countries)).select(pl.col("entity_id").alias("s1_id")).with_columns(fold_expr())
    return s.filter(pl.col("fold").is_in(list(folds)))["s1_id"]


def stage1():
    for name, tr, es in (("A", A_TR, A_ES), ("B", B_TR, B_ES)):
        Ftr, Fes = load(COUNTRIES, tr, sub="s1"), load(COUNTRIES, es, sub="s1")
        print(f"stage1 {name}: train {Ftr.shape} pos {Ftr['label'].mean():.4f}", flush=True)
        feats = feature_cols(Ftr)
        m = fit(Ftr, Fes, feats)
        m.save_model(str(WORK_DIR / f"lgb_s1_{name}.txt"), num_iteration=m.best_iteration)
        del Ftr, Fes


def predict_batched(path, models, batch=3_000_000, fold_route=True):
    """Stream a feature parquet in batches; returns s1_id, r_id, fold, p1."""
    boosters = {k: lgb.Booster(model_file=str(v)) for k, v in models.items()}
    feats = next(iter(boosters.values())).feature_name()
    out = []
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch):
        F = pl.from_arrow(rb).with_columns(fold_expr())
        if fold_route:
            F = apply_token_mask(F)
        X = to_x(F, feats)
        if fold_route:
            hi = F["fold"].to_numpy() >= 10
            p = np.empty(len(F), dtype=np.float64)
            if hi.any():
                p[hi] = boosters["A"].predict(X[hi], num_threads=N_JOBS)
            if (~hi).any():
                p[~hi] = boosters["B"].predict(X[~hi], num_threads=N_JOBS)
        else:
            p = np.mean([b.predict(X, num_threads=N_JOBS) for b in boosters.values()], axis=0)
        out.append(F.select("s1_id", "r_id", "fold").with_columns(p1=pl.Series(p, dtype=pl.Float32)))
    return pl.concat(out)


def oof():
    models = {"A": WORK_DIR / "lgb_s1_A.txt", "B": WORK_DIR / "lgb_s1_B.txt"}
    for c in COUNTRIES:
        t = time.time()
        P = predict_batched(feat_path("train", c), models)
        add_cluster_features(add_stage2_features(P).drop("fold"), "train", c).write_parquet(WORK_DIR / f"oof_train_{c}.parquet")
        print(f"oof {c}: {P.height} pairs in {time.time() - t:.0f}s", flush=True)


def evaluate(P, s1_ids, col="p", tag=""):
    res = {}
    P11 = decide.one_to_one(P, col=col)
    res["thr0.5"] = macro_f05(decide.threshold(P, 0.5, col), truth(), s1_ids)[0]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        res[f"thr{t}+1:1"] = macro_f05(decide.threshold(P11, t, col), truth(), s1_ids)[0]
    res["expF+1:1"] = macro_f05(decide.select(P11, col=col), truth(), s1_ids)[0]
    print(tag, json.dumps({k: round(v, 5) for k, v in res.items()}), flush=True)
    return res


def tune_decision(P, s1_ids):
    best = (-1, None)
    for margin, miss, nb, mp in itertools.product((0.0, 0.05), (0.0, 0.1, 0.3), (0.8, 1.0, 1.2, 1.5), (0.0, 0.05, 0.15)):
        sel = decide.select(decide.one_to_one(P, margin=margin), miss=miss, none_bias=nb, min_p=mp)
        f = macro_f05(sel, truth(), s1_ids)[0]
        if f > best[0]:
            best = (f, dict(margin=margin, miss=miss, none_bias=nb, min_p=mp))
    print("tuned decision:", round(best[0], 5), best[1], flush=True)
    return best[1]


def stage2():
    extra = {c: WORK_DIR / f"oof_train_{c}.parquet" for c in COUNTRIES}
    Ftr, Fes = load(COUNTRIES, S2_TR, extra, sub="s2"), load(COUNTRIES, S2_ES, extra, sub="s2")
    feats = [f for f in feature_cols(Ftr)] + ["p1"]
    print(f"stage2: train {Ftr.shape}", flush=True)
    m = fit(Ftr, Fes, feats)
    m.save_model(str(WORK_DIR / "lgb_stage2.txt"), num_iteration=m.best_iteration)
    del Ftr, Fes
    ev = load(COUNTRIES, TUNE + REPORT, extra)
    ev = ev.select("s1_id", "r_id", "fold", "p1").with_columns(p=m.predict(to_x(ev, feats), num_threads=N_JOBS))
    ev.write_parquet(WORK_DIR / "eval_pred.parquet")
    report(ev)


def report(ev):
    tune_ids, rep_ids = s1_ids_for(COUNTRIES, TUNE), s1_ids_for(COUNTRIES, REPORT)
    tune_p, rep_p = ev.filter(pl.col("fold").is_in(list(TUNE))), ev.filter(pl.col("fold").is_in(list(REPORT)))
    evaluate(rep_p, rep_ids, "p1", "stage1 REPORT:")
    evaluate(rep_p, rep_ids, "p", "stage2 REPORT:")
    cfg = tune_decision(tune_p, tune_ids)
    sel = decide.select(decide.one_to_one(rep_p, margin=cfg["margin"]), miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    f, per = macro_f05(sel, truth(), rep_ids)
    print(f"FINAL REPORT macro F0.5 = {f:.5f}", flush=True)
    for c in COUNTRIES:
        ids = s1_ids_for([c], REPORT)
        print(f"  {c}: {macro_f05(sel, truth(), ids)[0]:.5f}", flush=True)
    m = pl.DataFrame({"s1_id": rep_ids}).with_columns(masked=mask_expr())
    for flag, name in ((False, "with token features"), (True, "without token features (unseen-vocabulary proxy)")):
        ids = m.filter(pl.col("masked") == flag)["s1_id"]
        print(f"  {name}: {macro_f05(sel, truth(), ids)[0]:.5f}", flush=True)
    with open(WORK_DIR / "decision.json", "w") as fh:
        json.dump(cfg, fh)
    per.write_parquet(WORK_DIR / "report_per_entity.parquet")
    sel.write_parquet(WORK_DIR / "report_pred.parquet")


def xcountry():
    """France proxy: train stage 1 on one country, evaluate on the other with the
    token log-odds features blanked (as they are for an uncovered vocabulary)."""
    for tr, va in (("US", "India"), ("India", "US")):
        Ftr, Fes = load([tr], A_TR), load([tr], A_ES)
        feats = feature_cols(Ftr)
        m = fit(Ftr, Fes, feats)
        Fva = load([va], REPORT)
        Fva = Fva.with_columns([pl.lit(None, pl.Float32).alias(c) for c in Fva.columns if c.startswith("tk_")])
        P = Fva.select("s1_id", "r_id").with_columns(p=m.predict(to_x(Fva, feats), num_threads=N_JOBS))
        ids = s1_ids_for([va], REPORT)
        evaluate(P, ids, "p", f"stage1 trained {tr} -> {va}:")
        P.write_parquet(WORK_DIR / f"xcountry_{tr}_to_{va}.parquet")
        sel = decide.select(decide.one_to_one(P))
        T = truth().join(pl.DataFrame({"s1_id": ids}), on="s1_id")
        tp = sel.join(T, on=["s1_id", "r_id"]).height
        print(f"  predicted pairs/entity {sel.height / len(ids):.3f} vs true {T.height / len(ids):.3f}; "
              f"precision {tp / max(sel.height, 1):.4f}, recall {tp / T.height:.4f}", flush=True)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("stage1", "all"):
        stage1()
    if step in ("oof", "all"):
        oof()
    if step in ("stage2", "all"):
        stage2()
    if step == "ctx":  # add cluster-consensus features to existing out-of-fold files, then stage 2
        for c in COUNTRIES:
            P = pl.read_parquet(WORK_DIR / f"oof_train_{c}.parquet")
            P = add_cluster_features(P.drop([x for x in P.columns if x.startswith("cc_")]), "train", c)
            P.write_parquet(WORK_DIR / f"oof_train_{c}.parquet")
            print(f"cluster features {c}: {P.shape}", flush=True)
        stage2()
    if step == "report":
        report(pl.read_parquet(WORK_DIR / "eval_pred.parquet"))
    if step == "xcountry":
        xcountry()


if __name__ == "__main__":
    main()
