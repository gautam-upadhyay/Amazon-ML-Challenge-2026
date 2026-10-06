"""Second stage-2 model with XGBoost on the same features; blended with LightGBM's p on held-out."""
import json
import time

import numpy as np
import polars as pl
import xgboost as xgb

import decide
import s05_train as S
from config import N_JOBS, WORK_DIR
from metrics import macro_f05

extra = {c: WORK_DIR / f"oof_train_{c}.parquet" for c in S.COUNTRIES}
Ftr = S.load(S.COUNTRIES, S.S2_TR, extra, sub="s2")
Fes = S.load(S.COUNTRIES, S.S2_ES, extra, sub="s2")
feats = S.feature_cols(Ftr) + ["p1"]
t = time.time()
dtr = xgb.DMatrix(S.to_x(Ftr, feats), label=Ftr["label"].to_numpy(), weight=Ftr["w"].to_numpy(), feature_names=feats)
des = xgb.DMatrix(S.to_x(Fes, feats), label=Fes["label"].to_numpy(), weight=Fes["w"].to_numpy(), feature_names=feats)
del Ftr, Fes
params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=8, eta=0.08,
              subsample=0.7, colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0, nthread=N_JOBS, seed=S.SEED)
m = xgb.train(params, dtr, 3000, evals=[(des, "es")], early_stopping_rounds=100, verbose_eval=250)
print(f"xgb trained {m.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)
m.save_model(str(WORK_DIR / "xgb_stage2.json"))
del dtr, des

ev = S.load(S.COUNTRIES, S.TUNE + S.REPORT, extra)
px = m.predict(xgb.DMatrix(S.to_x(ev, feats), feature_names=feats), iteration_range=(0, m.best_iteration + 1))
lg = pl.read_parquet(WORK_DIR / "eval_pred.parquet").select("s1_id", "r_id", "p")
E = ev.select("s1_id", "r_id", "fold").with_columns(px=pl.Series(px)).join(lg, on=["s1_id", "r_id"])
cfg = json.load(open(WORK_DIR / "decision.json"))
tid, rid = S.s1_ids_for(S.COUNTRIES, S.TUNE), S.s1_ids_for(S.COUNTRIES, S.REPORT)
for a in (0.0, 0.3, 0.5, 0.7, 1.0):
    X = E.with_columns(pb=a * pl.col("px") + (1 - a) * pl.col("p"))
    r = {}
    for nm, folds, ids in (("tune", S.TUNE, tid), ("report", S.REPORT, rid)):
        P = X.filter(pl.col("fold").is_in(list(folds)))
        sel = decide.select(decide.one_to_one(P, col="pb", margin=cfg["margin"]), col="pb", miss=cfg["miss"],
                            none_bias=cfg["none_bias"], min_p=cfg["min_p"])
        r[nm] = round(macro_f05(sel, S.truth(), ids)[0], 5)
    print(f"weight on XGBoost {a}: {r}", flush=True)
