"""Decompose macro-F0.5 loss on held-out entities into blocking misses, model false
positives and model false negatives, and print examples of each."""
import sys

import polars as pl

import decide
import s05_train as S
from config import cand_path, norm_path
from metrics import macro_f05

pl.Config.set_tbl_rows(30)
pl.Config.set_fmt_str_lengths(48)
pl.Config.set_tbl_width_chars(250)


def main(country, col="p1", thr=0.7, path=None):
    P = pl.read_parquet(path or S.WORK_DIR / f"oof_train_{country}.parquet", columns=["s1_id", "r_id", col]).with_columns(S.fold_expr())
    ids = S.s1_ids_for([country], S.REPORT)
    P11 = decide.one_to_one(P, col=col).filter(pl.col("fold").is_in(list(S.REPORT)))
    pred = decide.threshold(P11, thr, col)
    f, per = macro_f05(pred, S.truth(), ids)
    T = S.truth().join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    C = pl.read_parquet(cand_path("train", country), columns=["s1_id", "r_id"]).join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    fn = T.join(pred, on=["s1_id", "r_id"], how="anti")
    fn_block = fn.join(C, on=["s1_id", "r_id"], how="anti")
    fn_model = fn.join(C, on=["s1_id", "r_id"])
    fp = pred.join(T, on=["s1_id", "r_id"], how="anti")
    loss = 1 - per["f"]
    print(f"{country} F={f:.5f}  total loss={loss.sum():.0f} entities-equivalent over {len(ids)}")
    per = per.with_columns(
        has_fp=pl.col("s1_id").is_in(fp["s1_id"].implode()),
        has_fnb=pl.col("s1_id").is_in(fn_block["s1_id"].implode()),
        has_fnm=pl.col("s1_id").is_in(fn_model["s1_id"].implode()),
        loss=1 - pl.col("f"))
    print(per.group_by("has_fp", "has_fnb", "has_fnm").agg(pl.len(), pl.col("loss").sum().round(0)).sort("loss", descending=True))
    print("singletons predicted non-empty:", per.filter((pl.col("nt") == 0) & (pl.col("np") > 0)).height,
          "| entities with nothing predicted but matches exist:", per.filter((pl.col("nt") > 0) & (pl.col("np") == 0)).height)
    s1 = pl.read_parquet(norm_path("train", "1"), columns=["entity_id", "n_full", "a_full"]).rename({"entity_id": "s1_id", "n_full": "n1", "a_full": "a1"})
    r = pl.concat([pl.read_parquet(norm_path("train", s), columns=["entity_id", "n_full", "a_full"]) for s in ("2", "3")]).rename({"entity_id": "r_id", "n_full": "n2", "a_full": "a2"})
    Pc = P.select("s1_id", "r_id", col)
    for name, D in (("FALSE POSITIVES", fp), ("MODEL FALSE NEGATIVES", fn_model)):
        print(name, D.height)
        print(D.sample(min(20, D.height), seed=0).join(Pc, on=["s1_id", "r_id"], how="left").join(s1, on="s1_id").join(r, on="r_id").select(col, "n1", "n2", "a1", "a2"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
