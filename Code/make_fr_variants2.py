"""France-only variants on top of a submission using the combined cross-encoder scores
(MiniLM run + XLM-R run). India/US rows are copied unchanged from the base file.

usage: python make_fr_variants2.py <base_dir> <minilm_dir> <xlmr_dir> <variant> <out_root> w [w ...]
  variant: minilm | xlmr | ens
"""
import os
import sys

import polars as pl

import decide
from config import WORK_DIR, norm_path
from s06_predict import LOWCOV_CFG
from s08_blend import blend
from s08b_blend2 import combine, rd


def main(base_dir, d1, d2, variant, out_root, weights):
    P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    ce = combine(rd(d1, "test"), rd(d2, "testfr"), variant)
    fr = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"]
    Pf = P.filter(pl.col("s1_id").is_in(fr.implode()))
    base = pl.read_csv(os.path.join(base_dir, "matching_results.tsv"), separator="\t", quote_char=None, infer_schema=False).fill_null("")
    for w in weights:
        c = LOWCOV_CFG
        sel = decide.select(decide.one_to_one(blend(Pf, ce, w), col="pb", margin=c["margin"]), col="pb",
                            miss=c["miss"], none_bias=c["none_bias"], min_p=c["min_p"])
        lists = sel.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias("new"))
        out = (base.join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
                   .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr.implode()))
                                   .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                                   .alias("matched_entity_ids")).drop("new"))
        d = os.path.join(out_root, f"fr_{variant}_w{w}")
        os.makedirs(d, exist_ok=True)
        out.write_csv(os.path.join(d, "matching_results.tsv"), separator="\t", quote_style="never")
        print(f"{d}: France matches {sel.height}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4], [float(x) for x in a[5:]])
