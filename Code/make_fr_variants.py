"""France-only variants of a submission: India/US rows copied unchanged from the base file,
France re-selected from the blended scores with a different blend weight / empty-set bias.

usage: python make_fr_variants.py <base_dir> <ce_dir> <out_root> w:nb [w:nb ...]
"""
import os
import sys

import polars as pl

import decide
from config import WORK_DIR, norm_path
from s06_predict import LOWCOV_CFG
from s08_blend import blend


def main(base_dir, ce_dir, out_root, specs):
    P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
    ce = pl.read_parquet(os.path.join(ce_dir, "ce_test.parquet")).select("s1_id", "r_id", "ce")
    fr = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"]
    Pf = P.filter(pl.col("s1_id").is_in(fr.implode()))
    base = pl.read_csv(os.path.join(base_dir, "matching_results.tsv"), separator="\t", quote_char=None, infer_schema=False).fill_null("")
    for spec in specs:
        w, nb = map(float, spec.split(":"))
        c = dict(LOWCOV_CFG, none_bias=nb)
        B = blend(Pf, ce, w)
        sel = decide.select(decide.one_to_one(B, col="pb", margin=c["margin"]), col="pb", miss=c["miss"],
                            none_bias=c["none_bias"], min_p=c["min_p"])
        lists = sel.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias("new"))
        out = (base.join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
                   .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr.implode()))
                                   .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                                   .alias("matched_entity_ids")).drop("new"))
        d = os.path.join(out_root, f"fr_w{w}_nb{nb}")
        os.makedirs(d, exist_ok=True)
        out.write_csv(os.path.join(d, "matching_results.tsv"), separator="\t", quote_style="never")
        print(f"{d}: France matches {sel.height}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
