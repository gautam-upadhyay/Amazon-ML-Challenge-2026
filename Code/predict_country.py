"""Rescore one country with the current models and splice it into an existing submission
(matching + candidate files, and test_scores.parquet), leaving other countries untouched.

usage: python predict_country.py <country> <base_dir> <out_dir>
  base_dir holds matching_results.tsv and candidate_pairs.tsv to splice into.
"""
import json
import os
import sys

import polars as pl

import decide
from config import WORK_DIR, norm_path
from s06_predict import LOWCOV_CFG, score_country


def splice(base_file, ids, lists, col, out_file):
    m = pl.read_csv(base_file, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    m = (m.join(lists.rename({"s1_id": "source1_entity_id", "v": "new"}), on="source1_entity_id", how="left")
          .with_columns(pl.when(pl.col("source1_entity_id").is_in(ids.implode()))
                          .then(pl.col("new").fill_null("")).otherwise(pl.col(col)).alias(col)).drop("new"))
    m.write_csv(out_file, separator="\t", quote_style="never")


def main(country, base_dir, out_dir):
    with open(WORK_DIR / "decision.json") as f:
        cfg = json.load(f)
    P, lowcov = score_country(country)
    c = LOWCOV_CFG if lowcov else cfg
    sel = decide.select(decide.one_to_one(P, margin=c["margin"]), miss=c["miss"], none_bias=c["none_bias"], min_p=c["min_p"])
    ids = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).filter(pl.col("country") == country)["entity_id"]
    agg = lambda D: D.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias("v"))
    os.makedirs(out_dir, exist_ok=True)
    splice(os.path.join(base_dir, "matching_results.tsv"), ids, agg(sel), "matched_entity_ids",
           os.path.join(out_dir, "matching_results.tsv"))
    splice(os.path.join(base_dir, "candidate_pairs.tsv"), ids, agg(P.select("s1_id", "r_id")), "candidate_entity_ids",
           os.path.join(out_dir, "candidate_pairs.tsv"))
    ts = pl.read_parquet(WORK_DIR / "test_scores.parquet").filter(~pl.col("s1_id").is_in(ids.implode()))
    pl.concat([ts, P.select(ts.columns)]).write_parquet(WORK_DIR / "test_scores.parquet")
    print(f"{country}: {P.height} candidate pairs, {sel.height} matches ({'self-trained path' if lowcov else 'default'}) -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
