"""Build a submission variant: rescore one country with the self-training path off
(model and default decision rule unchanged) and splice it into a base submission.

usage: python make_variant.py <base_matching.tsv> <country> <out_dir>
"""
import json
import os
import sys

import polars as pl

import decide
from config import WORK_DIR, norm_path
from s06_predict import score_country


def main(base, country, out_dir):
    with open(WORK_DIR / "decision.json") as f:
        cfg = json.load(f)
    P, _ = score_country(country, selftrain=False)
    sel = decide.select(decide.one_to_one(P, margin=cfg["margin"]), miss=cfg["miss"],
                        none_bias=cfg["none_bias"], min_p=cfg["min_p"])
    lists = sel.group_by("s1_id").agg(pl.col("r_id").unique().sort().str.join(",").alias("new"))
    ids = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).filter(pl.col("country") == country)
    m = pl.read_csv(base, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    m = (m.join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
          .with_columns(pl.when(pl.col("source1_entity_id").is_in(ids["entity_id"].implode()))
                          .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                          .alias("matched_entity_ids")).drop("new"))
    os.makedirs(out_dir, exist_ok=True)
    m.write_csv(os.path.join(out_dir, "matching_results.tsv"), separator="\t", quote_style="never")
    print(f"{country}: {sel.height} matches spliced into {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
