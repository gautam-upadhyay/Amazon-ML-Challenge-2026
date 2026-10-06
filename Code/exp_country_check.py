"""Label-free comparison of a country's predictions across submission variants:
matches per entity, empty rate, and how often predicted matches have address words
of the S1 record missing (aw1_none >= 1) or a mismatched house number.

usage: python exp_country_check.py <country> <matching.tsv> [<matching.tsv> ...]
"""
import sys

import polars as pl

from config import feat_path, norm_path


def pairs(path):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).drop_nulls("matched_entity_ids")
    return (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"}))


def main(country, files):
    ids = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"]).filter(pl.col("country") == country)
    F = pl.read_parquet(feat_path("test", country), columns=["s1_id", "r_id", "aw1_none", "house_state"])
    for f in files:
        P = pairs(f).join(ids.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
        J = P.join(F, on=["s1_id", "r_id"], how="left")
        n_ent = P["s1_id"].n_unique()
        print(f"{f}\n  matches/entity {P.height / ids.height:.3f} | empty rate {1 - n_ent / ids.height:.4f} | "
              f"with S1 address words missing {(J['aw1_none'] >= 1).mean():.3f} | house number mismatch {(J['house_state'] == 3).mean():.3f} "
              f"| unmatched features {J['aw1_none'].is_null().mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
