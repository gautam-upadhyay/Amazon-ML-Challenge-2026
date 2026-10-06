"""Label-free blocking check: pairs sharing the exact sorted core name AND house
number are near-certain matches; measure how many of them the blocker retrieved."""
import sys

import polars as pl

from config import cand_path, norm_path
from metrics import load_truth_pairs


def key(df):
    return df.filter((pl.col("a_house") != "") & (pl.col("n_core") != "")).select(
        "entity_id", k=pl.col("n_core").str.split(" ").list.sort().list.join(" ") + "#" + pl.col("a_house"))


def main(split, country):
    s1 = pl.read_parquet(norm_path(split, "1"), columns=["entity_id", "country", "n_core", "a_house"]).filter(pl.col("country") == country)
    r = pl.concat([pl.read_parquet(norm_path(split, s), columns=["entity_id", "country", "n_core", "a_house"]) for s in ("2", "3")]).filter(pl.col("country") == country)
    uniq = s1.group_by("n_core").len("c").filter(pl.col("c") == 1).select("n_core")
    ks, kr = key(s1.join(uniq, on="n_core")), key(r)
    kr = kr.join(kr.group_by("k").len("kn"), on="k").filter(pl.col("kn") <= 20)
    P = ks.rename({"entity_id": "s1_id"}).join(kr.rename({"entity_id": "r_id"}), on="k").select("s1_id", "r_id")
    C = pl.read_parquet(cand_path(split, country), columns=["s1_id", "r_id"])
    rec = P.join(C, on=["s1_id", "r_id"]).height / P.height
    msg = f"{split}/{country}: proxy pairs {P.height}, per S1 {P.height / s1.height:.2f}, retrieved {rec:.4f}"
    if split == "train":
        _, T = load_truth_pairs()
        msg += f", proxy precision {P.join(T, on=['s1_id', 'r_id']).height / P.height:.4f}"
    print(msg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
