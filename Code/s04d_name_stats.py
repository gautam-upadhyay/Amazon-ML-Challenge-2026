"""Namesake counts, appended to the feature parquets.

An S2/S3 record whose exact core name belongs to a single S1 entity is almost always
that entity's record (even with an empty address), while a name shared by several
S1 entities is ambiguous. Counts come from the unlabeled source files of the same
split and country, so they are equally available for the test set.
"""
import os
import sys
import time

import polars as pl
import pyarrow.parquet as pq

from config import feat_path, norm_path

FEATS = ["ns_r_in_s1", "ns_q_in_s1", "ns_r_in_r", "ns_q_in_r", "ns_key_eq"]


def key_expr():
    return pl.col("n_core").str.split(" ").list.unique().list.sort().list.join(" ").alias("k")


def run(split, country, batch=3_000_000):
    t = time.time()
    src = feat_path(split, country)
    pf = pq.ParquetFile(src)
    if FEATS[0] in pf.schema_arrow.names:
        print(f"{split}/{country}: namesake features already present")
        return
    s1 = (pl.read_parquet(norm_path(split, "1"), columns=["entity_id", "country", "n_core"])
            .filter(pl.col("country") == country).select("entity_id", key_expr()))
    r = (pl.concat([pl.read_parquet(norm_path(split, s), columns=["entity_id", "country", "n_core"]) for s in ("2", "3")])
           .filter(pl.col("country") == country).select("entity_id", key_expr()))
    c1 = s1.group_by("k").len("c1")
    c2 = r.group_by("k").len("c2")
    q = s1.join(c1, on="k", how="left").join(c2, on="k", how="left").select(
        pl.col("entity_id").alias("s1_id"), pl.col("k").alias("kq"),
        pl.col("c1").alias("ns_q_in_s1"), pl.col("c2").fill_null(0).alias("ns_q_in_r"))
    rr = r.join(c1, on="k", how="left").join(c2, on="k", how="left").select(
        pl.col("entity_id").alias("r_id"), pl.col("k").alias("kr"),
        pl.col("c1").fill_null(0).alias("ns_r_in_s1"), pl.col("c2").alias("ns_r_in_r"))
    dst = src.with_suffix(".tmp.parquet")
    writer = None
    for rb in pf.iter_batches(batch_size=batch):
        F = pl.from_arrow(rb)
        J = (F.select("s1_id", "r_id").with_row_index("_i").join(q, on="s1_id", how="left")
               .join(rr, on="r_id", how="left").sort("_i"))
        D = J.select(
            *[pl.col(c).cast(pl.Float32) for c in ("ns_r_in_s1", "ns_q_in_s1", "ns_r_in_r", "ns_q_in_r")],
            ns_key_eq=(pl.col("kq") == pl.col("kr")).cast(pl.Float32))
        out = pl.concat([F, D], how="horizontal").to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(dst, out.schema)
        writer.write_table(out)
    writer.close()
    os.replace(dst, src)
    print(f"{split}/{country}: namesake features done in {time.time() - t:.0f}s", flush=True)


def main():
    split = sys.argv[1]
    countries = sys.argv[2:] or sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    for c in countries:
        run(split, c)


if __name__ == "__main__":
    main()
