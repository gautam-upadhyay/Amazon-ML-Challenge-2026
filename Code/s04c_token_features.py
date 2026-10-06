"""Append learned token log-odds features (tokstats.py) to the feature parquets.

Tables are learned from train folds reserved for this purpose. If most one-sided
tokens of a country are missing from the tables (a vocabulary never seen in
training), the token features are left missing for that whole country and the model
falls back on its other features (it is trained with random masking for this case).
"""
import os
import sys
import time

import polars as pl
import pyarrow.parquet as pq

import tokstats as K
from config import WORK_DIR, feat_path, norm_path
from metrics import load_truth_pairs

MIN_COVERAGE = 0.5


def coverage(split, country, tables, sample=500_000):
    s1, r = K.token_frames(split, country)
    P = pl.scan_parquet(feat_path(split, country)).select("s1_id", "r_id").head(sample).collect()
    P = P.join(s1, on="s1_id").join(r, on="r_id").with_row_index("_i")
    known = total = 0
    for side in ("n2", "n1"):
        e = K.one_sided(P, side)
        total += e.height
        known += e.join(tables[side], on="t", how="semi").height
    return known / max(total, 1), s1, r


def run(split, country, tables, batch=2_000_000):
    t = time.time()
    src = feat_path(split, country)
    pf = pq.ParquetFile(src)
    if "tk_n2_sum" in pf.schema_arrow.names:
        print(f"{split}/{country}: token features already present")
        return
    cov, s1, r = coverage(split, country, tables)
    use = cov >= MIN_COVERAGE
    print(f"{split}/{country}: token-table coverage {cov:.3f} -> {'use' if use else 'leave missing'}", flush=True)
    dst = src.with_suffix(".tmp.parquet")
    writer = None
    for rb in pf.iter_batches(batch_size=batch):
        F = pl.from_arrow(rb)
        if use:
            J = F.select("s1_id", "r_id").join(s1, on="s1_id", how="left").join(r, on="r_id", how="left")
            D = K.token_features(J, tables)
        else:
            D = pl.DataFrame({c: pl.Series([None] * F.height, dtype=pl.Float32) for c in K.FEATS})
        out = pl.concat([F, D], how="horizontal").to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(dst, out.schema)
        writer.write_table(out)
    writer.close()
    os.replace(dst, src)
    print(f"{split}/{country}: token features done in {time.time() - t:.0f}s", flush=True)


def main():
    split = sys.argv[1]
    countries = sys.argv[2:] or sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    if not (WORK_DIR / "tok_n2.parquet").exists():
        K.build_tables(["India", "US"], load_truth_pairs()[1])
    tables = K.load_tables()
    for c in countries:
        run(split, c, tables)


if __name__ == "__main__":
    main()
