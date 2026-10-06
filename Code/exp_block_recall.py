"""Blocking recall experiment on a sample of S1 entities (full indices both ways)."""
import sys

import numpy as np
import polars as pl

import s03_block as B
from metrics import load_truth_pairs

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(50)
pl.Config.set_tbl_width_chars(260)


def main(country, n=20000):
    _, T = load_truth_pairs()
    s1, r = B.load("train", country)
    s1 = s1.with_row_index("qi")
    r = r.with_row_index("ri")
    samp = s1.sample(n, seed=1)
    T = (T.join(samp.select(pl.col("entity_id").alias("s1_id"), "qi"), on="s1_id")
          .join(r.select(pl.col("entity_id").alias("r_id"), "ri"), on="r_id"))
    q_rows = np.sort(samp["qi"].to_numpy()).astype(np.int64)
    r_rows = np.unique(T["ri"].to_numpy()).astype(np.int64)
    mats = B.vectorize(s1, r)
    cand = B.retrieve(mats, q_rows=q_rows, r_rows=r_rows).join(samp.select("qi"), on="qi")
    per_q = cand.height / n
    hit = cand.join(T.select("qi", "ri"), on=["qi", "ri"])
    print(f"{country}: recall {hit.height / T.height:.4f}  (approx cands/q fwd {per_q:.1f})")
    cols = [c for c in cand.columns if c not in ("qi", "ri")]
    for c in cols:
        only = hit.filter(pl.col(c).is_not_null() & pl.all_horizontal(pl.col(o).is_null() for o in cols if o != c))
        print(f"  {c}: recall alone {hit[c].is_not_null().sum() / T.height:.4f}, unique {only.height / T.height:.4f}")
    miss = T.join(hit.select("qi", "ri"), on=["qi", "ri"], how="anti")
    M = (miss.join(s1.select("qi", pl.col("n_full").alias("n1"), pl.col("a_full").alias("a1")), on="qi")
             .join(r.select("ri", pl.col("n_full").alias("n2"), pl.col("a_full").alias("a2")), on="ri"))
    print("missed", M.height, "r addr empty:", (M["a2"] == "").mean())
    print(M.sample(min(25, M.height), seed=0).select("n1", "n2", "a1", "a2"))


if __name__ == "__main__":
    main(sys.argv[1])
