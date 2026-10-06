"""Candidate generation: union of TF-IDF cosine top-k retrievers, run per country,
in both directions.

Representations
  X  cross features: every (name token x address token) pair. Rare even when both
     tokens are common, which is what separates "sun services, flat 701, thane" from
     hundreds of other "sun services".
  A  name char-3grams + address words (combined).
  B  address words only (rebrands whose name is unrelated).
  C  name char-3grams on the space-free name (domain-style names, empty addresses).

Forward (S1 -> S2/S3) takes the top-k per S1 entity. Reverse (S2/S3 -> S1) takes the
top few S1 entities per record: each S2/S3 record belongs to at most one S1 entity,
so its best-scoring S1 entities are strong candidates even when the S1 entity's own
forward list is crowded out by near-duplicate distractors.
"""
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS, cand_path, norm_path

# name -> (representation, k, max_df, rows); rows="short" restricts to S2/S3 records
# whose address has at most SHORT_ADDR tokens, where the name is the only signal.
FWD = {"Xf": ("X", 15, 0.002, None), "Af": ("A", 25, 0.003, None)}
REV = {"Xr": ("X", 3, 0.01, None), "Ar": ("A", 3, 0.004, None), "Br": ("B", 3, 0.02, None),
       "Es": ("C", 10, 0.08, "short")}
SHORT_ADDR = 4
REV_CAP = 30  # max reverse-only candidates kept per S1 entity (stops "hub" entities)
Q_CHUNK = 50000

_H = dict(n_features=2 ** 23, alternate_sign=False, norm=None, dtype=np.float32)
NAME_VEC = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False, **_H)
ADDR_VEC = HashingVectorizer(analyzer=str.split, **_H)
CROSS_VEC = HashingVectorizer(analyzer=lambda x: x, **_H)


def name_text(df):
    """Space-free name variants, so 'gkindia' and 'gk india' share their 3-grams."""
    return [" ".join(v for v in (c, a.replace(" ", ""), d.replace(" ", "")) if v)
            for c, a, d in zip(df["n_concat"], df["n_alt"], df["n_dom"])]


def cross_tokens(df):
    out = []
    for c, alt, cc, a in zip(df["n_core"], df["n_alt"], df["n_concat"], df["a_full"]):
        nt = set((c + " " + alt).split())
        if cc:
            nt.add("#" + cc)
        at = set(a.split())
        out.append([n + "|" + x for n in nt for x in at])
    return out


def vectorize(s1, r):
    t = time.time()
    raw = {
        "C": (NAME_VEC.transform(name_text(s1)), NAME_VEC.transform(name_text(r))),
        "B": (ADDR_VEC.transform(s1["a_full"].to_list()), ADDR_VEC.transform(r["a_full"].to_list())),
        "X": (CROSS_VEC.transform(cross_tokens(s1)), CROSS_VEC.transform(cross_tokens(r))),
    }
    out = {}
    for k, (q, rr) in raw.items():
        n = q.shape[0] + rr.shape[0]
        df = np.bincount(np.r_[q.indices, rr.indices], minlength=q.shape[1])
        idf = sp.diags((np.log((n + 1) / (df + 1)) + 1).astype(np.float32))
        out[k] = (normalize((q @ idf).tocsr()), normalize((rr @ idf).tocsr()))
    out["A"] = (normalize(sp.hstack([out["C"][0], out["B"][0]]).tocsr()),
                normalize(sp.hstack([out["C"][1], out["B"][1]]).tocsr()))
    out["short"] = (r["a_full"].str.count_matches(" ") < SHORT_ADDR).to_numpy() | (r["a_full"] == "").to_numpy()
    print(f"  vectorised in {time.time() - t:.0f}s", flush=True)
    return out


def prune(m, index, max_df):
    """Zero features that are too frequent on the index side (slow and uninformative).
    Scores stay comparable because both sides are then re-normalised."""
    df = np.bincount(index.indices, minlength=index.shape[1])
    keep = sp.diags((df <= max_df * index.shape[0]).astype(np.float32))
    a = normalize(m @ keep)
    a.eliminate_zeros()
    return a.tocsr()


def topk(q, index, k, max_df):
    qp, ip = prune(q, index, max_df), prune(index, index, max_df)
    it = ip.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, qp.shape[0], Q_CHUNK):
        c = sp_matmul_topn(qp[s:s + Q_CHUNK], it, top_n=k, threshold=0.05, n_threads=N_JOBS).tocoo()
        rows.append(c.row + s)
        cols.append(c.col)
        vals.append(c.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def retrieve(mats, fwd=None, rev=None, q_rows=None, r_rows=None):
    """Return DataFrame(qi, ri, <tag>_f / <tag>_r scores). Optional row subsets
    restrict the queries (used for fast recall experiments)."""
    fwd = FWD if fwd is None else fwd
    rev = REV if rev is None else rev
    parts = []
    for tag, (rep, k, mdf, rows) in fwd.items():
        t = time.time()
        q, rr = mats[rep]
        qq = q if q_rows is None else q[q_rows]
        i, j, v = topk(qq, rr, k, mdf)
        if q_rows is not None:
            i = q_rows[i]
        parts.append(pl.DataFrame({"qi": i.astype(np.int32), "ri": j.astype(np.int32), tag: v}))
        print(f"  {tag}: {len(i)} pairs in {time.time() - t:.0f}s", flush=True)
    for tag, (rep, k, mdf, rows) in rev.items():
        t = time.time()
        q, rr = mats[rep]
        sel = np.arange(rr.shape[0]) if r_rows is None else r_rows
        if rows == "short":
            sel = sel[mats["short"][sel]]
        j, i, v = topk(rr[sel], q, k, mdf)
        j = sel[j]
        parts.append(pl.DataFrame({"qi": i.astype(np.int32), "ri": j.astype(np.int32), tag: v}))
        print(f"  {tag}: {len(i)} pairs from {len(sel)} queries in {time.time() - t:.0f}s", flush=True)
    cand = parts[0]
    for p in parts[1:]:
        cand = cand.join(p, on=["qi", "ri"], how="full", coalesce=True)
    return cand


def cap_reverse(cand):
    fwd = [c for c in cand.columns if c in FWD]
    rev = [c for c in cand.columns if c in REV]
    is_fwd = pl.any_horizontal(pl.col(c).is_not_null() for c in fwd)
    rscore = pl.max_horizontal(rev)
    rank = pl.when(is_fwd).then(0).otherwise(
        rscore.rank("ordinal", descending=True).over(pl.col("s1_id" if "s1_id" in cand.columns else "qi"), is_fwd))
    return cand.filter(rank <= REV_CAP)


def block_country(split, country, s1, r):
    t = time.time()
    cand = cap_reverse(retrieve(vectorize(s1, r)))
    cand = cand.with_columns(
        s1_id=pl.Series(s1["entity_id"].to_numpy()[cand["qi"].to_numpy()]),
        r_id=pl.Series(r["entity_id"].to_numpy()[cand["ri"].to_numpy()]),
    ).drop("qi", "ri")
    cand.write_parquet(cand_path(split, country))
    print(f"{split}/{country}: {s1.height} queries, {cand.height} candidates "
          f"({cand.height / max(s1.height, 1):.1f}/query) in {time.time() - t:.0f}s", flush=True)


def load(split, country):
    s1 = pl.read_parquet(norm_path(split, "1")).filter(pl.col("country") == country)
    r = pl.concat([pl.read_parquet(norm_path(split, s)) for s in ("2", "3")]).filter(pl.col("country") == country)
    return s1, r


def main():
    if sys.argv[1] == "cap":
        for path in sys.argv[2:]:
            c = pl.read_parquet(path)
            capped = cap_reverse(c)
            capped.write_parquet(path)
            print(path, c.height, "->", capped.height)
        return
    split = sys.argv[1]
    countries = sys.argv[2:] or sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    for country in countries:
        block_country(split, country, *load(split, country))


if __name__ == "__main__":
    main()
