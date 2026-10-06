"""Pairwise similarity features for every candidate pair, computed in chunks.

String scores are stored as UInt8 (0-100) and cosines as Float32 to keep the full
train/test feature tables on disk at a manageable size.
"""
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from config import N_JOBS, WORK_DIR, cand_path, feat_path, norm_path

CHUNK = 2_000_000
BLOCK_COLS = ("Xf", "Af", "Xr", "Ar", "Br", "Es")

VECS = {
    "cn_word": HashingVectorizer(analyzer=lambda s: s.split(), n_features=2 ** 22,
                                 alternate_sign=False, norm=None, dtype=np.float32),
    "cn_char": HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 22,
                                 alternate_sign=False, norm=None, lowercase=False, dtype=np.float32),
    "ca_word": HashingVectorizer(analyzer=lambda s: s.split(), n_features=2 ** 22,
                                 alternate_sign=False, norm=None, dtype=np.float32),
    "ca_char": HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 22,
                                 alternate_sign=False, norm=None, lowercase=False, dtype=np.float32),
}
FIELD = {"cn_word": "nm", "cn_char": "nm", "ca_word": "a_full", "ca_char": "a_full"}


def with_nm(df):
    return df.with_columns(
        nm=pl.when(pl.col("n_core") != "").then(pl.col("n_core")).otherwise(pl.col("n_full"))
    )


def tfidf_pair(q_txt, r_txt, vec):
    qc, rc = vec.transform(q_txt), vec.transform(r_txt)
    n = rc.shape[0] + qc.shape[0]
    df = np.bincount(np.r_[qc.indices, rc.indices], minlength=rc.shape[1])
    idf = sp.diags((np.log((n + 1) / (df + 1)) + 1).astype(np.float32))
    return normalize((qc @ idf).tocsr()), normalize((rc @ idf).tocsr())


def rowdot(a, b):
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def u8(x):
    return np.clip(np.round(np.nan_to_num(x, nan=255)), 0, 255).astype(np.uint8)


def score(a, b, scorer, **kw):
    return u8(cpdist(a, b, scorer=scorer, workers=N_JOBS, **kw))


def chunk_features(P, S1, R, mats):
    qi, ri = P["qi"].to_numpy(), P["ri"].to_numpy()
    q = S1[qi]
    r = R[ri]
    out = {"s1_id": q["entity_id"], "r_id": r["entity_id"]}
    for c in BLOCK_COLS:
        out["blk_" + c] = P[c] if c in P.columns else pl.Series([None] * P.height, dtype=pl.Float32)
    out["novel2"] = r["novel"]
    out["novel1"] = q["novel"]
    for k, (mq, mr) in mats.items():
        out[k] = rowdot(mq[qi], mr[ri])
    out["cos_comb"] = ((out["cn_char"] + out["ca_word"]) / 2).astype(np.float32)

    n1, n2 = q["nm"].to_list(), r["nm"].to_list()
    out["n_ratio"] = score(n1, n2, fuzz.ratio)
    out["n_tsort"] = score(n1, n2, fuzz.token_sort_ratio)
    out["n_tset"] = score(n1, n2, fuzz.token_set_ratio)
    out["n_partial"] = score(n1, n2, fuzz.partial_ratio)
    out["n_jw"] = u8(cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=N_JOBS) * 100)
    out["nf_tset"] = score(q["n_full"].to_list(), r["n_full"].to_list(), fuzz.token_set_ratio)
    c1, c2 = q["n_concat"].to_list(), r["n_concat"].to_list()
    out["cc_ratio"] = score(c1, c2, fuzz.ratio)
    out["cc_partial"] = score(c1, c2, fuzz.partial_ratio)

    alt2, alt1 = r["n_alt"], q["n_alt"]
    a = score(n1, alt2.to_list(), fuzz.token_set_ratio)
    a[(alt2 == "").to_numpy()] = 255
    out["alt_tset"] = a
    b = score(alt1.to_list(), n2, fuzz.token_set_ratio)
    b[(alt1 == "").to_numpy()] = 255
    out["alt1_tset"] = b
    dom2 = r["n_dom"]
    d = score(c1, dom2.to_list(), fuzz.partial_ratio)
    d[(dom2 == "").to_numpy()] = 255
    out["dom_partial"] = d

    a1, a2 = q["a_full"].to_list(), r["a_full"].to_list()
    out["a_ratio"] = score(a1, a2, fuzz.ratio)
    out["a_tsort"] = score(a1, a2, fuzz.token_sort_ratio)
    out["a_tset"] = score(a1, a2, fuzz.token_set_ratio)
    out["a_partial"] = score(a1, a2, fuzz.partial_ratio)

    df = pl.DataFrame(out)
    tok = pl.DataFrame({
        "n1": q["nm"], "n2": r["nm"], "l1": q["n_legal"], "l2": r["n_legal"],
        "d1": q["n_digits"], "d2": r["n_digits"], "u1": q["a_nums"], "u2": r["a_nums"],
        "h1": q["a_house"], "h2": r["a_house"], "e1": q["a_full"], "e2": r["a_full"],
    }).with_columns([pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias(c + "L")
                     for c in ("n1", "n2", "l1", "l2", "d1", "d2", "u1", "u2", "e1", "e2")])

    def inter(a, b):
        return pl.col(a).list.set_intersection(pl.col(b)).list.len()

    feats = tok.select(
        n_len1=pl.col("n1L").list.len().cast(pl.UInt8),
        n_len2=pl.col("n2L").list.len().cast(pl.UInt8),
        n_inter=inter("n1L", "n2L").cast(pl.UInt8),
        n_only1=pl.col("n1L").list.set_difference(pl.col("n2L")).list.len().cast(pl.UInt8),
        n_only2=pl.col("n2L").list.set_difference(pl.col("n1L")).list.len().cast(pl.UInt8),
        legal_state=pl.when((pl.col("l1") == "") & (pl.col("l2") == "")).then(0)
            .when((pl.col("l1") == "") | (pl.col("l2") == "")).then(1)
            .when(pl.col("l1") == pl.col("l2")).then(2)
            .when(inter("l1L", "l2L") > 0).then(3).otherwise(4).cast(pl.UInt8),
        digit_state=pl.when((pl.col("d1") == "") & (pl.col("d2") == "")).then(0)
            .when((pl.col("d1") == "") | (pl.col("d2") == "")).then(1)
            .when(pl.col("d1") == pl.col("d2")).then(2).otherwise(3).cast(pl.UInt8),
        house_state=pl.when((pl.col("h1") == "") | (pl.col("h2") == "")).then(0)
            .when(pl.col("h1") == pl.col("h2")).then(2)
            .when(pl.col("u2L").list.contains(pl.col("h1")) | pl.col("u1L").list.contains(pl.col("h2"))).then(1)
            .otherwise(3).cast(pl.UInt8),
        num_inter=inter("u1L", "u2L").cast(pl.UInt8),
        num_len1=pl.col("u1L").list.len().cast(pl.UInt8),
        num_len2=pl.col("u2L").list.len().cast(pl.UInt8),
        a_len1=pl.col("e1L").list.len().cast(pl.UInt8),
        a_len2=pl.col("e2L").list.len().cast(pl.UInt8),
        a_inter=inter("e1L", "e2L").cast(pl.UInt8),
    )
    flags = pl.DataFrame({
        "f_domain2": r["f_domain"], "f_alias2": r["f_alias"], "f_phone2": r["f_phone"],
        "f_nonlatin2": r["f_nonlatin"], "f_addrnl2": r["f_addr_nonlatin"],
        "f_domain1": q["f_domain"], "f_alias1": q["f_alias"],
        "src3": r["entity_id"].str.starts_with("S3-"),
    }).cast(pl.UInt8)
    return pl.concat([df, feats, flags], how="horizontal")


def add_context(F):
    """Competition features: how this pair compares with other candidates of the same
    S1 entity and with other S1 entities claiming the same S2/S3 record."""
    s = pl.col("cos_comb") + pl.col("n_tset").cast(pl.Float32) / 400
    return F.with_columns(
        q_cnt=pl.len().over("s1_id").cast(pl.UInt32),
        q_max=s.max().over("s1_id"),
        q_rank=s.rank("ordinal", descending=True).over("s1_id").cast(pl.UInt32),
        r_cnt=pl.len().over("r_id").cast(pl.UInt32),
        r_max=s.max().over("r_id"),
        r_rank=s.rank("ordinal", descending=True).over("r_id").cast(pl.UInt32),
        n_tset_rmax=pl.col("n_tset").max().over("r_id"),
        a_tset_rmax=pl.col("a_tset").max().over("r_id"),
    ).with_columns(
        q_gap=pl.col("q_max") - s,
        r_gap=pl.col("r_max") - s,
        r_second=pl.when(pl.col("r_rank") == 1)
            .then(s - s.filter(pl.col("r_rank") == 2).max().over("r_id").fill_null(0))
            .otherwise(s - pl.col("r_max")),
    )


def add_novelty(s1, r):
    """Share of name tokens never used by any S1 name of this country: invented
    rebrand names and distractor-only vocabulary score high."""
    vocab = s1.select(pl.col("n_full").str.split(" ").explode().alias("t")).unique().with_columns(v=pl.lit(1))

    def novel(df):
        tok = (df.select(pl.int_range(pl.len()).alias("i"), pl.col("n_full").str.split(" ").alias("t"))
                 .explode("t").filter(pl.col("t") != "")
                 .join(vocab, on="t", how="left")
                 .group_by("i").agg(pl.col("v").is_null().mean().alias("novel")))
        return (df.with_row_index("i").join(tok.with_columns(pl.col("i").cast(pl.UInt32)), on="i", how="left")
                  .drop("i").with_columns(pl.col("novel").fill_null(0).cast(pl.Float32)))
    return novel(s1), novel(r)


KEEP = ["entity_id", "nm", "n_full", "n_concat", "n_alt", "n_dom", "n_legal", "n_digits", "a_full", "a_nums",
        "a_house", "f_domain", "f_alias", "f_phone", "f_nonlatin", "f_addr_nonlatin", "novel"]


def build(split, country):
    t = time.time()
    s1 = with_nm(pl.read_parquet(norm_path(split, "1")).filter(pl.col("country") == country))
    r = with_nm(pl.concat([pl.read_parquet(norm_path(split, s)) for s in ("2", "3")])
                .filter(pl.col("country") == country))
    s1, r = add_novelty(s1, r)
    s1, r = s1.select(KEEP), r.select(KEEP)
    cand = (pl.read_parquet(cand_path(split, country))
              .join(s1.select(pl.col("entity_id").alias("s1_id")).with_row_index("qi"), on="s1_id")
              .join(r.select(pl.col("entity_id").alias("r_id")).with_row_index("ri"), on="r_id")
              .sort("qi"))
    mats = {k: tfidf_pair(s1[FIELD[k]].to_list(), r[FIELD[k]].to_list(), v) for k, v in VECS.items()}
    print(f"  {split}/{country}: {cand.height} pairs, tfidf ready in {time.time() - t:.0f}s", flush=True)
    tmp = []
    for i, s in enumerate(range(0, cand.height, CHUNK)):
        path = WORK_DIR / f"tmp_feat_{split}_{country}_{i:03d}.parquet"
        chunk_features(cand[s:s + CHUNK], s1, r, mats).write_parquet(path)
        tmp.append(path)
        print(f"    {min(s + CHUNK, cand.height)}/{cand.height} ({time.time() - t:.0f}s)", flush=True)
    del mats, s1, r, cand
    F = add_context(pl.read_parquet(tmp))
    F.write_parquet(feat_path(split, country))
    for p in tmp:
        p.unlink()
    print(f"{split}/{country}: features {F.shape} in {time.time() - t:.0f}s", flush=True)


def main():
    split = sys.argv[1]
    countries = sys.argv[2:]
    if not countries:
        countries = sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    for c in countries:
        build(split, c)


if __name__ == "__main__":
    main()
