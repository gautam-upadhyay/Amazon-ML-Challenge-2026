"""Structured difference features, appended to the feature parquet in place.

Distractors in this data are near-copies of real entities with one or two deliberate
edits: a changed house number, a changed legal form, one substituted name word, or a
phonetic/ending mutation of the distinctive word. True matches carry different noise:
added filler words, dropped legal words, random typos and extra prefix numbers.
These features describe *what kind* of difference separates two records rather than
only how similar they are.
"""
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

import jellyfish
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from config import N_JOBS, WORK_DIR, feat_path, norm_path

LEGAL_KEYS = ("ltd", "pvt", "public", "llc", "llp", "lp", "inc", "corp", "co", "pc", "pllc", "plc",
              "sa", "sas", "sasu", "sarl", "eurl", "sci")
LEGAL_FULL = {"private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc",
              "company": "co", "public": "public"}
STOP = {"the", "and", "of", "a", "an", "du", "de", "des", "la", "le", "les", "l", "d", "en", "for", "at"}
CLOSE = 0.6
NAMES = (
    ["n1_exact", "n1_close", "n1_none", "n2_exact", "n2_close", "n2_none", "close_min", "close_meta_eq",
     "close_prefix", "close_suffix", "none1_idf", "none2_idf", "subst", "legal_eq", "legal_symdiff"]
    + [f"lg1_{k}" for k in LEGAL_KEYS] + [f"lg2_{k}" for k in LEGAL_KEYS]
    + ["num_miss1", "num_extra2", "num_best_sim", "num_absdiff", "house1_in2", "house2_in1",
       "aw1_none", "aw2_none", "aw1_close", "aw_frac1"]
)

_IDF = {}
_MAXIDF = 0.0


def legal_of(t):
    if t in LEGAL_FULL:
        return LEGAL_FULL[t]
    if t in LEGAL_KEYS:
        return t
    if len(t) >= 5:
        for full, key in LEGAL_FULL.items():
            if Levenshtein.normalized_similarity(t, full) >= 0.75:
                return key
    return None


def split_name(s):
    core, legal = [], set()
    for t in s.split():
        k = legal_of(t)
        if k is not None:
            legal.add(k)
        elif t not in STOP:
            core.append(t)
    return core, legal


def best_match(a, pool):
    best, bs = None, 0.0
    for b in pool:
        s = Levenshtein.normalized_similarity(a, b)
        if s > bs:
            best, bs = b, s
    return best, bs


def pair_feats(n1, n2, a1, a2, u1, u2):
    c1, l1 = split_name(n1)
    c2, l2 = split_name(n2)
    s1, s2 = set(c1), set(c2)
    e1 = cl1 = no1 = 0
    close_min, meta_eq, pref, suf = 1.0, 0, 1.0, 1.0
    none1_idf = none2_idf = 0.0
    for a in c1:
        if a in s2:
            e1 += 1
            continue
        b, s = best_match(a, c2)
        if s >= CLOSE:
            cl1 += 1
            close_min = min(close_min, s)
            if jellyfish.metaphone(a) == jellyfish.metaphone(b):
                meta_eq += 1
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            pref = min(pref, k / max(len(a), 1))
            k2 = 0
            while k2 < min(len(a), len(b)) and a[-1 - k2] == b[-1 - k2]:
                k2 += 1
            suf = min(suf, k2 / max(len(a), 1))
        else:
            no1 += 1
            none1_idf = max(none1_idf, _IDF.get(a, _MAXIDF))
    e2 = cl2 = no2 = 0
    for b in c2:
        if b in s1:
            e2 += 1
        elif best_match(b, c1)[1] >= CLOSE:
            cl2 += 1
        else:
            no2 += 1
            none2_idf = max(none2_idf, _IDF.get(b, _MAXIDF))
    if cl1 == 0:
        close_min, pref, suf = -1.0, -1.0, -1.0
    legal = [int(l1 == l2), len(l1 ^ l2)] + [int(k in l1) for k in LEGAL_KEYS] + [int(k in l2) for k in LEGAL_KEYS]

    n_1, n_2 = u1.split(), u2.split()
    sn1, sn2 = set(n_1), set(n_2)
    miss1 = [x for x in n_1 if x not in sn2]
    extra2 = [y for y in n_2 if y not in sn1]
    best_sim, absdiff = -1.0, -1.0
    if miss1 and extra2:
        best_sim = max(Levenshtein.normalized_similarity(x, y) for x in miss1 for y in extra2)
        diffs = [abs(int(x[:9]) - int(y[:9])) for x in miss1 for y in extra2]
        absdiff = math.log1p(min(diffs))
    house1_in2 = (1 if n_1[0] in sn2 else 0) if n_1 else -1
    house2_in1 = (1 if n_2[0] in sn1 else 0) if n_2 else -1
    w1 = [t for t in a1.split() if not any(ch.isdigit() for ch in t)]
    w2 = [t for t in a2.split() if not any(ch.isdigit() for ch in t)]
    sw2 = set(w2)
    aw1_none = aw1_close = 0
    for t in w1:
        if t in sw2:
            continue
        if w2 and best_match(t, w2)[1] >= 0.8:
            aw1_close += 1
        else:
            aw1_none += 1
    sw1 = set(w1)
    aw2_none = sum(1 for t in w2 if t not in sw1)
    aw_frac1 = (len(w1) - aw1_none) / len(w1) if w1 else -1.0
    return ([e1, cl1, no1, e2, cl2, no2, close_min, meta_eq, pref, suf, none1_idf, none2_idf,
             int(no1 > 0 and no2 > 0)] + legal
            + [len(miss1), len(extra2), best_sim, absdiff, house1_in2, house2_in1,
               aw1_none, aw2_none, aw1_close, aw_frac1])


def _init(idf_path):
    global _IDF, _MAXIDF
    with open(idf_path, "rb") as f:
        _IDF, _MAXIDF = pickle.load(f)


def work(rows):
    return [pair_feats(*r) for r in rows]


def build_idf(s1, r):
    toks = pl.concat([s1.select("n_full"), r.select("n_full")]).select(pl.col("n_full").str.split(" ").explode().alias("t"))
    df = toks.group_by("t").len()
    n = s1.height + r.height
    idf = df.with_columns(idf=(np.log(n / pl.col("len"))).cast(pl.Float32))
    return dict(zip(idf["t"].to_list(), idf["idf"].to_list())), float(np.log(n))


def run(split, country, batch=2_000_000):
    t = time.time()
    cols = ["entity_id", "n_full", "a_full", "a_nums"]
    s1 = pl.read_parquet(norm_path(split, "1"), columns=cols + ["country"]).filter(pl.col("country") == country).drop("country")
    r = pl.concat([pl.read_parquet(norm_path(split, s), columns=cols + ["country"]) for s in ("2", "3")]).filter(pl.col("country") == country).drop("country")
    idf_path = WORK_DIR / f"idf_{split}_{country}.pkl"
    with open(idf_path, "wb") as f:
        pickle.dump(build_idf(s1, r), f)
    s1 = s1.rename({c: c + "_1" for c in cols}).rename({"entity_id_1": "s1_id"})
    r = r.rename({c: c + "_2" for c in cols}).rename({"entity_id_2": "r_id"})
    src = feat_path(split, country)
    dst = src.with_suffix(".tmp.parquet")
    pf = pq.ParquetFile(src)
    if "n1_exact" in pf.schema_arrow.names:
        print(f"{split}/{country}: diff features already present")
        return
    writer = None
    with mp.get_context("spawn").Pool(N_JOBS, initializer=_init, initargs=(str(idf_path),)) as pool:
        for i, rb in enumerate(pf.iter_batches(batch_size=batch)):
            F = pl.from_arrow(rb)
            J = (F.select("s1_id", "r_id").with_row_index("_i")
                   .join(s1, on="s1_id", how="left").join(r, on="r_id", how="left").sort("_i")
                   .fill_null(""))
            rows = list(zip(J["n_full_1"], J["n_full_2"], J["a_full_1"], J["a_full_2"], J["a_nums_1"], J["a_nums_2"]))
            step = max(1, len(rows) // (N_JOBS * 8))
            res = [x for part in pool.map(work, [rows[k:k + step] for k in range(0, len(rows), step)]) for x in part]
            D = pl.DataFrame(np.asarray(res, dtype=np.float32), schema=NAMES, orient="row")
            out = pl.concat([F, D], how="horizontal").to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, out.schema)
            writer.write_table(out)
            print(f"  {split}/{country} batch {i}: {(i + 1) * batch} rows ({time.time() - t:.0f}s)", flush=True)
    writer.close()
    os.replace(dst, src)
    print(f"{split}/{country}: diff features done in {time.time() - t:.0f}s", flush=True)


def main():
    split = sys.argv[1]
    countries = sys.argv[2:] or sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    for c in countries:
        run(split, c)


if __name__ == "__main__":
    main()
