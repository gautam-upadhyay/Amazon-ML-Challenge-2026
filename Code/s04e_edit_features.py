"""Edit-type features, appended to the feature parquet in place.

True-match noise and distractor edits leave different fingerprints:
- names: true-match typos are scrambles (insertions, deletions, transpositions that keep
  the letters: "Innvataions", "cflinic"); distractor mutations are same-length
  substitutions of the distinctive word ("Aditin"->"Adityx", "Yamuika"->"Yamuica");
- house numbers: true-match noise truncates or prefixes ("16901"->"1690", "##1160");
  distractors change digits at the same length ("20216"->"20221", "7010"->"7014").
These describe the KIND of edit between the closest non-identical tokens and numbers.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein, OSA

from config import N_JOBS, feat_path, norm_path

STOP = {"the", "and", "of", "a", "an", "du", "de", "des", "la", "le", "les", "l", "d", "en", "for", "at"}
LEGAL = {"ltd", "pvt", "public", "llc", "llp", "lp", "inc", "corp", "co", "pc", "pllc", "plc", "sa", "sas", "sasu",
         "sarl", "eurl", "sci", "private", "limited", "corporation", "incorporated", "company"}
NAMES = ["ed_subs", "ed_indels", "ed_transp", "ed_anagram", "ed_samelen_sub", "ed_max_sub_word", "ed_close_n",
         "ed_cc_subs", "ed_cc_indels",
         "hn_exact", "hn_prefix", "hn_subseq", "hn_samelen_1", "hn_samelen_2p", "hn_digit_anagram", "hn_lendiff",
         "hn_absdiff_log", "hn_any_in"]


def core(s):
    return [t for t in s.split() if t not in STOP and t not in LEGAL]


def op_counts(a, b):
    subs = ins = dels = 0
    for op in Levenshtein.editops(a, b):
        if op.tag == "replace":
            subs += 1
        elif op.tag == "insert":
            ins += 1
        else:
            dels += 1
    return subs, ins + dels


def is_subseq(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def pair_feats(n1, n2, c1, c2, h1, u2):
    t1, t2 = core(n1), core(n2)
    s2 = set(t2)
    subs = indels = transp = anag = samelen_sub = close = 0
    max_sub_word = 0
    for a in t1:
        if a in s2 or not t2:
            continue
        b = max(t2, key=lambda x: Levenshtein.normalized_similarity(a, x))
        if Levenshtein.normalized_similarity(a, b) < 0.6:
            continue
        close += 1
        s, d = op_counts(a, b)
        subs += s
        indels += d
        if OSA.distance(a, b) < Levenshtein.distance(a, b):
            transp += 1
        if sorted(a) == sorted(b):
            anag += 1
        if len(a) == len(b) and s > 0 and d == 0:
            samelen_sub += 1
            max_sub_word = max(max_sub_word, s)
    cs, cd = op_counts(c1, c2) if (c1 and c2 and c1 != c2) else (0, 0)
    nums = u2.split()
    if h1 and nums:
        best = max(nums, key=lambda x: (x == h1, Levenshtein.normalized_similarity(h1, x)))
        exact = int(best == h1)
        prefix = int(not exact and (h1.startswith(best) or best.startswith(h1)))
        short, long = (best, h1) if len(best) <= len(h1) else (h1, best)
        subseq = int(not exact and is_subseq(short, long))
        diffd = sum(x != y for x, y in zip(h1, best)) if len(best) == len(h1) else -1
        hn = [exact, prefix, subseq, int(diffd == 1), int(diffd >= 2),
              int(not exact and sorted(h1) == sorted(best)), len(best) - len(h1),
              float(np.log1p(abs(int(h1[:9]) - int(best[:9])))), int(h1 in nums)]
    else:
        hn = [-1, -1, -1, -1, -1, -1, 0, -1.0, -1]
    return [subs, indels, transp, anag, samelen_sub, max_sub_word, close, cs, cd] + hn


def work(rows):
    return [pair_feats(*r) for r in rows]


def run(split, country, batch=2_000_000):
    t = time.time()
    src = feat_path(split, country)
    pf = pq.ParquetFile(src)
    if NAMES[0] in pf.schema_arrow.names:
        print(f"{split}/{country}: edit features already present")
        return
    cols = ["entity_id", "country", "n_full", "n_concat", "a_house", "a_nums"]
    s1 = (pl.read_parquet(norm_path(split, "1"), columns=cols).filter(pl.col("country") == country)
            .select(pl.col("entity_id").alias("s1_id"), pl.col("n_full").alias("n1"), pl.col("n_concat").alias("c1"), pl.col("a_house").alias("h1")))
    r = (pl.concat([pl.read_parquet(norm_path(split, s), columns=cols) for s in ("2", "3")]).filter(pl.col("country") == country)
           .select(pl.col("entity_id").alias("r_id"), pl.col("n_full").alias("n2"), pl.col("n_concat").alias("c2"), pl.col("a_nums").alias("u2")))
    dst = src.with_suffix(".tmp.parquet")
    writer = None
    with mp.get_context("spawn").Pool(N_JOBS) as pool:
        for i, rb in enumerate(pf.iter_batches(batch_size=batch)):
            F = pl.from_arrow(rb)
            J = (F.select("s1_id", "r_id").with_row_index("_i").join(s1, on="s1_id", how="left")
                   .join(r, on="r_id", how="left").sort("_i").fill_null(""))
            rows = list(zip(J["n1"], J["n2"], J["c1"], J["c2"], J["h1"], J["u2"]))
            step = max(1, len(rows) // (N_JOBS * 8))
            res = [x for part in pool.map(work, [rows[k:k + step] for k in range(0, len(rows), step)]) for x in part]
            D = pl.DataFrame(np.asarray(res, dtype=np.float32), schema=NAMES, orient="row")
            out = pl.concat([F, D], how="horizontal").to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, out.schema)
            writer.write_table(out)
    writer.close()
    os.replace(dst, src)
    print(f"{split}/{country}: edit features done in {time.time() - t:.0f}s", flush=True)


def main():
    split = sys.argv[1]
    countries = sys.argv[2:] or sorted(pl.read_parquet(norm_path(split, "1"))["country"].unique().to_list())
    for c in countries:
        run(split, c)


if __name__ == "__main__":
    main()
