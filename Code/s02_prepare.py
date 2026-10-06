"""Normalise every source file into a parquet of canonical name/address fields."""
import json
import sys
import time
from multiprocessing import Pool

import polars as pl

import textnorm
from config import N_JOBS, SOURCES, SPLITS, WORK_DIR, norm_path, raw_path

CHUNK = 20000


def _init():
    with open(WORK_DIR / "translit.json") as f:
        d = json.load(f)
    textnorm.set_translit(d["name"], d["addr"])


def _work(rows):
    out = []
    for name, addr, country in rows:
        rec = textnorm.normalize_name(name)
        rec.update(textnorm.normalize_address(addr, country))
        out.append(rec)
    return out


def prepare(split, src, pool):
    t = time.time()
    df = pl.read_csv(raw_path(split, src), separator="\t", quote_char=None, infer_schema=False)
    df = df.with_columns(pl.col("business_address").fill_null("").str.replace_all("<NULL>", ""),
                         pl.col("business_name").fill_null(""), pl.col("country").fill_null(""))
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    recs = [r for part in pool.imap(_work, chunks) for r in part]
    norm = pl.DataFrame(recs)
    out = pl.concat([df.select("entity_id", "country"), norm], how="horizontal")
    out.write_parquet(norm_path(split, src))
    print(f"{split} s{src}: {out.height} rows in {time.time() - t:.0f}s", flush=True)


def main():
    splits = sys.argv[1:] or SPLITS
    with Pool(N_JOBS, initializer=_init) as pool:
        for split in splits:
            for src in SOURCES:
                prepare(split, src, pool)


if __name__ == "__main__":
    main()
