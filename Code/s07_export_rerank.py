"""Export the uncertain pairs as raw text for the GPU cross-encoders.

Uncertain = the LightGBM pipeline's probability is strictly between LO and HI; the
confident ~97% of pairs are left alone. Files written to <out_dir>:
  train_pairs.parquet   labelled training pairs (never from tuning/report folds 7,17,8,18)
  heldout_pairs.parquet uncertain pairs of tuning/report folds, for blend tuning + report
  test_pairs.parquet    uncertain test pairs
  stress_pairs.parquet  India report-fold pairs uncertain under a US-only model (France proxy)

Usage: python s07_export_rerank.py <out_dir>
  <out_dir> defaults to $RERANK_DATA (= $ER_WORK/rerank_data) when run via run_all.sh.
  rerank.py and rerank2.py read from the same directory via the RERANK_DATA env var.
"""
import os
import sys

# pyrefly: ignore [missing-import]
import polars as pl

import s05_train as S
from config import DATA_DIR, WORK_DIR

LO, HI = 0.003, 0.997
N_TRAIN = 900_000


def texts(split):
    def rd(src):
        return pl.read_csv(DATA_DIR / split / f"{split}_source{src}.tsv", separator="\t", quote_char=None, infer_schema=False)
    t = lambda d: d.select("entity_id", "country", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("text"))
    return t(rd("1")), t(pl.concat([rd("2"), rd("3")]))


def attach(P, s1, r):
    return (P.join(s1.rename({"entity_id": "s1_id", "text": "text_a"}), on="s1_id")
             .join(r.drop("country").rename({"entity_id": "r_id", "text": "text_b"}), on="r_id"))


def band(col):
    return (pl.col(col) > LO) & (pl.col(col) < HI)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    s1, r = texts("train")
    T = S.truth().select("s1_id", "r_id").with_columns(label=pl.lit(1, pl.Int8))
    pool = pl.concat([
        pl.scan_parquet(WORK_DIR / f"oof_train_{c}.parquet").select("s1_id", "r_id", "p1").with_columns(S.fold_expr())
          .filter(~pl.col("fold").is_in(list(S.TUNE + S.REPORT)) & band("p1")).collect()
        for c in S.COUNTRIES])
    pool = pool.sample(min(N_TRAIN, pool.height), seed=S.SEED)
    tr = attach(pool, s1, r).join(T, on=["s1_id", "r_id"], how="left").with_columns(pl.col("label").fill_null(0))
    tr.select("s1_id", "r_id", "country", "text_a", "text_b", "label").write_parquet(os.path.join(out_dir, "train_pairs.parquet"))
    print(f"train: {tr.height:,} pairs, positives {tr['label'].mean():.3f}, by country {tr.group_by('country').len().rows()}")

    ev = pl.read_parquet(WORK_DIR / "eval_pred.parquet").filter(band("p")).select("s1_id", "r_id")
    ev = attach(ev, s1, r)
    ev.select("s1_id", "r_id", "country", "text_a", "text_b").write_parquet(os.path.join(out_dir, "heldout_pairs.parquet"))
    print(f"heldout: {ev.height:,} pairs")

    st = pl.read_parquet(WORK_DIR / "xcountry_US_to_India.parquet").filter(band("p")).select("s1_id", "r_id")
    st = attach(st, s1, r)
    st.select("s1_id", "r_id", "country", "text_a", "text_b").write_parquet(os.path.join(out_dir, "stress_pairs.parquet"))
    print(f"stress (US-trained model, India report folds): {st.height:,} pairs")

    s1t, rt = texts("test")
    te = pl.read_parquet(WORK_DIR / "test_scores.parquet").filter(band("p")).select("s1_id", "r_id")
    te = attach(te, s1t, rt)
    te.select("s1_id", "r_id", "country", "text_a", "text_b").write_parquet(os.path.join(out_dir, "test_pairs.parquet"))
    print(f"test: {te.height:,} pairs, by country {te.group_by('country').len().rows()}")


if __name__ == "__main__":
    main(sys.argv[1])
