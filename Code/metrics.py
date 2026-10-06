"""Macro F0.5 exactly as defined in the challenge README (singletons included)."""
import polars as pl

from config import DATA_DIR


def load_truth_pairs():
    gt = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    pairs = (gt.drop_nulls("matched_entity_ids")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"}))
    return gt.select(pl.col("source1_entity_id").alias("s1_id")), pairs


def macro_f05(pred_pairs, truth_pairs, s1_ids):
    """pred_pairs/truth_pairs: DataFrames (s1_id, r_id); s1_ids: Series of evaluated entities."""
    ids = pl.DataFrame({"s1_id": s1_ids}).unique()
    pred = pred_pairs.join(ids, on="s1_id").select("s1_id", "r_id").unique()
    truth = truth_pairs.join(ids, on="s1_id").select("s1_id", "r_id").unique()
    tp = pred.join(truth, on=["s1_id", "r_id"]).group_by("s1_id").len("tp")
    per = (ids.join(pred.group_by("s1_id").len("np"), on="s1_id", how="left")
              .join(truth.group_by("s1_id").len("nt"), on="s1_id", how="left")
              .join(tp, on="s1_id", how="left")
              .fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    per = per.with_columns(f=f)
    return per["f"].mean(), per
