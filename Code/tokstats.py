"""Learned token log-odds: how a token that appears on only one side of a pair relates
to the pair being a true match. Filler words the data generator adds to true matches
("services", "center") score high; words it substitutes into near-duplicate
distractors ("steels", "foods") score low. Tables are learned only from folds
reserved for this purpose (never used to train or evaluate the models)."""
import numpy as np
import polars as pl

from config import SEED, WORK_DIR, feat_path, norm_path

STAT_FOLDS = (9, 15, 19)
N_FOLDS = 20
MIN_COUNT = 20
SIDES = {"n2": ("nm_2", "nm_1"), "n1": ("nm_1", "nm_2"), "a2": ("aw_2", "aw_1"), "a1": ("aw_1", "aw_2")}
FEATS = [f"tk_{s}_{a}" for s in SIDES for a in ("sum", "min", "max", "unk")]


def fold_expr():
    return (pl.col("s1_id").hash(seed=SEED) % N_FOLDS).cast(pl.Int8).alias("fold")


def token_frames(split, country):
    cols = ["entity_id", "n_full", "a_full"]
    words = pl.col("a_full").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element() != "")))
    names = pl.col("n_full").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    s1 = (pl.read_parquet(norm_path(split, "1"), columns=cols + ["country"]).filter(pl.col("country") == country)
            .select(pl.col("entity_id").alias("s1_id"), names.alias("nm_1"), words.alias("aw_1")))
    r = (pl.concat([pl.read_parquet(norm_path(split, s), columns=cols + ["country"]) for s in ("2", "3")])
           .filter(pl.col("country") == country)
           .select(pl.col("entity_id").alias("r_id"), names.alias("nm_2"), words.alias("aw_2")))
    return s1, r


def one_sided(P, side):
    a, b = SIDES[side]
    return (P.select("_i", pl.col(a).list.set_difference(pl.col(b)).alias("t"), *([c for c in ("label",) if c in P.columns]))
             .explode("t").drop_nulls("t"))


def build_tables(countries, truth):
    rows = {s: [] for s in SIDES}
    for c in countries:
        s1, r = token_frames("train", c)
        P = (pl.scan_parquet(feat_path("train", c)).select("s1_id", "r_id").with_columns(fold_expr())
               .filter(pl.col("fold").is_in(list(STAT_FOLDS))).collect())
        P = P.join(truth.select("s1_id", "r_id").with_columns(label=pl.lit(1, pl.Int8)), on=["s1_id", "r_id"], how="left")
        P = P.with_columns(pl.col("label").fill_null(0)).join(s1, on="s1_id").join(r, on="r_id").with_row_index("_i")
        for side in SIDES:
            rows[side].append(one_sided(P, side).group_by("t").agg(pos=pl.col("label").sum(), n=pl.len()))
    tables = {}
    for side, parts in rows.items():
        t = pl.concat(parts).group_by("t").agg(pl.col("pos").sum(), pl.col("n").sum()).filter(pl.col("n") >= MIN_COUNT)
        prior = t["pos"].sum() / t["n"].sum()
        t = t.with_columns(lo=(np.log((pl.col("pos") + 1) / (pl.col("n") - pl.col("pos") + 1)) - np.log(prior / (1 - prior))).cast(pl.Float32))
        tables[side] = t.select("t", "lo")
        t.select("t", "lo").write_parquet(WORK_DIR / f"tok_{side}.parquet")
    return tables


def soft_tables(P, s1, r):
    """Token tables from the pipeline's own scores (column p) as soft labels — for a
    country whose vocabulary the label-learned tables do not cover. Validated on the
    cross-country stress test: recovers ~+0.4 of the ~+2.6 points the label-learned
    tables are worth."""
    import numpy as np
    P = P.join(s1, on="s1_id").join(r, on="r_id").with_row_index("_i")
    tables = {}
    for side in SIDES:
        e = one_sided(P.rename({"p": "label"}), side)
        t = e.group_by("t").agg(pos=pl.col("label").sum(), n=pl.len()).filter(pl.col("n") >= MIN_COUNT)
        prior = t["pos"].sum() / t["n"].sum()
        t = t.with_columns(lo=(np.log((pl.col("pos") + 1) / (pl.col("n") - pl.col("pos") + 1))
                               - np.log(prior / (1 - prior))).cast(pl.Float32))
        tables[side] = t.select("t", "lo")
    return tables


def load_tables():
    return {s: pl.read_parquet(WORK_DIR / f"tok_{s}.parquet") for s in SIDES}


def token_features(P, tables):
    """P: frame with nm_1, nm_2, aw_1, aw_2 list columns (any order). Returns features in P's row order."""
    P = P.with_row_index("_i")
    out = P.select("_i")
    for side, tab in tables.items():
        e = one_sided(P, side).join(tab, on="t", how="left")
        agg = e.group_by("_i").agg(
            pl.col("lo").sum().alias(f"tk_{side}_sum"), pl.col("lo").min().alias(f"tk_{side}_min"),
            pl.col("lo").max().alias(f"tk_{side}_max"), pl.col("lo").is_null().sum().cast(pl.Float32).alias(f"tk_{side}_unk"))
        out = out.join(agg, on="_i", how="left")
    return out.sort("_i").drop("_i").with_columns(
        [pl.col(c).cast(pl.Float32) for c in FEATS]).with_columns(
        [pl.col(f"tk_{s}_sum").fill_null(0.0) for s in SIDES] + [pl.col(f"tk_{s}_unk").fill_null(0.0) for s in SIDES])
