"""Context features built from stage-1 probabilities (column p1)."""
import polars as pl

from config import norm_path

MEMBER_P = 0.3
KEYS = (("h", "house"), ("nk", "name"), ("lk", "legal"), ("uk", "nums"))


def add_stage2_features(F):
    p = pl.col("p1")
    F = F.with_columns(
        p1_rank_q=p.rank("ordinal", descending=True).over("s1_id").cast(pl.UInt32),
        p1_max_q=p.max().over("s1_id"),
        p1_sum_q=p.sum().over("s1_id"),
        p1_n50_q=(p > 0.5).sum().over("s1_id").cast(pl.UInt32),
        p1_rank_r=p.rank("ordinal", descending=True).over("r_id").cast(pl.UInt32),
        p1_sum_r=p.sum().over("r_id"),
        p1_max_r=p.max().over("r_id"),
    )
    second_r = p.filter(pl.col("p1_rank_r") == 2).max().over("r_id").fill_null(0)
    return F.with_columns(
        p1_other_r=pl.when(pl.col("p1_rank_r") == 1).then(second_r).otherwise(pl.col("p1_max_r")),
        p1_gap_q=pl.col("p1_max_q") - p,
    ).with_columns(p1_gap_r=p - pl.col("p1_other_r"))


def _keys(df, suffix):
    def h(col):
        return pl.when(pl.col(col) != "").then(pl.col(col).hash()).otherwise(None)
    return df.select(
        pl.col("entity_id"),
        h("a_house").alias("h" + suffix),
        pl.col("n_core").str.split(" ").list.unique().list.sort().list.join(" ").hash().alias("nk" + suffix),
        pl.col("n_legal").hash().alias("lk" + suffix),
        h("a_nums").alias("uk" + suffix),
    )


def add_cluster_features(F, split, country, parts=6, col="p1", prefix="cc_"):
    """Partitioned by S1 entity (statistics are per entity) to bound memory; row order is kept."""
    cols = ["entity_id", "country", "a_house", "n_core", "n_legal", "a_nums"]
    r = pl.concat([pl.read_parquet(norm_path(split, s), columns=cols) for s in ("2", "3")]).filter(pl.col("country") == country)
    q = pl.read_parquet(norm_path(split, "1"), columns=cols).filter(pl.col("country") == country)
    rk = _keys(r, "").rename({"entity_id": "r_id"})
    qk = _keys(q, "q").rename({"entity_id": "s1_id"})
    del r, q
    F = F.with_row_index("_row")
    part = (pl.col("s1_id").hash(seed=11) % parts).alias("_part")
    out = [_cluster(F.filter(part == i), rk, qk, col, prefix) for i in range(parts)]
    return pl.concat(out).sort("_row").drop("_row")


def _cluster(F, rk, qk, col="p1", prefix="cc_"):
    """Cluster consensus: does this record agree with the *other* records the model
    already assigns to the same S1 entity? Distractors are isolated one-off copies
    (a deviating house number is shared by another attached record only ~9% of the
    time), while true matches agree with their cluster (~63%), even when the S1
    record itself is noisy. Weighted by stage-1 probability; the S1 record counts
    as one more member."""
    F = F.join(rk, on="r_id", how="left").join(qk, on="s1_id", how="left")
    w = pl.when(pl.col(col) >= MEMBER_P).then(pl.col(col)).otherwise(0.0)
    F = F.with_columns(_w=w, _m=(pl.col(col) >= MEMBER_P).cast(pl.Int32))
    F = F.with_columns(_W=pl.col("_w").sum().over("s1_id"), _M=pl.col("_m").sum().over("s1_id"))
    new = [(pl.col("_W") - pl.col("_w")).alias(prefix + "mass")]
    for k, name in KEYS:
        wk = pl.col("_w").sum().over("s1_id", k)
        mk = pl.col("_m").sum().over("s1_id", k)
        eq_q = (pl.col(k) == pl.col(k + "q")).fill_null(False).cast(pl.Float32)
        share = (wk - pl.col("_w") + eq_q) / (pl.col("_W") - pl.col("_w") + 1.0)
        new += [pl.when(pl.col(k).is_null()).then(None).otherwise(share).alias(f"{prefix}{name}_share"),
                pl.when(pl.col(k).is_null()).then(None).otherwise(mk - pl.col("_m")).alias(f"{prefix}{name}_n")]
    F = F.with_columns(new)
    return F.drop([k for k, _ in KEYS] + [k + "q" for k, _ in KEYS] + ["_w", "_m", "_W", "_M"])


def add_context(F, col, prefix):
    """Rank/gap context of a probability column within the S1 entity and the record."""
    p = pl.col(col)
    F = F.with_columns(
        (p.rank("ordinal", descending=True).over("s1_id")).cast(pl.UInt32).alias(prefix + "rank_q"),
        p.sum().over("s1_id").alias(prefix + "sum_q"),
        p.max().over("s1_id").alias(prefix + "max_q"),
        (p.rank("ordinal", descending=True).over("r_id")).cast(pl.UInt32).alias(prefix + "rank_r"),
        p.max().over("r_id").alias(prefix + "max_r"),
    )
    second = p.filter(pl.col(prefix + "rank_r") == 2).max().over("r_id").fill_null(0)
    other = pl.when(pl.col(prefix + "rank_r") == 1).then(second).otherwise(pl.col(prefix + "max_r"))
    return F.with_columns((p - other).alias(prefix + "gap_r"), (pl.col(prefix + "max_q") - p).alias(prefix + "gap_q"))
