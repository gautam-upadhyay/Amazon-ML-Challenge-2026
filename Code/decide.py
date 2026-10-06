"""Turn pair probabilities into per-entity match sets.

1. One-to-one: every S2/S3 record belongs to at most one S1 entity (holds for all
   7.6M training assignments), so each record is kept only for its best S1 entity.
2. Expected-F0.5 set selection: per entity F0.5 = 1.25*TP / (|pred| + 0.25*|true|).
   For each prefix of the probability-sorted candidates we estimate
   E[F] ~= 1.25*sum(p_top_k) / (k + 0.25*(sum(p_all) + miss)), and compare it
   with predicting nothing, which scores 1 exactly when no candidate is a match
   (probability prod(1 - p)).
"""
import polars as pl


def one_to_one(P, col="p", margin=0.0):
    best = pl.col(col).max().over("r_id")
    return P.filter(pl.col(col) >= best - margin)


def select(P, col="p", miss=0.0, none_bias=1.0, min_p=0.0):
    P = P.filter(pl.col(col) > min_p).sort(["s1_id", col], descending=[False, True])
    P = P.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s1_id"),
        cum=pl.col(col).cum_sum().over("s1_id"),
        tot=pl.col(col).sum().over("s1_id") + miss,
        ef0=(1 - pl.col(col)).clip(1e-9, 1).log().sum().over("s1_id").exp() * none_bias,
    ).with_columns(ef=1.25 * pl.col("cum") / (pl.col("k") + 0.25 * pl.col("tot")))
    P = P.with_columns(ef_best=pl.col("ef").max().over("s1_id"))
    kbest = pl.col("k").filter(pl.col("ef") == pl.col("ef_best")).min().over("s1_id")
    P = P.with_columns(kbest=kbest)
    return P.filter((pl.col("ef_best") > pl.col("ef0")) & (pl.col("k") <= pl.col("kbest"))).select("s1_id", "r_id")


def threshold(P, t, col="p"):
    return P.filter(pl.col(col) >= t).select("s1_id", "r_id")
