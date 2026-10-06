"""Label-free estimate of macro F0.5 from calibrated pair probabilities:
E[F] per entity ~= 1.25*sum(p in predicted set) / (|set| + 0.25*sum(all p)), and for an
empty prediction prod(1-p). Checked on held-out entities, then applied per test country."""
import sys

import polars as pl

import s05_train as S
from config import WORK_DIR, norm_path


def expected_f(P, sel, ids):
    tot = P.group_by("s1_id").agg(tot=pl.col("p").sum(), none=(1 - pl.col("p")).clip(1e-9, 1).log().sum().exp())
    inset = sel.join(P, on=["s1_id", "r_id"]).group_by("s1_id").agg(k=pl.len(), tp=pl.col("p").sum())
    E = (pl.DataFrame({"s1_id": ids}).join(tot, on="s1_id", how="left").join(inset, on="s1_id", how="left")
           .fill_null(0).with_columns(none=pl.when(pl.col("tot") == 0).then(1.0).otherwise(pl.col("none"))))
    E = E.with_columns(ef=pl.when(pl.col("k") == 0).then(pl.col("none"))
                         .otherwise(1.25 * pl.col("tp") / (pl.col("k") + 0.25 * pl.col("tot"))))
    return E


if __name__ == "__main__":
    if sys.argv[1] == "heldout":
        ev = pl.read_parquet(WORK_DIR / "eval_pred.parquet").filter(pl.col("fold").is_in(list(S.REPORT)))
        sel = pl.read_parquet(WORK_DIR / "report_pred.parquet")
        per = pl.read_parquet(WORK_DIR / "report_per_entity.parquet")
        E = expected_f(ev.select("s1_id", "r_id", "p"), sel, per["s1_id"])
        print(f"held-out: expected F {E['ef'].mean():.5f} vs actual {per['f'].mean():.5f}")
    else:
        P = pl.read_parquet(WORK_DIR / "test_scores.parquet")
        m = pl.read_csv(S.WORK_DIR.parent / "output" / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
        sel = (m.drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(","))
                .explode("matched_entity_ids").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"}))
        s1 = pl.read_parquet(norm_path("test", "1"), columns=["entity_id", "country"])
        E = expected_f(P, sel, s1["entity_id"]).join(s1.rename({"entity_id": "s1_id"}), on="s1_id")
        print(f"test overall expected F {E['ef'].mean():.5f}")
        print(E.group_by("country").agg(pl.col("ef").mean().round(5), pl.len()).sort("country"))
