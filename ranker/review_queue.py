"""Build the review queue: rank by score, cap pages per client, flag rows that need a human look."""
import pandas as pd


def build_queue(scored: pd.DataFrame, k: int = 50, per_client: int = 5) -> pd.DataFrame:
    """scored needs content_id, client_id, score. Returns at most k rows, at most per_client per client."""
    q = (scored.sort_values("score", ascending=False, kind="stable")
               .groupby("client_id", sort=False).head(per_client)
               .head(k)
               .reset_index(drop=True))
    q.insert(0, "rank", range(1, len(q) + 1))
    return q


def review_flags(raw: pd.DataFrame, X: pd.DataFrame) -> list[list[str]]:
    flags = []
    for i in range(len(raw)):
        r, f = raw.iloc[i], X.iloc[i]
        row = []
        if r["content_age_days"] < 90:
            row.append("younger than any page in the training data (90 days)")
        elif r["content_age_days"] <= 120:
            row.append("short history (<=120 days)")
        if f["early_momentum"] > 1:
            row.append("spike-driven (momentum > 1)")
        if pd.isna(r["search_volume"]):
            row.append("no keyword data")
        if r["impressions_prev_30d"] < 50:
            row.append("tiny traffic (<50 prev impressions)")
        flags.append(row)
    return flags
