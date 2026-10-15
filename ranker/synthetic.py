"""Synthetic pages with the same columns as the starter export: for tests and the public demo."""
import numpy as np
import pandas as pd


def make_export(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    older = rng.integers(0, 5000, n)
    prev = rng.integers(0, 5000, n)
    last = rng.integers(0, 5000, n)
    df = pd.DataFrame({
        "content_id": [f"p{i}" for i in range(n)],
        "client_id": [f"c{i % 12}" for i in range(n)],
        "impressions_prev_30d": prev, "impressions_last_30d": last, "impressions_90d": older + prev + last,
        "clicks_prev_30d": prev // 50, "clicks_last_30d": last // 50, "clicks_90d": (older + prev + last) // 50,
        "sessions_prev_30d": prev // 60, "sessions_last_30d": last // 60, "sessions_90d": (older + prev + last) // 60,
        "content_age_days": rng.integers(90, 2000, n),
        "word_count": np.where(rng.random(n) < 0.2, np.nan, rng.integers(300, 4000, n)),
        "search_volume": np.where(rng.random(n) < 0.3, np.nan, rng.integers(10, 9000, n)),
        "competition": rng.random(n),
        "cpc": rng.random(n) * 3,
        "content_type": rng.choice(["comparison article", "feedly article", "keyword article"], n),
        "main_intent": rng.choice(["commercial", "informational", "navigational", "transactional", None], n),
    })
    # label with a real signal: pages with a lot of recent traffic tend to "decline"
    p = 1 / (1 + np.exp(-(np.log1p(prev) - 7)))
    df["trend_direction"] = np.where(rng.random(n) < p, "down", "stable")
    return df
