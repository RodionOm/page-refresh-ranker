"""Turn raw page metrics into the 23 model features.

The model only sees two traffic windows that exist BEFORE the prediction point:
  older_30d = days 61-90 before the prediction point
  prev_30d  = days 31-60 before the prediction point
The most recent 30 days are the label window, so they never enter the features.
"""
import numpy as np
import pandas as pd

METRICS = ["impressions", "clicks", "sessions"]

# Fixed category lists, so training and serving always produce the same columns.
CONTENT_TYPES = ["comparison article", "feedly article", "keyword article"]
INTENTS = ["commercial", "informational", "navigational", "transactional", "unknown"]

FEATURES = (
    [f"log_{m}_{w}_30d" for m in METRICS for w in ("older", "prev")]
    + ["early_momentum", "ctr_prev_30d", "content_age_days", "has_word_count", "word_count",
       "has_keyword_data", "log_search_volume", "competition", "cpc"]
    + [f"content_type_{c}" for c in CONTENT_TYPES]
    + [f"main_intent_{i}" for i in INTENTS]
)

# Columns that leak the label. They must never be used as features.
BANNED = {"trend_direction", "trend_pct", "impressions_last_30d", "clicks_last_30d",
          "sessions_last_30d", "impressions_90d", "clicks_90d", "sessions_90d",
          "ctr", "avg_position", "days_since_last_update"}


def windows_from_export(df: pd.DataFrame) -> pd.DataFrame:
    """Map the starter CSV (90d / last / prev columns) to the two pre-prediction windows."""
    out = df.copy()
    for m in METRICS:
        out[f"{m}_older_30d"] = df[f"{m}_90d"] - df[f"{m}_last_30d"] - df[f"{m}_prev_30d"]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs {metric}_older_30d and {metric}_prev_30d plus page metadata."""
    X = pd.DataFrame(index=df.index)
    for m in METRICS:
        for w in ("older", "prev"):
            X[f"log_{m}_{w}_30d"] = np.log1p(df[f"{m}_{w}_30d"].astype(float))
    X["early_momentum"] = X["log_impressions_prev_30d"] - X["log_impressions_older_30d"]
    impr = df["impressions_prev_30d"].astype(float)
    X["ctr_prev_30d"] = np.where(impr > 0, df["clicks_prev_30d"] / impr.replace(0, 1) * 100, 0.0)
    X["content_age_days"] = df["content_age_days"].astype(float)
    X["has_word_count"] = df["word_count"].notna().astype(int)
    X["word_count"] = df["word_count"].fillna(0).astype(float)
    X["has_keyword_data"] = df["search_volume"].notna().astype(int)
    X["log_search_volume"] = np.log1p(df["search_volume"].fillna(0).astype(float))
    X["competition"] = df["competition"].fillna(0).astype(float)
    X["cpc"] = df["cpc"].fillna(0).astype(float)
    ctype = df["content_type"].fillna("unknown")
    intent = df["main_intent"].fillna("unknown")
    for c in CONTENT_TYPES:
        X[f"content_type_{c}"] = (ctype == c).astype(int)
    for i in INTENTS:
        X[f"main_intent_{i}"] = (intent == i).astype(int)
    return X[FEATURES]


def label_from_export(df: pd.DataFrame) -> pd.Series:
    """1 = impressions fell more than 20% in the last 30 days vs the previous 30 (trend_direction == 'down')."""
    return (df["trend_direction"] == "down").astype(int)


PREDICTION_COLUMNS = ["content_id", "client_id",
                      "impressions_older_30d", "impressions_prev_30d", "clicks_older_30d", "clicks_prev_30d",
                      "sessions_older_30d", "sessions_prev_30d", "content_age_days", "content_type", "main_intent",
                      "word_count", "search_volume", "competition", "cpc"]


def to_prediction_format(export: pd.DataFrame, with_outcome: bool = True) -> pd.DataFrame:
    """Starter export -> the CSV format the website accepts (optionally with the observed outcome)."""
    df = windows_from_export(export)
    out = df[PREDICTION_COLUMNS].copy()
    if with_outcome:
        out["declined_observed"] = label_from_export(df)
    return out
