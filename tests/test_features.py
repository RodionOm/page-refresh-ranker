import numpy as np
import pandas as pd

from ranker.features import BANNED, FEATURES, build_features, label_from_export
from ranker.features import windows_from_export
from tests.conftest import make_export


def test_23_features_in_fixed_order(export_df):
    X = build_features(export_df)
    assert list(X.columns) == FEATURES
    assert len(FEATURES) == 23


def test_no_leaky_columns():
    assert not set(FEATURES) & BANNED


def test_older_window_is_rebuilt_by_subtraction():
    raw = make_export(50)
    df = windows_from_export(raw)
    assert (df["impressions_older_30d"] == raw["impressions_90d"] - raw["impressions_last_30d"]
            - raw["impressions_prev_30d"]).all()


def test_label_window_does_not_change_features():
    """Change only the last 30 days (keeping the 90-day total consistent): features must not move."""
    raw = make_export(50)
    changed = raw.copy()
    for m in ("impressions", "clicks", "sessions"):
        changed[f"{m}_90d"] = raw[f"{m}_90d"] - raw[f"{m}_last_30d"]   # same older + prev ...
        changed[f"{m}_last_30d"] = 0                                     # ... but a different label window
    pd.testing.assert_frame_equal(build_features(windows_from_export(raw)),
                                  build_features(windows_from_export(changed)))


def test_missing_values_get_flags(export_df):
    X = build_features(export_df)
    missing = export_df["word_count"].isna()
    assert (X.loc[missing, "has_word_count"] == 0).all()
    assert (X.loc[missing, "word_count"] == 0).all()


def test_one_row_gives_the_same_columns(export_df):
    """Training/serving skew guard: a single page must produce all 23 columns."""
    X1 = build_features(export_df.head(1))
    assert list(X1.columns) == FEATURES
    assert X1.filter(like="main_intent_").sum(axis=1).iloc[0] == 1


def test_label_is_down_only():
    df = pd.DataFrame({"trend_direction": ["down", "stable", "up", "new", "flat"]})
    assert label_from_export(df).tolist() == [1, 0, 0, 0, 0]


def test_zero_impressions_ctr_is_zero(export_df):
    df = export_df.head(3).copy()
    df["impressions_prev_30d"] = 0
    assert (build_features(df)["ctr_prev_30d"] == 0).all()
    assert np.isfinite(build_features(df).to_numpy()).all()
