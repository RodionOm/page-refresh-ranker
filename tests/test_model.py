import pytest

from ranker import model as mdl
from ranker.features import build_features, label_from_export


def test_grouped_cv_reports_metrics(export_df):
    m = mdl.grouped_cv(build_features(export_df), label_from_export(export_df), export_df["client_id"])
    assert set(m) == {"auc", "precision_at_50", "precision_at_500", "base_rate", "pages", "clients"}
    assert m["auc"] > 0.6          # the synthetic label has a real signal
    assert m["clients"] == 12


def test_precision_at_k():
    assert mdl.precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == 0.5


@pytest.mark.parametrize("score, expected", [(0.71, "high"), (0.7, "medium"), (0.51, "medium"), (0.5, "low")])
def test_bands(score, expected):
    assert mdl.band(score) == expected


def test_reasons_are_at_most_two_and_readable(trained, export_df):
    reasons = mdl.explain(trained, build_features(export_df.head(20)))
    assert all(len(r) <= 2 for r in reasons)
    assert any("visibility" in " ".join(r) for r in reasons)


def test_save_and_load_roundtrip(trained, model_file, export_df):
    X = build_features(export_df.head(5))
    assert (mdl.load(model_file).predict_proba(X) == trained.predict_proba(X)).all()


def test_load_rejects_other_feature_list(tmp_path, trained):
    import joblib
    path = tmp_path / "old.joblib"
    joblib.dump({"model": trained, "features": ["something_else"]}, path)
    with pytest.raises(ValueError):
        mdl.load(str(path))


def test_every_reason_is_plain_text(trained, export_df):
    for row in mdl.explain(trained, build_features(export_df)):
        for text in row:
            assert "_" not in text


def test_reason_direction_tracks_actual_feature_value():
    assert mdl.reason_text('log_impressions_prev_30d', -2).startswith('below')
    assert mdl.reason_text('log_impressions_prev_30d', 2).startswith('above')
    assert mdl.reason_text('log_clicks_prev_30d', 2).startswith('above')
