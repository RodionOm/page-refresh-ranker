"""Synthetic data with the same columns as the starter export, so tests need no real data."""
import pytest

from ranker import model as mdl
from ranker.features import build_features, label_from_export, windows_from_export
from ranker.synthetic import make_export  # noqa: F401  (re-exported for tests)


@pytest.fixture(scope="session")
def export_df():
    return windows_from_export(make_export())


@pytest.fixture(scope="session")
def trained(export_df):
    X, y = build_features(export_df), label_from_export(export_df)
    return mdl.train(X, y)


@pytest.fixture()
def model_file(tmp_path, trained):
    path = tmp_path / "model.joblib"
    mdl.save(trained, str(path))
    return str(path)


def page_payload(i: int = 0, client: str = "c1", **over) -> dict:
    p = {"content_id": f"p{i}", "client_id": client,
         "impressions_older_30d": 1200, "impressions_prev_30d": 3000,
         "clicks_older_30d": 30, "clicks_prev_30d": 20,
         "sessions_older_30d": 25, "sessions_prev_30d": 18,
         "content_age_days": 400, "content_type": "keyword article", "main_intent": "informational",
         "word_count": 1800, "search_volume": 900, "competition": 0.4, "cpc": 1.2}
    p.update(over)
    return p
