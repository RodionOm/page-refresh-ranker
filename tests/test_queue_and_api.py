import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app import db
from ranker.features import to_prediction_format
from ranker.review_queue import build_queue
from ranker.synthetic import make_export
from tests.conftest import page_payload


@pytest.fixture()
def client(model_file, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", model_file)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    main._state.clear()
    return TestClient(main.app)


def csv_bytes(n=60, seed=5) -> bytes:
    df = to_prediction_format(make_export(n, seed=seed))
    return df.to_csv(index=False).encode()


def upload(client, content: bytes, k=10, per_client=3):
    return client.post("/api/runs", files={"file": ("pages.csv", content, "text/csv")},
                       data={"k": str(k), "per_client": str(per_client)})


# ---------- queue logic ----------

def test_queue_caps_pages_per_client():
    scored = pd.DataFrame({"content_id": [f"p{i}" for i in range(20)],
                           "client_id": ["big"] * 15 + ["small"] * 5,
                           "score": [1 - i / 100 for i in range(20)]})
    q = build_queue(scored, k=8, per_client=5)
    assert len(q) == 8
    assert q["client_id"].value_counts().max() <= 5
    assert q["rank"].tolist() == list(range(1, 9))
    assert q["score"].is_monotonic_decreasing


# ---------- JSON API ----------

def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True


def test_score_returns_band_reasons_flags(client):
    r = client.post("/api/score", json={"pages": [page_payload(0), page_payload(1, content_age_days=60)]})
    assert r.status_code == 200
    a, b = r.json()
    assert 0 <= a["score"] <= 1 and a["band"] in {"low", "medium", "high"}
    assert len(a["reasons"]) <= 2
    assert "younger than any page in the training data (90 days)" in b["review_flags"]


def test_queue_endpoint_caps_clients(client):
    pages = [page_payload(i, client="big" if i < 12 else "small") for i in range(15)]
    rows = client.post("/api/queue", json={"pages": pages, "k": 10, "per_client": 5}).json()
    assert [x["rank"] for x in rows] == list(range(1, len(rows) + 1))
    assert sum(x["client_id"] == "big" for x in rows) <= 5


@pytest.mark.parametrize("bad", [{"impressions_prev_30d": -1}, {"content_type": "video"}, {"competition": 2}])
def test_bad_json_is_rejected_with_422(client, bad):
    assert client.post("/api/score", json={"pages": [page_payload(0, **bad)]}).status_code == 422


def test_missing_model_gives_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "none.joblib"))
    main._state.clear()
    assert TestClient(main.app).post("/api/score", json={"pages": [page_payload(0)]}).status_code == 503


# ---------- runs: upload, store, feedback ----------

def test_upload_creates_a_saved_run(client):
    r = upload(client, csv_bytes())
    assert r.status_code == 200
    run = r.json()
    assert run["n_pages"] == 60 and run["k"] == 10 and run["per_client"] == 3
    assert len(run["results"]) == 10
    assert pd.Series([x["client_id"] for x in run["results"]]).value_counts().max() <= 3
    assert run["summary"]["precision_observed"] is not None      # the demo file has an outcome column
    listed = client.get("/api/runs").json()
    assert [x["id"] for x in listed] == [run["id"]]


def test_feedback_is_upserted_and_summarised(client):
    run = upload(client, csv_bytes()).json()
    page = run["results"][0]["content_id"]
    s1 = client.post(f"/api/runs/{run['id']}/feedback", json={"content_id": page, "declined": True}).json()
    s2 = client.post(f"/api/runs/{run['id']}/feedback", json={"content_id": page, "declined": False}).json()
    assert (s1["reviewed"], s1["reviewed_declined"]) == (1, 1)
    assert (s2["reviewed"], s2["reviewed_declined"], s2["precision_from_feedback"]) == (1, 0, 0.0)


def test_feedback_for_unknown_page_is_404(client):
    run = upload(client, csv_bytes()).json()
    r = client.post(f"/api/runs/{run['id']}/feedback", json={"content_id": "nope", "declined": True})
    assert r.status_code == 404


def test_csv_errors_point_to_the_row(client):
    df = to_prediction_format(make_export(5, seed=1))
    df.loc[2, "impressions_prev_30d"] = -3
    r = upload(client, df.to_csv(index=False).encode())
    assert r.status_code == 422
    assert "row 4" in r.json()["detail"][0]          # header is row 1, so index 2 is row 4


def test_csv_missing_column_is_rejected(client):
    df = to_prediction_format(make_export(5, seed=1)).drop(columns=["clicks_prev_30d"])
    r = upload(client, df.to_csv(index=False).encode())
    assert r.status_code == 422 and "clicks_prev_30d" in r.json()["detail"]


def test_delete_run_cascades(client):
    run = upload(client, csv_bytes()).json()
    client.post(f"/api/runs/{run['id']}/feedback", json={"content_id": run["results"][0]["content_id"], "declined": True})
    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_website_and_template_are_served(client):
    assert "Page Refresh Ranker" in client.get("/").text
    assert client.get("/api/template.csv").text.startswith("content_id,client_id,")
    assert client.get("/demo_pages.csv").status_code == 200
