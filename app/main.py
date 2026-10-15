"""Web app around the ranker: a JSON API under /api and a single-page site at /.

GET    /api/health                   is the model loaded, what it was trained on
POST   /api/score                    score pages sent as JSON
POST   /api/queue                    top-k review queue from pages sent as JSON
POST   /api/runs                     upload a CSV -> score, build the queue, save it as a run
GET    /api/runs                     list saved runs
GET    /api/runs/{id}                one run with its queue and summary
POST   /api/runs/{id}/feedback       reviewer marks a queued page as declined or not
DELETE /api/runs/{id}                delete a run
GET    /api/template.csv             empty CSV with the expected columns
"""
import io
import json
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from app import db
from ranker import model as mdl
from ranker.features import build_features
from ranker.review_queue import build_queue, review_flags
from ranker.schemas import Page, QueueRequest, QueueRow, ScoredPage, ScoreRequest

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MAX_UPLOAD_ROWS = 20_000
PAGE_COLUMNS = list(Page.model_fields)

app = FastAPI(title="Page refresh ranker", version="2.0.0")
_state: dict = {}


def get_model():
    if "model" not in _state:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail=f"model not found at {MODEL_PATH}; run train.py")
        _state["model"] = mdl.load(MODEL_PATH)
    return _state["model"]


def model_info() -> dict:
    path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        m = json.load(f)
    return {k: m.get(k) for k in ("trained_on", "auc", "precision_at_50", "pages", "clients", "holdout")}


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    X = build_features(raw)
    model = get_model()
    out = raw[["content_id", "client_id"]].copy()
    out["score"] = model.predict_proba(X)[:, 1].round(3)
    out["band"] = [mdl.band(s) for s in out["score"]]
    out["reasons"] = mdl.explain(model, X)
    out["review_flags"] = review_flags(raw, X)
    return out


def pages_frame(pages: list[Page]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump() for p in pages])


# ---------- JSON API ----------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH), "model": model_info()}


@app.post("/api/score", response_model=list[ScoredPage])
def score(req: ScoreRequest):
    return score_frame(pages_frame(req.pages)).to_dict(orient="records")


@app.post("/api/queue", response_model=list[QueueRow])
def queue(req: QueueRequest):
    scored = score_frame(pages_frame(req.pages))
    return build_queue(scored, k=req.k, per_client=req.per_client).to_dict(orient="records")


@app.get("/api/template.csv", response_class=PlainTextResponse)
def template() -> str:
    return ",".join(PAGE_COLUMNS + ["declined_observed"]) + "\n"


def parse_csv(content: bytes) -> pd.DataFrame:
    """Validate every row with the same Page schema the JSON API uses; report the first errors by row."""
    try:
        df = pd.read_csv(io.BytesIO(content), dtype={"content_id": str, "client_id": str})
    except Exception:
        raise HTTPException(status_code=422, detail="could not read the file as CSV")
    missing = [c for c in PAGE_COLUMNS if c not in df.columns and Page.model_fields[c].is_required()]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {', '.join(missing)}")
    if df.empty:
        raise HTTPException(status_code=422, detail="the file has no rows")
    if len(df) > MAX_UPLOAD_ROWS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_UPLOAD_ROWS} rows per upload")
    errors, pages = [], []
    for i, row in enumerate(df.to_dict(orient="records"), start=2):          # row 1 is the header
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items() if k in PAGE_COLUMNS}
        try:
            pages.append(Page(**clean))
        except ValidationError as e:
            first = e.errors()[0]
            errors.append(f"row {i}, {first['loc'][0]}: {first['msg']}")
            if len(errors) == 5:
                break
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    out = pages_frame(pages)
    if "declined_observed" in df.columns:
        out["observed"] = df["declined_observed"].astype("Int64").tolist()
    return out


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...), k: int = Form(50), per_client: int = Form(5)) -> dict:
    if not (1 <= k <= 500 and 1 <= per_client <= 50):
        raise HTTPException(status_code=422, detail="k must be 1-500 and per_client 1-50")
    raw = parse_csv(await file.read())
    scored = score_frame(raw)
    if "observed" in raw.columns:
        scored["observed"] = [None if pd.isna(v) else int(v) for v in raw["observed"]]
    q = build_queue(scored, k=k, per_client=per_client)
    run_id = db.create_run(file.filename or "upload.csv", len(raw), k, per_client, q.to_dict(orient="records"))
    return db.get_run(run_id)


@app.get("/api/runs")
def runs() -> list[dict]:
    return db.list_runs()


@app.get("/api/runs/{run_id}")
def run(run_id: int) -> dict:
    found = db.get_run(run_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such run")
    return found


class Feedback(BaseModel):
    content_id: str
    declined: bool


@app.post("/api/runs/{run_id}/feedback")
def feedback(run_id: int, body: Feedback) -> dict:
    if not db.set_feedback(run_id, body.content_id, body.declined):
        raise HTTPException(status_code=404, detail="page not in this run")
    return db.get_run(run_id)["summary"]


@app.delete("/api/runs/{run_id}", status_code=204)
def delete(run_id: int) -> None:
    if not db.delete_run(run_id):
        raise HTTPException(status_code=404, detail="no such run")


# ---------- website ----------
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="site")
