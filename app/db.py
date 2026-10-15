"""SQLite storage for review runs, their ranked results and reviewer feedback. Plain SQL, parameterised."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "data/app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    n_pages     INTEGER NOT NULL CHECK (n_pages > 0),
    k           INTEGER NOT NULL,
    per_client  INTEGER NOT NULL,
    base_rate_observed REAL                -- share of ALL uploaded pages that declined, if the file has the outcome
);
CREATE TABLE IF NOT EXISTS results (
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rank        INTEGER NOT NULL,
    content_id  TEXT    NOT NULL,
    client_id   TEXT    NOT NULL,
    score       REAL    NOT NULL,
    band        TEXT    NOT NULL,
    reasons     TEXT    NOT NULL,          -- JSON list
    flags       TEXT    NOT NULL,          -- JSON list
    observed    INTEGER,                   -- 1/0 if the upload had declined_observed, else NULL
    PRIMARY KEY (run_id, content_id)
);
CREATE TABLE IF NOT EXISTS feedback (
    run_id      INTEGER NOT NULL,
    content_id  TEXT    NOT NULL,
    declined    INTEGER NOT NULL CHECK (declined IN (0, 1)),
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (run_id, content_id),
    FOREIGN KEY (run_id, content_id) REFERENCES results(run_id, content_id) ON DELETE CASCADE
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_ready: set[str] = set()


def _migrate(con) -> None:
    """Databases made by version 2.0 have no base_rate_observed column: add it instead of failing."""
    cols = {row["name"] for row in con.execute("PRAGMA table_info(runs)")}
    if "base_rate_observed" not in cols:
        con.execute("ALTER TABLE runs ADD COLUMN base_rate_observed REAL")


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if DB_PATH not in _ready:                 # create tables on first use of this database file
        con.executescript(SCHEMA)
        _migrate(con)
        _ready.add(DB_PATH)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def create_run(filename: str, n_pages: int, k: int, per_client: int, rows: list[dict],
               base_rate_observed: float | None = None) -> int:
    """One transaction: the run and all its result rows are saved together or not at all."""
    with connect() as con:
        cur = con.execute("INSERT INTO runs (created_at, filename, n_pages, k, per_client, base_rate_observed) "
                          "VALUES (?, ?, ?, ?, ?, ?)", (_now(), filename, n_pages, k, per_client, base_rate_observed))
        run_id = cur.lastrowid
        con.executemany(
            "INSERT INTO results (run_id, rank, content_id, client_id, score, band, reasons, flags, observed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, r["rank"], r["content_id"], r["client_id"], r["score"], r["band"],
              json.dumps(r["reasons"]), json.dumps(r["review_flags"]), r.get("observed")) for r in rows])
        return run_id


def list_runs() -> list[dict]:
    with connect() as con:
        rows = con.execute("""
            SELECT r.id, r.created_at, r.filename, r.n_pages, r.k, r.per_client,
                   COUNT(f.content_id) AS reviewed,
                   (SELECT COUNT(*) FROM results x WHERE x.run_id = r.id) AS queue_size
            FROM runs r
            LEFT JOIN feedback f ON f.run_id = r.id
            GROUP BY r.id
            ORDER BY r.id DESC
        """).fetchall()
        return [dict(x) for x in rows]


def get_run(run_id: int) -> dict | None:
    with connect() as con:
        run = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        rows = con.execute("""
            SELECT res.*, f.declined AS feedback
            FROM results res
            LEFT JOIN feedback f ON f.run_id = res.run_id AND f.content_id = res.content_id
            WHERE res.run_id = ?
            ORDER BY res.rank
        """, (run_id,)).fetchall()
    results = []
    for x in rows:
        d = dict(x)
        d["reasons"], d["review_flags"] = json.loads(d.pop("reasons")), json.loads(d.pop("flags"))
        d.pop("run_id")
        results.append(d)
    run = dict(run)
    return {**run, "results": results, "summary": summarise(results, run["base_rate_observed"])}


def summarise(results: list[dict], base_rate: float | None = None) -> dict:
    reviewed = [r["feedback"] for r in results if r["feedback"] is not None]
    observed = [r["observed"] for r in results if r["observed"] is not None]
    precision_observed = round(sum(observed) / len(observed), 2) if observed else None
    return {
        "queue_size": len(results),
        "reviewed": len(reviewed),
        "reviewed_declined": sum(reviewed),
        "precision_from_feedback": round(sum(reviewed) / len(reviewed), 2) if reviewed else None,
        "precision_observed": precision_observed,
        "file_base_rate": None if base_rate is None else round(base_rate, 2),
        "lift": round(precision_observed / base_rate, 2) if precision_observed is not None and base_rate else None,
    }


def set_feedback(run_id: int, content_id: str, declined: bool) -> bool:
    """Upsert: a reviewer can change their answer. Returns False if the page is not in this run."""
    with connect() as con:
        exists = con.execute("SELECT 1 FROM results WHERE run_id = ? AND content_id = ?",
                             (run_id, content_id)).fetchone()
        if not exists:
            return False
        con.execute("""
            INSERT INTO feedback (run_id, content_id, declined, updated_at) VALUES (?, ?, ?, ?)
            ON CONFLICT (run_id, content_id) DO UPDATE SET declined = excluded.declined,
                                                          updated_at = excluded.updated_at
        """, (run_id, content_id, int(declined), _now()))
        return True


def delete_run(run_id: int) -> bool:
    with connect() as con:
        return con.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount == 1
