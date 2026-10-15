"""Train the ranker and save it.

    python train.py --data path/to/content_refresh_anonymized.csv --holdout 0.2
    python train.py --synthetic 3000              # demo model for a public deployment

1. Grouped cross-validation (GroupKFold by client) on the training clients -> honest metrics.
2. Optional holdout: a share of CLIENTS is kept out completely. The model is scored on them and
   their pages are written to data/holdout_pages.csv, ready to upload on the website.
3. The final model is fitted on the training clients and saved to models/model.joblib.

The FlyRank data file is not part of this repository and stays under the programme's rules.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from ranker import model as mdl
from ranker.features import (BANNED, FEATURES, build_features, label_from_export, to_prediction_format,
                             windows_from_export)
from ranker.review_queue import build_queue
from ranker.synthetic import make_export


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="content_refresh_anonymized.csv")
    src.add_argument("--synthetic", type=int, metavar="N", help="train on N synthetic pages (public demo)")
    ap.add_argument("--holdout", type=float, default=0.0, help="share of clients kept out, e.g. 0.2")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    export = make_export(args.synthetic, seed=7) if args.synthetic else pd.read_csv(args.data)
    df = windows_from_export(export)
    assert (df["impressions_older_30d"] >= 0).all(), "older window negative: check the export"

    holdout = None
    if args.holdout > 0:
        split = GroupShuffleSplit(n_splits=1, test_size=args.holdout, random_state=42)
        train_idx, hold_idx = next(split.split(df, groups=df["client_id"]))
        holdout, df = df.iloc[hold_idx].reset_index(drop=True), df.iloc[train_idx].reset_index(drop=True)

    X, y, groups = build_features(df), label_from_export(df), df["client_id"]
    assert not (set(X.columns) & BANNED), "a leaky column reached the features"
    metrics = {"trained_on": "synthetic" if args.synthetic else "flyrank_starter", **mdl.grouped_cv(X, y, groups)}
    print("grouped CV on training clients:", metrics)

    model = mdl.train(X, y)
    if holdout is not None:
        Xh, yh = build_features(holdout), label_from_export(holdout)
        s = model.predict_proba(Xh)[:, 1]
        q = build_queue(pd.DataFrame({"content_id": holdout["content_id"], "client_id": holdout["client_id"],
                                      "score": s, "y": yh}), k=50, per_client=5)
        metrics["holdout"] = {"clients": int(holdout["client_id"].nunique()), "pages": int(len(holdout)),
                              "auc": round(float(roc_auc_score(yh, s)), 3),
                              "precision_at_50": round(mdl.precision_at_k(yh, s, 50), 2),
                              "queue_precision_capped": round(float(q["y"].mean()), 2),
                              "base_rate": round(float(yh.mean()), 3)}
        print("holdout clients (never seen):", metrics["holdout"])
        os.makedirs("data", exist_ok=True)
        to_prediction_format(export.loc[export["client_id"].isin(holdout["client_id"])]) \
            .to_csv("data/holdout_pages.csv", index=False)
        print("wrote data/holdout_pages.csv: upload it on the website to see the queue for unseen clients")

    os.makedirs(args.out, exist_ok=True)
    mdl.save(model, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({**metrics, "features": FEATURES}, f, indent=2)
    print(f"saved {args.out}/model.joblib and {args.out}/metrics.json")


if __name__ == "__main__":
    main()
