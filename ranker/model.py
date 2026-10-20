"""Train, evaluate, save and use the logistic-regression ranker."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

REASON_TEXT = {
    "log_impressions_prev_30d": "high recent visibility (lots to lose)",
    "log_impressions_older_30d": "strong older traffic base",
    "early_momentum": "recent spike (spikes often fall back)",
    "content_age_days": "younger page, still settling",
    "log_clicks_prev_30d": "few clicks for its impressions",
    "log_clicks_older_30d": "few clicks historically",
    "ctr_prev_30d": "low click-through rate",
}


# Readable names for features without a hand-written reason; the word high/low comes from the data.
FEATURE_LABELS = {
    "log_sessions_prev_30d": "recent visits", "log_sessions_older_30d": "visits in the older window",
    "word_count": "word count", "has_word_count": "word count recorded",
    "has_keyword_data": "keyword data", "log_search_volume": "keyword search volume",
    "competition": "keyword competition", "cpc": "keyword cost per click",
}


def reason_text(feature: str, z: float) -> str:
    """Plain-English reason. z is the standardised value, so we can say high or low."""
    if feature in REASON_TEXT:
        labels = {
            "log_impressions_prev_30d": "recent visibility",
            "log_impressions_older_30d": "older traffic",
            "early_momentum": "recent momentum",
            "content_age_days": "page age",
            "log_clicks_prev_30d": "recent clicks",
            "log_clicks_older_30d": "older clicks",
            "ctr_prev_30d": "click-through rate",
        }
        return f"{'above' if z > 0 else 'below'} training mean: {labels[feature]}"
    if feature.startswith("content_type_"):
        return f"content type: {feature[len('content_type_'):]}" if z > 0 else "not a " + feature[len("content_type_"):]
    if feature.startswith("main_intent_"):
        return f"search intent: {feature[len('main_intent_'):]}" if z > 0 else "intent not " + feature[len("main_intent_"):]
    return f"{'high' if z > 0 else 'low'} {FEATURE_LABELS.get(feature, feature.replace('_', ' '))}"


def new_model() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def precision_at_k(y, scores, k: int) -> float:
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.asarray(y)[top].mean())


def grouped_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> dict:
    """Out-of-fold scores where every page is scored by a model that never saw its client."""
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        oof[te] = new_model().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return {
        "auc": round(float(roc_auc_score(y, oof)), 3),
        "precision_at_50": round(precision_at_k(y, oof, 50), 2),
        "precision_at_500": round(precision_at_k(y, oof, 500), 2),
        "base_rate": round(float(y.mean()), 3),
        "pages": int(len(X)),
        "clients": int(groups.nunique()),
    }


def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    assert list(X.columns) == FEATURES, "feature columns out of order"
    return new_model().fit(X, y)


def save(model: Pipeline, path: str) -> None:
    joblib.dump({"model": model, "features": FEATURES}, path)


def load(path: str) -> Pipeline:
    bundle = joblib.load(path)
    if bundle["features"] != FEATURES:
        raise ValueError("model was trained on a different feature list")
    return bundle["model"]


def band(score: float) -> str:
    """Bands are ordered, not calibrated: in the internship data 'high' pages declined ~69% of the time."""
    return "high" if score > 0.7 else "medium" if score > 0.5 else "low"


def explain(model: Pipeline, X: pd.DataFrame, n: int = 2) -> list[list[str]]:
    """Top-n reasons per row: standardised value x coefficient, positive contributions only."""
    scaler, lr = model[0], model[-1]
    z = scaler.transform(X)
    contrib = z * lr.coef_[0]
    out = []
    for zi, row in zip(z, contrib):
        order = np.argsort(-row)[:n]
        out.append([reason_text(FEATURES[j], zi[j]) for j in order if row[j] > 0])
    return out
