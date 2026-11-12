import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC, TARGET, add_features


def build(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def main():
    df = add_features(pd.read_csv("data/bids.csv"))
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "logistic_regression": build(LogisticRegression(max_iter=1000)),
        "gradient_boosting": build(CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42),
            method="isotonic", cv=3)),
    }

    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        results[name] = {
            "roc_auc": round(roc_auc_score(yte, p), 4),
            "pr_auc": round(average_precision_score(yte, p), 4),
            "brier": round(brier_score_loss(yte, p), 4),
        }
        fitted[name] = m
        print(name, results[name])

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print("Best model:", best)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(fitted[best], "models/model.joblib")

    imp = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
    (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
       .sort_values("importance", ascending=False)
       .to_csv("models/importance.csv", index=False))

    json.dump({"best": best, "base_rate": round(float(y.mean()), 4), "results": results},
              open("models/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()