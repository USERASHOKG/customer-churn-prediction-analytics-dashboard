"""Reusable training and inference utilities for Telco customer churn."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known quirks in the public IBM Telco dataset."""
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    if TARGET in data.columns:
        data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0}).astype("Int64")
    return data


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = clean_data(df).dropna(subset=[TARGET])
    X = data.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in data.columns])
    y = data[TARGET].astype(int)
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(X)
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", RandomForestClassifier(n_estimators=350, max_depth=12, min_samples_leaf=3,
                                                   class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", make_preprocessor(X)),
            ("classifier", GradientBoostingClassifier(n_estimators=180, learning_rate=0.06,
                                                       max_depth=3, random_state=42)),
        ]),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Return cross-validated metrics for all candidate classifiers."""
    scoring = {"precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict[str, Any]] = []
    for name, model in build_models(X).items():
        fit_params = {}
        if name == "Gradient Boosting":
            fit_params["classifier__sample_weight"] = compute_sample_weight("balanced", y)
        result = cross_validate(model, X, y, cv=cv, scoring=scoring, params=fit_params,
                                return_train_score=False, n_jobs=-1)
        rows.append({"model": name, **{metric: float(np.mean(result[f"test_{metric}"])) for metric in scoring}})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def train_best_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    X, y = split_features_target(df)
    metrics = evaluate_models(X, y)
    best_name = str(metrics.iloc[0]["model"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_models(X)[best_name]
    fit_params = {}
    if best_name == "Gradient Boosting":
        fit_params["classifier__sample_weight"] = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, **fit_params)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    holdout = {
        "model": best_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
        "churn_rate": float(y.mean()),
    }
    return model, metrics, holdout


def save_artifacts(model: Pipeline, metrics: pd.DataFrame, holdout: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "churn_model.joblib")
    metrics.to_csv(path / "model_metrics.csv", index=False)
    (path / "model_metadata.json").write_text(json.dumps(holdout, indent=2))


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """Extract model-native importance values after one-hot encoding."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        values = np.abs(classifier.coef_[0])
    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)


def predict_customer(model: Pipeline, customer: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame([customer])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"probability": probability, "prediction": int(probability >= 0.5),
            "label": "High churn risk" if probability >= 0.5 else "Low churn risk"}
