from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAIN_PATH = Path("data/training/train.csv")
DEFAULT_TEST_PATH = Path("data/training/test.csv")
DEFAULT_MODEL_PATH = Path("models/stroke_model.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics.json")
DEFAULT_REPORT_PATH = Path("reports/classification_report.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo de prediccion de ACV.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def build_pipeline(df_train: pd.DataFrame) -> Pipeline:
    feature_df = df_train.drop(columns=["stroke"])
    numeric_features = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], str]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    return metrics, report


def main() -> None:
    args = parse_args()
    df_train = pd.read_csv(args.train_path)
    df_test = pd.read_csv(args.test_path)

    X_train = df_train.drop(columns=["stroke"])
    y_train = df_train["stroke"]
    X_test = df_test.drop(columns=["stroke"])
    y_test = df_test["stroke"]

    model = build_pipeline(df_train)
    model.fit(X_train, y_train)
    metrics, report = evaluate_model(model, X_test, y_test)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_path)
    with args.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with args.report_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print("Entrenamiento completado correctamente.")
    print(f"Modelo guardado en: {args.model_path}")
    print(f"Metricas guardadas en: {args.metrics_path}")
    print(f"Reporte de clasificacion guardado en: {args.report_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

