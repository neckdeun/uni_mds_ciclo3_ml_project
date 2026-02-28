from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("data/training/train.csv")
TEST_PATH = Path("data/training/test.csv")
OUTPUT_CSV = Path("experiments/model_comparison.csv")
OUTPUT_MD = Path("experiments/model_comparison.md")


def build_preprocessor(df_train: pd.DataFrame) -> ColumnTransformer:
    X = df_train.drop(columns=["stroke"])
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
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


def evaluate_pipeline(
    name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float | str]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "modelo": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines: list[str] = []
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


def main() -> None:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train = df_train.drop(columns=["stroke"])
    y_train = df_train["stroke"]
    X_test = df_test.drop(columns=["stroke"])
    y_test = df_test["stroke"]

    preprocessor = build_preprocessor(df_train)

    models = [
        (
            "RegresionLogistica",
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                    ),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    results: list[dict[str, float | str]] = []
    for model_name, model_pipeline in models:
        results.append(evaluate_pipeline(model_name, model_pipeline, X_train, y_train, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    md_lines = [
        "# Comparacion de modelos",
        "",
        "Resultados sobre el conjunto de prueba:",
        "",
        dataframe_to_markdown_table(results_df),
        "",
        f"Modelo con mejor ROC-AUC: **{results_df.iloc[0]['modelo']}**",
    ]
    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("Experimentos ejecutados correctamente.")
    print(f"Tabla de resultados: {OUTPUT_CSV}")
    print(f"Resumen en Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

