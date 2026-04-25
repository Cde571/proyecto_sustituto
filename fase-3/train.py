"""Entrena un modelo de clasificación para la competición Predicting Heart Disease.

Este script recibe un archivo CSV con datos etiquetados, construye el mismo
pipeline utilizado en la fase 1, ajusta una regresión logística y guarda el
modelo entrenado en disco para su reutilización posterior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_TARGET_COLUMN = "Heart Disease"
DEFAULT_ID_COLUMN = "id"
DEFAULT_POSITIVE_LABEL = "Presence"
DEFAULT_MODEL_PATH = Path("models/model.joblib")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de enfermedad cardíaca y lo guarda en disco."
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        required=True,
        help="Ruta al archivo CSV de entrenamiento con la columna objetivo.",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta donde se guardará el modelo entrenado.",
    )
    parser.add_argument(
        "--target_column",
        default=DEFAULT_TARGET_COLUMN,
        help="Nombre de la columna objetivo.",
    )
    parser.add_argument(
        "--id_column",
        default=DEFAULT_ID_COLUMN,
        help="Nombre de la columna identificadora que debe excluirse del entrenamiento.",
    )
    parser.add_argument(
        "--positive_label",
        default=DEFAULT_POSITIVE_LABEL,
        help="Etiqueta considerada como clase positiva para el cálculo de ROC AUC.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporción de datos reservada para validación.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla utilizada para la partición y el modelo.",
    )
    return parser.parse_args()


def resolve_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Define manualmente las variables categóricas y numéricas del challenge.

    Si el archivo tuviera columnas adicionales, estas se agregan automáticamente
    al grupo que corresponda según su tipo de dato.
    """
    preferred_cat_cols = [
        "Sex",
        "Chest pain type",
        "FBS over 120",
        "EKG results",
        "Exercise angina",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium",
    ]
    preferred_num_cols = [
        "Age",
        "BP",
        "Cholesterol",
        "Max HR",
        "ST depression",
    ]

    cat_cols = [col for col in preferred_cat_cols if col in df.columns]
    num_cols = [col for col in preferred_num_cols if col in df.columns]

    remaining_cols = [col for col in df.columns if col not in cat_cols + num_cols]
    inferred_cat_cols = [
        col for col in remaining_cols if df[col].dtype == "object" or str(df[col].dtype).startswith("category")
    ]
    inferred_num_cols = [col for col in remaining_cols if col not in inferred_cat_cols]

    cat_cols.extend(inferred_cat_cols)
    num_cols.extend(inferred_num_cols)
    return num_cols, cat_cols


def build_pipeline(num_cols: list[str], cat_cols: list[str], random_state: int) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )


def train_and_save_model(
    train_csv: Path,
    model_out: Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Entrena el modelo, lo guarda y retorna métricas básicas."""
    if not train_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {train_csv}")

    df = pd.read_csv(train_csv)
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el archivo.")

    feature_df = df.drop(columns=[target_column])
    if id_column in feature_df.columns:
        feature_df = feature_df.drop(columns=[id_column])

    X = feature_df.copy()
    y = df[target_column].copy()

    num_cols, cat_cols = resolve_feature_groups(X)
    model = build_pipeline(num_cols=num_cols, cat_cols=cat_cols, random_state=random_state)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(positive_label) if positive_label in classes else 1
    y_proba = model.predict_proba(X_valid)[:, positive_index]
    y_valid_binary = (y_valid == positive_label).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid_binary, y_proba)),
        "classes": classes,
        "positive_label": positive_label,
        "num_features": num_cols,
        "cat_features": cat_cols,
        "validation_report": classification_report(y_valid, y_pred),
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    return metrics


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""
    args = parse_args()
    metrics = train_and_save_model(
        train_csv=args.train_csv,
        model_out=args.model_out,
        target_column=args.target_column,
        id_column=args.id_column,
        positive_label=args.positive_label,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Entrenamiento finalizado correctamente.")
    print(f"Modelo guardado en: {args.model_out}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Clases aprendidas:", metrics["classes"])
    print("Variables numéricas:", metrics["num_features"])
    print("Variables categóricas:", metrics["cat_features"])
    print("\nReporte de clasificación:\n")
    print(metrics["validation_report"])


if __name__ == "__main__":
    main()
