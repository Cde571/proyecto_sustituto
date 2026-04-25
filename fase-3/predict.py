"""Genera predicciones utilizando un modelo previamente entrenado.

El script carga un modelo serializado, recibe un CSV con observaciones nuevas y
produce un archivo CSV con una probabilidad por fila para la clase positiva.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_ID_COLUMN = "id"
DEFAULT_PREDICTION_COLUMN = "Heart Disease"
DEFAULT_POSITIVE_LABEL = "Presence"
DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_OUTPUT_PATH = Path("output/predictions.csv")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(
        description="Genera predicciones para nuevos datos usando el modelo entrenado."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al archivo joblib del modelo entrenado.",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Ruta al archivo CSV con los datos de entrada.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta donde se guardarán las predicciones en formato CSV.",
    )
    parser.add_argument(
        "--id_column",
        default=DEFAULT_ID_COLUMN,
        help="Nombre de la columna identificadora que debe conservarse en la salida.",
    )
    parser.add_argument(
        "--prediction_column",
        default=DEFAULT_PREDICTION_COLUMN,
        help="Nombre de la columna de predicción en el CSV de salida.",
    )
    parser.add_argument(
        "--positive_label",
        default=DEFAULT_POSITIVE_LABEL,
        help="Etiqueta cuya probabilidad se exportará en el archivo de salida.",
    )
    return parser.parse_args()


def predict_from_csv(
    model_path: Path,
    input_csv: Path,
    output_csv: Path,
    id_column: str = DEFAULT_ID_COLUMN,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
) -> pd.DataFrame:
    """Carga el modelo, predice probabilidades y guarda el CSV de resultados."""
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {model_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_csv}")

    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    if id_column in df.columns:
        ids = df[id_column].copy()
        features = df.drop(columns=[id_column])
    else:
        ids = pd.Series(range(len(df)), name=id_column)
        features = df.copy()

    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(positive_label) if positive_label in classes else 1
    probabilities = model.predict_proba(features)[:, positive_index]

    predictions = pd.DataFrame({id_column: ids, prediction_column: probabilities})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    return predictions


def main() -> None:
    """Punto de entrada del script cuando se ejecuta desde consola."""
    args = parse_args()
    predictions = predict_from_csv(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        id_column=args.id_column,
        prediction_column=args.prediction_column,
        positive_label=args.positive_label,
    )
    print("Predicciones generadas correctamente.")
    print(f"Archivo de salida: {args.output_csv}")
    print("Primeras filas de la salida:")
    print(predictions.head())


if __name__ == "__main__":
    main()
