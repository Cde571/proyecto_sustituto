"""Cliente simple para consumir la API REST de la fase 3.

Permite invocar programáticamente los endpoints /train y /predict para probar
el contenedor del API una vez desplegado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_BASE_URL = "http://localhost:5000"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del cliente."""
    parser = argparse.ArgumentParser(description="Cliente de prueba para la API REST.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Invoca el endpoint de entrenamiento.")
    train_parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="URL base del API REST.")
    train_parser.add_argument(
        "--train_csv",
        default="data/train.csv",
        help="Ruta que el contenedor debe usar como dataset de entrenamiento.",
    )

    predict_parser = subparsers.add_parser("predict", help="Invoca el endpoint de predicción.")
    predict_parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="URL base del API REST.")
    predict_parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="CSV local con registros para enviar al endpoint /predict.",
    )
    return parser.parse_args()


def call_train(base_url: str, train_csv: str) -> dict[str, Any]:
    """Llama al endpoint /train y devuelve la respuesta JSON."""
    response = requests.post(
        f"{base_url.rstrip('/')}/train",
        json={"train_csv": train_csv},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def call_predict(base_url: str, input_csv: Path) -> dict[str, Any]:
    """Carga un CSV local, lo transforma a JSON y llama al endpoint /predict."""
    if not input_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_csv}")
    rows = pd.read_csv(input_csv).to_dict(orient="records")
    response = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"rows": rows},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Punto de entrada del cliente cuando se ejecuta desde consola."""
    args = parse_args()
    if args.command == "train":
        result = call_train(base_url=args.base_url, train_csv=args.train_csv)
    else:
        result = call_predict(base_url=args.base_url, input_csv=args.input_csv)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
