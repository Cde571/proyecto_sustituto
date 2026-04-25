"""Expone una API REST para entrenar y predecir con el modelo cardíaco.

La API utiliza Flask y ofrece dos endpoints principales:
- POST /train: reentrena el modelo usando un dataset estándar.
- POST /predict: recibe uno o varios registros nuevos y devuelve sus predicciones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from train import train_and_save_model

DEFAULT_TRAIN_CSV = Path("data/train.csv")
DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_ID_COLUMN = "id"
DEFAULT_TARGET_COLUMN = "Heart Disease"
DEFAULT_POSITIVE_LABEL = "Presence"

app = Flask(__name__)


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Carga el modelo entrenado desde disco."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe un modelo entrenado en la ruta esperada: {model_path}"
        )
    return joblib.load(model_path)


@app.get("/")
def healthcheck():
    """Devuelve un mensaje simple para confirmar que la API está activa."""
    return jsonify(
        {
            "message": "API REST de Predicting Heart Disease activa.",
            "endpoints": ["POST /train", "POST /predict"],
        }
    )


@app.post("/train")
def train_endpoint():
    """Entrena el modelo usando un dataset estándar o uno enviado por JSON."""
    payload = request.get_json(silent=True) or {}
    train_csv = Path(payload.get("train_csv", DEFAULT_TRAIN_CSV))
    model_path = Path(payload.get("model_path", DEFAULT_MODEL_PATH))
    target_column = payload.get("target_column", DEFAULT_TARGET_COLUMN)
    id_column = payload.get("id_column", DEFAULT_ID_COLUMN)
    positive_label = payload.get("positive_label", DEFAULT_POSITIVE_LABEL)

    try:
        metrics = train_and_save_model(
            train_csv=train_csv,
            model_out=model_path,
            target_column=target_column,
            id_column=id_column,
            positive_label=positive_label,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 400

    return jsonify(
        {
            "status": "ok",
            "message": "Entrenamiento ejecutado correctamente.",
            "model_path": str(model_path),
            "accuracy": round(metrics["accuracy"], 6),
            "roc_auc": round(metrics["roc_auc"], 6),
            "classes": metrics["classes"],
            "positive_label": metrics["positive_label"],
        }
    )


@app.post("/predict")
def predict_endpoint():
    """Recibe un registro o una lista de registros y devuelve probabilidades."""
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    single_record = payload.get("data")
    positive_label = payload.get("positive_label", DEFAULT_POSITIVE_LABEL)

    if rows is None and single_record is None:
        return jsonify(
            {
                "status": "error",
                "message": "El cuerpo JSON debe incluir 'rows' o 'data'.",
            }
        ), 400

    records = rows if rows is not None else [single_record]

    try:
        model = load_model()
        frame = pd.DataFrame(records)
        if frame.empty:
            return jsonify({"status": "error", "message": "No se recibieron registros válidos."}), 400
        if DEFAULT_ID_COLUMN in frame.columns:
            frame = frame.drop(columns=[DEFAULT_ID_COLUMN])
        classes = list(model.named_steps["classifier"].classes_)
        positive_index = classes.index(positive_label) if positive_label in classes else 1
        probabilities = model.predict_proba(frame)[:, positive_index].tolist()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": str(exc)}), 400

    return jsonify(
        {
            "status": "ok",
            "positive_label": positive_label,
            "predictions": probabilities,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
