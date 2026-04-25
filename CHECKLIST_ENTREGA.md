# Checklist de cumplimiento del proyecto sustitutorio

## Fase 1

- [x] Existe el directorio `fase-1/`.
- [x] Existe al menos un notebook que muestra entrenamiento y predicción.
- [x] El notebook trabaja la competición seleccionada de Kaggle.
- [x] El notebook genera `submission.csv`.
- [x] El notebook guarda el modelo entrenado como `model.joblib`.

## Fase 2

- [x] Existe el directorio `fase-2/`.
- [x] Existe `train.py`.
- [x] Existe `predict.py`.
- [x] Existe `Dockerfile`.
- [x] Existe `requirements.txt` específico de la fase 2.
- [x] `train.py` reentrena el modelo y guarda una nueva versión en disco.
- [x] `predict.py` recibe un CSV de entrada y emite una predicción por fila.
- [x] El `Dockerfile` instala dependencias y permite ejecutar los scripts en contenedor.

## Fase 3

- [x] Existe el directorio `fase-3/`.
- [x] Existen `train.py` y `predict.py`.
- [x] Existe `apirest.py`.
- [x] Existe `client.py`.
- [x] Existe `Dockerfile`.
- [x] Existe `requirements.txt` específico de la fase 3.
- [x] `apirest.py` expone el endpoint `POST /predict`.
- [x] `apirest.py` expone el endpoint `POST /train`.
- [x] `client.py` ilustra el consumo programático del API.
- [x] El `Dockerfile` de fase 3 extiende la imagen construida en la fase 2.

## README y documentación

- [x] Existe `README.md` en la raíz del repositorio.
- [x] El `README.md` describe cómo ejecutar la fase 1.
- [x] El `README.md` describe cómo ejecutar la fase 2.
- [x] El `README.md` describe cómo ejecutar la fase 3.
- [x] Los scripts incluyen docstrings.
- [x] Los `Dockerfile` tienen comentarios explicativos en cada línea.
