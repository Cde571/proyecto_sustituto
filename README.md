# Proyecto sustitutorio Modelos I - Predicting Heart Disease

Este repositorio contiene las tres fases del proyecto sustitutorio para la competición de Kaggle **Predicting Heart Disease** (Playground Series - Season 6 Episode 2).

La solución quedó organizada de acuerdo con lo solicitado por el curso:

- **fase-1/**: notebook con el entrenamiento, la validación y la generación de predicciones.
- **fase-2/**: scripts `train.py` y `predict.py`, más un `Dockerfile` para ejecutar el modelo en contenedor.
- **fase-3/**: scripts anteriores, `apirest.py`, `client.py` y un `Dockerfile` que extiende la imagen de la fase 2 para exponer una API REST.

## Estructura del repositorio

```text
proyecto_sustituto-main/
├─ README.md
├─ requirements.txt
├─ fase-1/
│  └─ 01_modelo_predictivo_heart_disease.ipynb
├─ fase-2/
│  ├─ train.py
│  ├─ predict.py
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ data/
│  ├─ models/
│  └─ output/
└─ fase-3/
   ├─ train.py
   ├─ predict.py
   ├─ apirest.py
   ├─ client.py
   ├─ Dockerfile
   ├─ requirements.txt
   ├─ data/
   ├─ models/
   └─ output/
```

## Datos requeridos

Los archivos de la competición **no se incluyen** en el repositorio por su tamaño. Deben descargarse desde Kaggle:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Enlace de la competición: `https://www.kaggle.com/competitions/playground-series-s6e2`

## Dependencias generales

Si deseas ejecutar el notebook o los scripts fuera de Docker, instala primero las dependencias base:

```bash
python -m venv venv
```

### Activación del entorno en Windows

```bash
venv\Scripts\activate
```

### Activación del entorno en Linux o macOS

```bash
source venv/bin/activate
```

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Fase 1 - Notebook

Archivo principal:

- `fase-1/01_modelo_predictivo_heart_disease.ipynb`

### Qué muestra el notebook

1. Carga de datos.
2. Exploración inicial y distribución de la variable objetivo.
3. Definición de variables numéricas y categóricas.
4. Preprocesamiento con `Pipeline` y `ColumnTransformer`.
5. Entrenamiento de una `LogisticRegression`.
6. Validación con `Accuracy`, `ROC AUC`, `classification_report` y matriz de confusión.
7. Generación de `submission.csv`.
8. Guardado de `model.joblib`.

### Ejecución sugerida

El notebook puede ejecutarse en **Google Colab** o en **Jupyter Notebook**.

Si se ejecuta localmente, coloca los CSV descargados en el mismo directorio del notebook o ajusta las rutas según tu entorno.

## Fase 2 - Scripts y Docker

Archivos principales:

- `fase-2/train.py`
- `fase-2/predict.py`
- `fase-2/Dockerfile`
- `fase-2/requirements.txt`

### Preparación de datos para la fase 2

Copia los archivos necesarios dentro de `fase-2/data/`.

- `fase-2/data/train.csv`
- `fase-2/data/test.csv`

### Ejecución local de la fase 2

#### Entrenamiento

```bash
cd fase-2
python train.py --train_csv data/train.csv --model_out models/model.joblib
```

#### Predicción

```bash
python predict.py --model_path models/model.joblib --input_csv data/test.csv --output_csv output/predictions.csv
```

### Ejecución con Docker en la fase 2

#### Construcción de la imagen

```bash
cd fase-2
docker build -t heart-disease-fase2 .
```

#### Entrenamiento dentro del contenedor

```bash
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" heart-disease-fase2 python train.py --train_csv data/train.csv --model_out models/model.joblib
```

En Linux o macOS usa esta variante del volumen:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" heart-disease-fase2 python train.py --train_csv data/train.csv --model_out models/model.joblib
```

#### Predicción dentro del contenedor

```bash
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" -v "%cd%/output:/app/output" heart-disease-fase2 python predict.py --model_path models/model.joblib --input_csv data/test.csv --output_csv output/predictions.csv
```

En Linux o macOS usa esta variante del volumen:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/output:/app/output" heart-disease-fase2 python predict.py --model_path models/model.joblib --input_csv data/test.csv --output_csv output/predictions.csv
```

## Fase 3 - API REST

Archivos principales:

- `fase-3/train.py`
- `fase-3/predict.py`
- `fase-3/apirest.py`
- `fase-3/client.py`
- `fase-3/Dockerfile`
- `fase-3/requirements.txt`

### Preparación de datos para la fase 3

Copia al menos el conjunto de entrenamiento en:

- `fase-3/data/train.csv`

Opcionalmente puedes copiar también un archivo de prueba para usarlo con `client.py`.

### Construcción de las imágenes

Primero construye la imagen de la fase 2, porque la fase 3 la extiende:

```bash
cd fase-2
docker build -t heart-disease-fase2 .
```

Luego construye la imagen del API:

```bash
cd ../fase-3
docker build -t heart-disease-fase3 .
```

### Levantar la API REST

```bash
docker run --rm -p 5000:5000 -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" heart-disease-fase3
```

En Linux o macOS usa esta variante del volumen:

```bash
docker run --rm -p 5000:5000 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" heart-disease-fase3
```

### Endpoint de entrenamiento

Con la API ya encendida, ejecuta:

```bash
curl -X POST http://localhost:5000/train -H "Content-Type: application/json" -d "{}"
```

Eso hará que el contenedor entrene usando por defecto `data/train.csv` y guarde el modelo en `models/model.joblib`.

### Endpoint de predicción

Ejemplo de llamado con un registro:

```bash
curl -X POST http://localhost:5000/predict   -H "Content-Type: application/json"   -d '{"data": {"Age": 63, "Sex": 1, "Chest pain type": 1, "BP": 145, "Cholesterol": 233, "FBS over 120": 1, "EKG results": 2, "Max HR": 150, "Exercise angina": 0, "ST depression": 2.3, "Slope of ST": 3, "Number of vessels fluro": 0, "Thallium": "normal"}}'
```

### Uso del cliente programático

Con la API corriendo, puedes probar desde otra terminal:

```bash
cd fase-3
python client.py train --base_url http://localhost:5000 --train_csv data/train.csv
```

```bash
python client.py predict --base_url http://localhost:5000 --input_csv data/test.csv
```

## Notas finales

- Los scripts incluyen **docstrings** en funciones y módulo.
- Los `Dockerfile` incluyen **comentarios explicativos en cada línea**, tal como se pide en la rúbrica.
- El flujo completo permite pasar del notebook al script, del script al contenedor y del contenedor al API REST.

## Autores

- Esteban Andrés Castaño Gallo
- Cristian Echeverry
