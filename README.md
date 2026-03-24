# Predicción de Enfermedades Cardíacas - Fase 1

Este proyecto corresponde a la primera etapa de desarrollo de un modelo predictivo basado en la competición de Kaggle: [Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2).

El objetivo principal es entrenar un modelo funcional que emita predicciones sobre la presencia de patologías cardíacas basándose en datos clínicos.

---

## 📂 Contenido de la Entrega
* **fase-1/**: Directorio que contiene el Notebook principal.
* **01_modelo_predictivo_heart_disease.ipynb**: Notebook con el análisis de datos, entrenamiento del modelo y generación de resultados.
* **requirements.txt**: Listado de librerías necesarias para asegurar la reproducibilidad.

---

## 🚀 Instrucciones de Ejecución Paso a Paso

Para ejecutar este proyecto correctamente, siga estas instrucciones en su entorno local:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO
```

### 2. Configurar el entorno de Python
Se recomienda el uso de un entorno virtual para evitar conflictos de versiones:

#### Crear el entorno
```bash
python -m venv venv
```
#### Activar el entorno (Windows)
```bash
venv\Scripts\activate
```
#### Activar el entorno (Linux/Mac)
```bash
source venv/bin/activate
```

### 3. Instalar dependencias
Este paso es fundamental para que el Notebook funcione sin errores de "ModuleNotFound":
```bash
pip install -r requirements.txt
```

### 4. Ejecución del Notebook
1. Inicie el servidor de Jupyter:

```bash
jupyter notebook
```

2. En la interfaz del navegador, entre a la carpeta `fase-1/`.

3. Abra el archivo `01_modelo_predictivo_heart_disease.ipynb`.

4. Seleccione **Cell > Run All** en el menú superior para ejecutar todo el flujo de trabajo.

---

## Descripción del Notebook
El código está organizado en las siguientes secciones documentadas:

1. Carga y Exploración: Importación de datos y análisis de variables.

2. Preprocesamiento: Limpieza y transformación de datos usando Pipelines.

3. Entrenamiento: Implementación de un modelo de Regresión Logística.

4. Resultados: Generación del archivo `submission.csv` y exportación del modelo `model.joblib`.

Desarrollado por: Esteban Andrés Castaño Gallo y Cristian Echeverry.
