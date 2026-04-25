# Usamos una imagen oficial ligera de Python 3.11 como base para la fase 2.
FROM python:3.11-slim
# Definimos el directorio de trabajo donde vivirán los scripts y los datos.
WORKDIR /app
# Copiamos el archivo de dependencias específicas de la fase 2 al contenedor.
COPY requirements.txt /app/requirements.txt
# Instalamos las librerías necesarias para entrenar y predecir dentro del contenedor.
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copiamos el script de entrenamiento al directorio de trabajo del contenedor.
COPY train.py /app/train.py
# Copiamos el script de predicción al directorio de trabajo del contenedor.
COPY predict.py /app/predict.py
# Creamos carpetas internas para organizar datos, modelos y salidas generadas.
RUN mkdir -p /app/data /app/models /app/output
# Dejamos un comando por defecto sencillo que muestra la ayuda del script de entrenamiento.
CMD ["python", "train.py", "--help"]
