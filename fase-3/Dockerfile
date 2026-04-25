# Reutilizamos la imagen construida en la fase 2 para extender sus dependencias y scripts.
FROM heart-disease-fase2:latest
# Mantenemos el mismo directorio de trabajo dentro del contenedor.
WORKDIR /app
# Copiamos el archivo de dependencias de la fase 3, que añade Flask y Requests.
COPY requirements.txt /app/requirements_api.txt
# Instalamos las librerías adicionales necesarias para exponer y consumir la API REST.
RUN pip install --no-cache-dir -r /app/requirements_api.txt
# Copiamos nuevamente el script de entrenamiento para que quede disponible en la imagen final.
COPY train.py /app/train.py
# Copiamos nuevamente el script de predicción para que quede disponible en la imagen final.
COPY predict.py /app/predict.py
# Copiamos el archivo principal que levanta la API REST con Flask.
COPY apirest.py /app/apirest.py
# Copiamos el cliente de ejemplo para probar el consumo programático del API.
COPY client.py /app/client.py
# Exponemos el puerto 5000 para recibir peticiones HTTP desde fuera del contenedor.
EXPOSE 5000
# Definimos como comando por defecto el arranque de la aplicación REST.
CMD ["python", "apirest.py"]
