# Proyecto3FIA

API educativa de FastAPI para reconstruir mensajes mediante un algoritmo genético, clasificar perfiles cardíacos mediante un árbol de decisión y agrupar puntos mediante K-Means.

## Instalación y ejecución

Requiere Python 3.10 o posterior. Desde la raíz del repositorio, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

En macOS o Linux, usa `.venv/bin/python` en los dos últimos comandos.

Abre [Swagger UI](http://localhost:8000/docs), expande un endpoint, elige **Try it out**, pega el JSON correspondiente y pulsa **Execute**.

## Endpoints

Envía los cuerpos con `Content-Type: application/json`. Los tres endpoints devuelven JSON con HTTP 200 al completar el cálculo y HTTP 422 cuando la entrada es inválida. Los campos adicionales y las cadenas en campos numéricos se rechazan.

### POST /genetico

```json
{
  "mensaje_oculto": "IA EVOLUTIVA 2026",
  "tam_poblacion": 150,
  "max_generaciones": 300,
  "prob_cruce": 0.8,
  "prob_mutacion": 0.3
}
```

La población admite entre 11 y 500 individuos y el límite entre 2 y 1000 generaciones. Las probabilidades de cruce y mutación admiten de 0 a 1 y de 0 a 0.5, respectivamente. El algoritmo toma el alfabeto del mensaje objetivo, aplica selección por torneo, cruce de un punto, mutación de un carácter y elitismo. La aptitud cuenta las posiciones que coinciden con el objetivo. La primera entrada del historial corresponde a la población inicial. Si se agota el límite, la API devuelve el mejor individuo encontrado con `convergencia: false`.

### POST /arbol_decision

```json
{
  "edad": 55,
  "presion_sistolica": 145,
  "colesterol": 240,
  "frecuencia_max": 130,
  "angina_ejercicio": 1
}
```

La edad debe estar entre 18 y 100 años, la presión sistólica entre 80 y 220 mmHg, el colesterol entre 100 y 500 mg/dL y la frecuencia máxima entre 60 y 220 latidos por minuto. Usa 0 para ausencia de angina durante el ejercicio y 1 para presencia.

Al cargar la aplicación se entrena el árbol podado de la práctica con `dataset_cardiaco_simulado.csv`. Se usa una división estratificada de 80 % para entrenamiento y 20 % reservado para pruebas, con semilla 10. El árbol usa entropía, profundidad máxima de 3 y un mínimo de 5 muestras por hoja.

La respuesta incluye el perfil evaluado, la clase, su descripción, la proporción de clase 1 en la hoja y el nivel de riesgo. El nivel es bajo por debajo de 0.30, moderado desde 0.30 y alto desde 0.70. El dataset es simulado y la salida carece de validación clínica.

### POST /cluster

```json
{
  "k_clusters": 3,
  "max_iteraciones": 100,
  "datos": [[1, 1], [1, 2], [8, 8], [8, 9], [20, 1], [21, 1]]
}
```

Se admiten de 2 a 10 grupos y entre 6 y 500 iteraciones por inicialización. Cada fila representa un punto con el mismo número de coordenadas finitas y debe haber al menos tantos puntos distintos como grupos. K-Means conserva el orden de las muestras en las etiquetas y usa índices desde 0, que apuntan a los centroides. La inercia WCSS es la suma de las distancias euclidianas al cuadrado entre cada punto y su centroide asignado.

## Estructura

- `main.py` inicializa FastAPI y declara los tres endpoints.
- `models.py` concentra los modelos Pydantic de entrada y salida.
- `services.py` implementa el algoritmo genético, el árbol de decisión y K-Means.
- `dataset_cardiaco_simulado.csv` contiene los datos simulados de la práctica del árbol de decisión.
- `requirements.txt` declara las dependencias de ejecución.
