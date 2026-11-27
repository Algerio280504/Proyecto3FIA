# Proyecto3FIA

API educativa de FastAPI para reconstruir mensajes mediante un algoritmo genético, clasificar perfiles cardíacos mediante un árbol de decisión y agrupar puntos mediante K-Means.

Backend desarrollado en **Python 3.10+** con **FastAPI** que expone tres endpoints HTTP (`POST`) independientes para resolver problemas clásicos de Inteligencia Artificial:

1. **Algoritmo Genético (`POST /genetico`)**: Reconstrucción evolutiva de una cadena de texto objetivo.
2. **Aprendizaje Supervisado (`POST /arbol_decision`)**: Clasificación de riesgo cardiaco mediante un Árbol de Decisión (`DecisionTreeClassifier`).
3. **Aprendizaje No Supervisado (`POST /cluster`)**: Segmentación de datos multidimensionales mediante `KMeans`, retornando centroides finales, etiquetas e inercia WCSS.

---

## Arquitectura del Proyecto

El proyecto mantiene una arquitectura modular con punto de acceso único en `main.py`:

proyecto_final_ia/
├── main.py                         # Inicialización de FastAPI y declaración exclusiva de endpoints
├── models.py                       # Modelos Pydantic de validación estricta (Request / Response)
├── services.py                     # Implementación y ejecución de los algoritmos de IA
├── dataset_cardiaco_simulado.csv   # Conjunto de datos clínicos para entrenamiento del modelo
├── requirements.txt                # Dependencias del entorno
└── README.md                       # Documentación y manual de ejecución

---

## Instalación y Ejecución

### 1. Clonar el repositorio e ingresar al directorio

git clone <URL_DEL_REPOSITORIO>
cd proyecto_final_ia

### 2. Crear y activar un entorno virtual (recomendado)

En Linux / macOS:
python -m venv venv
source venv/bin/activate

En Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

### 3. Instalar dependencias

pip install -r requirements.txt

### 4. Iniciar el servidor con Uvicorn

uvicorn main:app --reload

La API quedará disponible en http://127.0.0.1:8000 y la interfaz interactiva Swagger UI en:
- http://127.0.0.1:8000/docs

---

## Especificación y Ejemplos de Consumo

### 1. Algoritmo Genético — POST /genetico

Reconstruye un mensaje oculto mediante selección por torneo, cruce de un punto, mutación puntual y elitismo.

Ejemplo cURL:
curl -X POST "http://127.0.0.1:8000/genetico" \
  -H "Content-Type: application/json" \
  -d '{
    "mensaje_oculto": "IA EVOLUTIVA 2026",
    "tam_poblacion": 120,
    "max_generaciones": 500,
    "prob_cruce": 0.85,
    "prob_mutacion": 0.15
  }'

Ejemplo de respuesta (200 OK):
{
  "estado": "exitoso",
  "mensaje_oculto_objetivo": "IA EVOLUTIVA 2026",
  "solucion_obtenida": "IA EVOLUTIVA 2026",
  "aptitud_alcanzada": 17,
  "generaciones_ejecutadas": 42,
  "convergencia": true,
  "historial_fitness_max": [5, 8, 11, 14, 17]
}

### 2. Árbol de Decisión — POST /arbol_decision

Evalúa parámetros clínicos de un paciente y devuelve el diagnóstico predicho, probabilidad de riesgo y nivel de riesgo (Bajo, Moderado, Alto).

Ejemplo cURL:
curl -X POST "http://127.0.0.1:8000/arbol_decision" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 55,
    "presion_sistolica": 145,
    "colesterol": 240,
    "frecuencia_max": 130,
    "angina_ejercicio": 1
  }'

Ejemplo de respuesta (200 OK):
{
  "estado": "exitoso",
  "paciente_evaluado": {
    "edad": 55,
    "presion_sistolica": 145,
    "colesterol": 240,
    "frecuencia_max": 130,
    "angina_ejercicio": 1
  },
  "clase_predicha": 1,
  "diagnostico_texto": "Riesgo de Afección Cardiaca Detectado",
  "probabilidad_riesgo": 0.88,
  "nivel_riesgo": "Alto"
}

### 3. Clustering K-Means — POST /cluster

Agrupa una matriz rectangular dinámica de datos numéricos en k_clusters grupos y calcula los centroides finales y la inercia WCSS.

Ejemplo cURL:
curl -X POST "http://127.0.0.1:8000/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "k_clusters": 3,
    "max_iteraciones": 100,
    "datos": [
      [25.0, 60.0],
      [55.0, 49.0],
      [80.0, 15.0],
      [26.0, 61.0],
      [54.0, 48.0]
    ]
  }'

Ejemplo de respuesta (200 OK):
{
  "estado": "exitoso",
  "k_utilizado": 3,
  "num_muestras": 5,
  "centroides_finales": [
    [25.5, 60.5],
    [80.0, 15.0],
    [54.5, 48.5]
  ],
  "etiquetas_asignadas": [0, 2, 1, 0, 2],
  "inercia_wcss": 2.0
}

---

## Manejo de Errores de Validación

Todas las entradas se validan mediante Pydantic. Si algún parámetro incumple su tipo, rango permitido o estructura esperada, la API responde automáticamente con código HTTP 422 Unprocessable Entity detallando el campo y el motivo del error.

