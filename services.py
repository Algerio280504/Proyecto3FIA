from pathlib import Path
from random import Random

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from models import (
    ArbolDecisionRequest,
    ArbolDecisionResponse,
    ClusterRequest,
    ClusterResponse,
    GeneticoRequest,
    GeneticoResponse,
)


def reconstruir_mensaje(entrada: GeneticoRequest) -> GeneticoResponse:
    objetivo = entrada.mensaje_oculto
    longitud = len(objetivo)
    alfabeto = tuple(dict.fromkeys(objetivo))
    azar = Random()
    poblacion = [
        "".join(azar.choices(alfabeto, k=longitud))
        for _ in range(entrada.tam_poblacion)
    ]
    historial = []

    for generacion in range(1, entrada.max_generaciones + 1):
        aptitudes = [
            sum(gen == esperado for gen, esperado in zip(individuo, objetivo))
            for individuo in poblacion
        ]
        indice_mejor = max(range(len(poblacion)), key=aptitudes.__getitem__)
        mejor = poblacion[indice_mejor]
        aptitud = aptitudes[indice_mejor]
        historial.append(aptitud)

        if aptitud == longitud or generacion == entrada.max_generaciones:
            break

        # Elitismo: conserva al mejor individuo de la generación.
        siguiente = [mejor]
        indices = range(len(poblacion))
        while len(siguiente) < entrada.tam_poblacion:
            padre = poblacion[max(azar.sample(indices, 3), key=aptitudes.__getitem__)]
            hijo = padre
            if longitud > 1 and azar.random() < entrada.prob_cruce:
                madre = poblacion[max(azar.sample(indices, 3), key=aptitudes.__getitem__)]
                corte = azar.randrange(1, longitud)
                hijo = padre[:corte] + madre[corte:]
            if azar.random() < entrada.prob_mutacion:
                posicion = azar.randrange(longitud)
                hijo = hijo[:posicion] + azar.choice(alfabeto) + hijo[posicion + 1:]
            siguiente.append(hijo)
        poblacion = siguiente

    return GeneticoResponse(
        mensaje_oculto_objetivo=objetivo,
        solucion_obtenida=mejor,
        aptitud_alcanzada=aptitud,
        generaciones_ejecutadas=len(historial),
        convergencia=aptitud == longitud,
        historial_fitness_max=historial,
    )


def entrenar_arbol_cardiaco() -> DecisionTreeClassifier:
    ruta = Path(__file__).with_name("dataset_cardiaco_simulado.csv")
    datos = np.genfromtxt(ruta, delimiter=",", names=True, encoding="utf-8-sig")
    columnas = ("Edad", "PresionSistolica", "Colesterol", "FrecuenciaMax", "AnginaEjercicio")
    perfiles = np.column_stack([datos[columna] for columna in columnas])
    clases = datos["DiagnosticoCardiopatia"].astype(int)
    semilla = 10
    x_train, _, y_train, _ = train_test_split(
        perfiles, clases, test_size=0.2, stratify=clases, random_state=semilla
    )
    arbol = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, min_samples_leaf=5, random_state=semilla
    )
    return arbol.fit(x_train, y_train)


_ARBOL = entrenar_arbol_cardiaco()


def evaluar_paciente(entrada: ArbolDecisionRequest) -> ArbolDecisionResponse:
    perfil = [[
        entrada.edad,
        entrada.presion_sistolica,
        entrada.colesterol,
        entrada.frecuencia_max,
        entrada.angina_ejercicio,
    ]]
    clase = int(_ARBOL.predict(perfil)[0])
    columna_riesgo = _ARBOL.classes_.tolist().index(1)
    probabilidad = round(float(_ARBOL.predict_proba(perfil)[0, columna_riesgo]), 4)
    nivel = "Bajo"
    if probabilidad >= 0.7:
        nivel = "Alto"
    elif probabilidad >= 0.3:
        nivel = "Moderado"

    return ArbolDecisionResponse(
        paciente_evaluado=entrada,
        clase_predicha=clase,
        diagnostico_texto=(
            "Riesgo de Afección Cardiaca Detectado"
            if clase
            else "Sin Riesgo de Afección Cardiaca Detectado"
        ),
        probabilidad_riesgo=probabilidad,
        nivel_riesgo=nivel,
    )


def agrupar_datos(entrada: ClusterRequest) -> ClusterResponse:
    datos = np.asarray(entrada.datos, dtype=np.float64)
    modelo = KMeans(
        n_clusters=entrada.k_clusters,
        max_iter=entrada.max_iteraciones,
        n_init=10,
        random_state=42,
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            modelo.fit(datos)
    except FloatingPointError as error:
        raise ValueError("La escala de datos desborda el cálculo de K-Means") from error
    if not np.isfinite(modelo.inertia_) or not np.isfinite(modelo.cluster_centers_).all():
        raise ValueError("K-Means produjo resultados fuera del rango numérico")
    if len(np.unique(modelo.labels_)) != entrada.k_clusters:
        raise ValueError("La precisión numérica impidió separar los grupos solicitados")

    return ClusterResponse(
        k_utilizado=entrada.k_clusters,
        num_muestras=datos.shape[0],
        centroides_finales=np.round(modelo.cluster_centers_, 4).tolist(),
        etiquetas_asignadas=modelo.labels_.tolist(),
        inercia_wcss=round(float(modelo.inertia_), 4),
    )
