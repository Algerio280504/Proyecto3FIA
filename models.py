from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class Esquema(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class GeneticoRequest(Esquema):
    mensaje_oculto: str = Field(min_length=1, examples=["IA EVOLUTIVA 2026", "ALGORITMO GENETICO"])
    tam_poblacion: int = Field(gt=10, le=500, examples=[150, 200])
    max_generaciones: int = Field(gt=1, le=1000, examples=[300, 500])
    prob_cruce: float = Field(ge=0.0, le=1.0, examples=[0.8, 0.9])
    prob_mutacion: float = Field(ge=0.0, le=0.5, examples=[0.3, 0.1])


class GeneticoResponse(Esquema):
    estado: Literal["exitoso"] = "exitoso"
    mensaje_oculto_objetivo: str
    solucion_obtenida: str
    aptitud_alcanzada: int
    generaciones_ejecutadas: int
    convergencia: bool
    historial_fitness_max: list[int]


class ArbolDecisionRequest(Esquema):
    edad: int = Field(ge=18, le=100, examples=[55])
    presion_sistolica: int = Field(ge=80, le=220, examples=[145])
    colesterol: int = Field(ge=100, le=500, examples=[240])
    frecuencia_max: int = Field(ge=60, le=220, examples=[130])
    angina_ejercicio: int = Field(ge=0, le=1, examples=[1])


class ArbolDecisionResponse(Esquema):
    estado: Literal["exitoso"] = "exitoso"
    paciente_evaluado: ArbolDecisionRequest
    clase_predicha: Literal[0, 1]
    diagnostico_texto: str
    probabilidad_riesgo: float = Field(ge=0.0, le=1.0)
    nivel_riesgo: Literal["Bajo", "Moderado", "Alto"]


class ClusterRequest(Esquema):
    k_clusters: int = Field(ge=2, le=10, examples=[3])
    max_iteraciones: int = Field(gt=5, le=500, examples=[100])
    datos: list[list[FiniteFloat]] = Field(
        min_length=2,
        examples=[[[1, 1], [1, 2], [8, 8], [8, 9], [20, 1], [21, 1]]],
    )

    @model_validator(mode="after")
    def validar_matriz(self) -> "ClusterRequest":
        dimensiones = len(self.datos[0])
        if not dimensiones or any(len(fila) != dimensiones for fila in self.datos):
            raise ValueError("datos debe ser una matriz rectangular con columnas")
        if len({tuple(fila) for fila in self.datos}) < self.k_clusters:
            raise ValueError("datos debe contener al menos k_clusters puntos distintos")
        return self


class ClusterResponse(Esquema):
    estado: Literal["exitoso"] = "exitoso"
    k_utilizado: int
    num_muestras: int
    centroides_finales: list[list[FiniteFloat]]
    etiquetas_asignadas: list[int]
    inercia_wcss: float = Field(ge=0.0)
