from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class Esquema(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class GeneticoRequest(Esquema):
    mensaje_oculto: str = Field(min_length=1)
    tam_poblacion: int = Field(gt=10, le=500)
    max_generaciones: int = Field(gt=1, le=1000)
    prob_cruce: float = Field(ge=0.0, le=1.0)
    prob_mutacion: float = Field(ge=0.0, le=0.5)


class GeneticoResponse(Esquema):
    estado: Literal["exitoso"] = "exitoso"
    mensaje_oculto_objetivo: str
    solucion_obtenida: str
    aptitud_alcanzada: int
    generaciones_ejecutadas: int
    convergencia: bool
    historial_fitness_max: list[int]

