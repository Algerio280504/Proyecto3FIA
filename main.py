from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import (
    ArbolDecisionRequest,
    ArbolDecisionResponse,
    ClusterRequest,
    ClusterResponse,
    GeneticoRequest,
    GeneticoResponse,
)
from services import agrupar_datos, evaluar_paciente, reconstruir_mensaje


app = FastAPI(title="API de Inteligencia Artificial", redoc_url=None)


@app.exception_handler(RequestValidationError)
async def error_validacion(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {campo: detalle[campo] for campo in ("loc", "msg", "type")}
                for detalle in error.errors()
            ]
        },
    )


@app.post("/genetico", response_model=GeneticoResponse)
def genetico(entrada: GeneticoRequest) -> GeneticoResponse:
    return reconstruir_mensaje(entrada)


@app.post("/arbol_decision", response_model=ArbolDecisionResponse)
def arbol_decision(entrada: ArbolDecisionRequest) -> ArbolDecisionResponse:
    return evaluar_paciente(entrada)


@app.post("/cluster", response_model=ClusterResponse)
def cluster(entrada: ClusterRequest) -> ClusterResponse:
    try:
        return agrupar_datos(entrada)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
