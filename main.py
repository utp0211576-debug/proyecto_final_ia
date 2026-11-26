from urllib import request

from fastapi import FastAPI
from models import (
    GeneticoRequest, GeneticoResponse,
    ArbolDecisionRequest, ArbolDecisionResponse,
    ClusterRequest, ClusterResponse
)
import services

app = FastAPI(title="API RESTful de IA", version="1.0.0")

@app.post("/genetico", response_model=GeneticoResponse)
def algoritmo_genetico_ejecutar(request: GeneticoRequest):
    return services.algoritmo_genetico(
        mensaje_oculto=request.mensaje_oculto,
        tam_poblacion=request.tam_poblacion,
        max_generaciones=request.max_generaciones,
        prob_cruce=request.prob_cruce,
        prob_mutacion=request.prob_mutacion
    )

@app.post("/arbol_decision", response_model=ArbolDecisionResponse)
def algoritmo_arbol_decision(request: ArbolDecisionRequest):
    return services.evaluar_paciente_arbol(request)

@app.post("/cluster", response_model=ClusterResponse)
def algoritmo_clusterizado(request: ClusterRequest):
    return services.ejecutar_kmeans_cluster(request)