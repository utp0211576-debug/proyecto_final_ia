#importacion de librerias
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from models import (
    ArbolDecisionRequest, ArbolDecisionResponse,
    ClusterRequest, ClusterResponse
)
#3.1 Endpoint 1: Algoritmo Genético (POST /genetico)
ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
#Definicion de funciones
def crear_individuo(longitud = int):
  """
  Genera una cadena aleatoria de longitud longitud_frase
  """
  return "".join(random.choice(ALFABETO)for _ in range(longitud))
#############################################################
def crear_poblacion(tamano = int, longitud=int):
  """
  Genera una poblacion (lista) de individuos aleatorios
  """
  return [crear_individuo(longitud) for _ in range(tamano)]
#evalaciones del fitness
def evaluar_fitness(individuo=str,objetivo=str):
  """
  Calcular el numero de caracteres que coinciden entre el individuo y el objetivo
  """
  return sum(1 for i,j in zip(individuo,objetivo) if i==j)
#definicion de torneo(seleccion del individuo)
def seleccion_torneo(poblacion: list[str], fitness_poblacion: list[int], k: int = 3):
  """
  Selecciona el mejor individuo aleatorio de una poblacion de tamaño k
  """
  seleccionados = random.sample(range(len(poblacion)), k)
  mejor_individuo = max(seleccionados, key=lambda x: fitness_poblacion[x])
  return poblacion[mejor_individuo]
#combinacion de padres en punto de corte aleatorio
def cruce_punto(padre1=str,madre1=str,prob_cruce=float):
  """
  cruza dos padrs en un punto de corte aleatorio
  """
  if random.random() < prob_cruce:
    punto_corte = random.randint(1,len(padre1)-1)
    hijo1= padre1[:punto_corte]+madre1[punto_corte:]
    hijo2= madre1[:punto_corte]+padre1[punto_corte:]
    return hijo1,hijo2
  return padre1,madre1
#funcio de mutacion
def mutacion(individuo=str, prob_mutacion=float)->str:
  """
  de manera aleatoria altera caracteres de los individuos con probabilidad de mutacion prob_mutacion
  """
  resultado = []
  for c in individuo:
    if random.random() < prob_mutacion:
      resultado.append(random.choice(ALFABETO))
    else:
      resultado.append(c)
  return "".join(resultado)
#definicion del algoritmo genetico
def algoritmo_genetico(
    mensaje_oculto: str,
    tam_poblacion: int,
    max_generaciones: int,
    prob_cruce: float,
    prob_mutacion: float,
    k_torneo: int = 3
) -> dict:
    longitud = len(mensaje_oculto)
    poblacion = crear_poblacion(tam_poblacion, longitud)
    
    historial_max = []
    mejor_solucion = ""
    mejor_fitness = 0
    generaciones_ejecutadas = 0
    convergencia = False

    for generacion in range(1, max_generaciones + 1):
        generaciones_ejecutadas = generacion
        fitness_poblacion = [evaluar_fitness(ind, mensaje_oculto) for ind in poblacion]
        
        mejor_fitness = max(fitness_poblacion)
        mejor_solucion = poblacion[fitness_poblacion.index(mejor_fitness)]
        historial_max.append(mejor_fitness)

        # Criterio de parada si alcanza la aptitud máxima (longitud del mensaje)
        if mejor_fitness == longitud:
            convergencia = True
            break

        # Elitismo: conservar al mejor individuo
        nueva_poblacion = [mejor_solucion]

        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, fitness_poblacion, k=k_torneo)
            padre2 = seleccion_torneo(poblacion, fitness_poblacion, k=k_torneo)
            hijo1, hijo2 = cruce_punto(padre1, padre2, prob_cruce)
            
            nueva_poblacion.append(mutacion(hijo1, prob_mutacion))
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(mutacion(hijo2, prob_mutacion))

        poblacion = nueva_poblacion

    return {
        "estado": "exitoso",
        "mensaje_oculto_objetivo": mensaje_oculto,
        "solucion_obtenida": mejor_solucion,
        "aptitud_alcanzada": mejor_fitness,
        "generaciones_ejecutadas": generaciones_ejecutadas,
        "convergencia": convergencia,
        "historial_fitness_max": historial_max
    }
    
#3.2 Endpoint 2: ML supervisado - Árbol de decisión (POST /arbol_decision)
def entrenar_arbol_decision():
    SEED = 10
    ruta_csv = os.path.join(os.path.dirname(__file__), "dataset_cardiaco_simulado.csv")
    
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró 'dataset_cardiaco_simulado.csv' en {os.path.dirname(__file__)}. "
            "Colócalo junto a services.py."
        )

    df_cardio = pd.read_csv(ruta_csv)
    x = df_cardio.drop(columns=['DiagnosticoCardiopatia'])
    y = df_cardio['DiagnosticoCardiopatia']

    X_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Árbol podado exacto de tu práctica
    arbol_podado = DecisionTreeClassifier(
        random_state=SEED,
        criterion='entropy',
        max_depth=3,
        min_samples_leaf=5
    )
    arbol_podado.fit(X_train, y_train)
    return arbol_podado, list(x.columns)

# Entrenamos el modelo una vez al iniciar la API
MODELO_ARBOL, COLUMNAS_ARBOL = entrenar_arbol_decision()

def evaluar_paciente_arbol(paciente: ArbolDecisionRequest) -> ArbolDecisionResponse:
    # Armamos el DataFrame con los mismos nombres de columnas del CSV
    datos_entrada = pd.DataFrame([{
        COLUMNAS_ARBOL[0]: paciente.edad,
        COLUMNAS_ARBOL[1]: paciente.presion_sistolica,
        COLUMNAS_ARBOL[2]: paciente.colesterol,
        COLUMNAS_ARBOL[3]: paciente.frecuencia_max,
        COLUMNAS_ARBOL[4]: paciente.angina_ejercicio
    }])

    clase = int(MODELO_ARBOL.predict(datos_entrada)[0])
    probabilidades = MODELO_ARBOL.predict_proba(datos_entrada)[0]
    prob_riesgo = float(round(probabilidades[1], 2))

    if clase == 1 or prob_riesgo >= 0.70:
        diagnostico = "Riesgo de Afección Cardiaca Detectado"
        nivel = "Alto"
    elif prob_riesgo >= 0.40:
        diagnostico = "Riesgo Moderado de Afección Cardiaca"
        nivel = "Medio"
    else:
        diagnostico = "Sin Riesgo Significativo de Afección Cardiaca"
        nivel = "Bajo"

    return ArbolDecisionResponse(
        estado="exitoso",
        paciente_evaluado=paciente,
        clase_predicha=clase,
        diagnostico_texto=diagnostico,
        probabilidad_riesgo=prob_riesgo,
        nivel_riesgo=nivel
    )
# Entrenamos el modelo una vez al iniciar la API
MODELO_ARBOL, COLUMNAS_ARBOL = entrenar_arbol_decision()

def evaluar_paciente_arbol(paciente: ArbolDecisionRequest) -> ArbolDecisionResponse:
    # Armamos el DataFrame con los mismos nombres de columnas del CSV
    datos_entrada = pd.DataFrame([{
        COLUMNAS_ARBOL[0]: paciente.edad,
        COLUMNAS_ARBOL[1]: paciente.presion_sistolica,
        COLUMNAS_ARBOL[2]: paciente.colesterol,
        COLUMNAS_ARBOL[3]: paciente.frecuencia_max,
        COLUMNAS_ARBOL[4]: paciente.angina_ejercicio
    }])

    clase = int(MODELO_ARBOL.predict(datos_entrada)[0])
    probabilidades = MODELO_ARBOL.predict_proba(datos_entrada)[0]
    prob_riesgo = float(round(probabilidades[1], 2))

    if clase == 1 or prob_riesgo >= 0.70:
        diagnostico = "Riesgo de Afección Cardiaca Detectado"
        nivel = "Alto"
    elif prob_riesgo >= 0.40:
        diagnostico = "Riesgo Moderado de Afección Cardiaca"
        nivel = "Medio"
    else:
        diagnostico = "Sin Riesgo Significativo de Afección Cardiaca"
        nivel = "Bajo"

    return ArbolDecisionResponse(
        estado="exitoso",
        paciente_evaluado=paciente,
        clase_predicha=clase,
        diagnostico_texto=diagnostico,
        probabilidad_riesgo=prob_riesgo,
        nivel_riesgo=nivel
    )
#3.3 Endpoint 3: ML no supervisado - Clustering K-Means (POST /cluster)
def inicializacion_centroides(X, K):
    """
    Definicion de K centroides aleatorios dentro del espacio muestral X
    """
    indices_aleatorios = np.random.choice(X.shape[0], K, replace=False)
    return X[indices_aleatorios].copy()

# Asignacion de los puntos mas cercanos a los centroides para conformar los clusters iniciales
def asignar_cluster(X, centroides):
    distancias = np.linalg.norm(X[:, np.newaxis, :] - centroides[np.newaxis, :, :], axis=2)
    indices_cluster = np.argmin(distancias, axis=1)
    return indices_cluster

# Funcion que ajusta la posicion con respecto al promedio de los individuos
def recalcular_centroides(X, etiquetas, K):
    N, D = X.shape
    nuevos_centroides = np.zeros((K, D))
    for k in range(K):
        puntos_cluster = X[etiquetas == k]
        if len(puntos_cluster) > 0:
            nuevos_centroides[k] = np.mean(puntos_cluster, axis=0)
        else:
            nuevos_centroides[k] = X[np.random.choice(N)]
    return nuevos_centroides

# Implementacion del algoritmo
def kmeans(X, K, max_iter=100, tolerancia=1e-4):
    centroides = inicializacion_centroides(X, K)
    historial_centroides = [centroides.copy()]
    etiquetas = np.zeros(X.shape[0], dtype=int)
    for interacion in range(max_iter):
        # Asignacion de cluster
        etiquetas = asignar_cluster(X, centroides)
        # Recalculando de las posiciones de los centroides
        nuevos_centroides = recalcular_centroides(X, etiquetas, K)
        historial_centroides.append(nuevos_centroides.copy())
        # Verificacion de la condicion de max_iter
        desplazamiento = np.linalg.norm(nuevos_centroides - centroides)
        centroides = nuevos_centroides
        if desplazamiento <= tolerancia:
            print(f"Convergencia alcanzada con exito en la iteracion {interacion}")
            break
    return centroides, etiquetas, historial_centroides

# Función puente que conecta tu algoritmo con FastAPI (main.py)
def ejecutar_kmeans_cluster(request: ClusterRequest) -> ClusterResponse:
    np.random.seed(20)
    X = np.array(request.datos, dtype=float)
    K = request.k_clusters
    max_iter = request.max_iteraciones

    # 1. Ejecutar tu función kmeans
    centroides_finales, etiquetas_finales, _ = kmeans(X, K, max_iter=max_iter)

    # 2. Calcular la inercia WCSS (como en tu libreta)
    inercia_wcss = 0.0
    for k in range(K):
        puntos_k = X[etiquetas_finales == k]
        if len(puntos_k) > 0:
            inercia_wcss += float(np.sum((puntos_k - centroides_finales[k]) ** 2))

    # 3. Devolver la respuesta en el formato exacto de ClusterResponse
    return ClusterResponse(
        estado="exitoso",
        k_utilizado=K,
        num_muestras=int(X.shape[0]),
        centroides_finales=np.round(centroides_finales, 2).tolist(),
        etiquetas_asignadas=etiquetas_finales.tolist(),
        inercia_wcss=round(inercia_wcss, 2)
    )