## Arquitectura y Estructura del Proyecto
El proyecto sigue una arquitectura modular con punto de acceso único
(main.py), validación estricta de datos mediante esquemas **Pydantic**
(models.py) y separación de la lógica matemática y de modelos en
"services.py":
proyecto_final_ia/
├── dataset_cardiaco_simulado.csv # Conjunto de datos clínicos para
entrenar el Árbol de Decisión
├── main.py # Punto de acceso único:
inicialización de FastAPI y endpoints
├── models.py # Esquemas Pydantic (Request /
Response) con validación de rangos
├── services.py # Lógica de negocio: Algoritmo
Genético, Árbol de Decisión y K-Means
├── requirements.txt # Dependencias del proyecto
├── .gitignore # Exclusión de entorno virtual
(venv/) y caché (__pycache__/)
└── README.md # Documentación técnica y guía
de uso
Instalación y Configuración
Sigue estos pasos para ejecutar el proyecto de manera local:
1. Clonar el repositorio
Si es estas en github clonalo desde el repositorio de este
Bash
git clone <https://github.com/utp0211576-debug/proyecto_final_ia.git>
cd proyecto_final_ia
2. Crear y activar el entorno virtual
--En Windows (PowerShell):
PowerShell
python -m venv venv
.\venv\Scripts\activate
source venv/bin/activate
3. Instalar dependencias
Bash
pip install -r requirements.txt
Ejecución del Servidor
Con el entorno virtual (venv) activo y ubicado en la carpeta raíz donde
se encuentra main.py, inicia el servidor con Uvicorn:
Bash
uvicorn main:app --reload
Documentación interactiva (Swagger UI): http://127.0.0.1:8000/docs
Documentación de Endpoints y Ejemplos de Prueba
Todas las peticiones reciben y devuelven datos exclusivamente en
formato application/json.
1. Algoritmo Genético (POST /genetico)
Reconstruye una cadena de texto objetivo mediante selección, cruce y
mutación genética.
Ejemplo JSON
{
"mensaje_oculto": "IA EVOLUTIVA 2026",
"tam_poblacion": 100,
"max_generaciones": 500,
"prob_cruce": 0.8,
"prob_mutacion": 0.05
}
Ejemplo con cURL:
curl -X POST "[http://127.0.0.1:8000/genetico](http://127.0.0.1:8000/
genetico)" \
-H "Content-Type: application/json" \
-d "{\"mensaje_oculto\": \"IA EVOLUTIVA 2026\", \"tam_poblacion\":
100, \"max_generaciones\": 500, \"prob_cruce\": 0.8, \"prob_mutacion\":
0.05}"
2. Aprendizaje Supervisado - Árbol de Decisión (POST /arbol_decision)
Evalúa los parámetros clínicos de un paciente utilizando un
clasificador de Árbol de Decisión (criterion='entropy', max_depth=3)
entrenado con datos cardiovasculares para predecir el riesgo de
cardiopatía.
Ejemplo JSON

{
"edad": 55,
"presion_sistolica": 145,
"colesterol": 240,
"frecuencia_max": 130,
"angina_ejercicio": 1
}
Ejemplo cURL
curl -X POST "[http://127.0.0.1:8000/arbol_decision](http://
127.0.0.1:8000/arbol_decision)" \
-H "Content-Type: application/json" \
-d "{\"edad\": 55, \"presion_sistolica\": 145, \"colesterol\": 240,
\"frecuencia_max\": 130, \"angina_ejercicio\": 1}"
3. Aprendizaje No Supervisado - Clustering K-Means (POST /
cluster)Agrupa una matriz dinámica de muestras numéricas sin etiquetar
en $K$ clústeres, devolviendo los centroides finales, las etiquetas
asignadas y la inercia total (WCSS).
Ejmplo JSON:
{
"k_clusters": 3,
"max_iteraciones": 100,
"datos": [
[10.2, 12.5],
[11.0, 10.8],
[9.8, 11.9],
[50.1, 52.3],
[52.0, 49.8],
[48.9, 51.0],
[90.5, 15.2],
[88.7, 14.0],
[91.2, 16.5]
]
}
Ejemplo cURL:
curl -X POST "[http://127.0.0.1:8000/cluster](http://127.0.0.1:8000/
cluster)" \
-H "Content-Type: application/json" \
-d "{\"k_clusters\": 3, \"max_iteraciones\": 100, \"datos\": [[10.2,
12.5], [11.0, 10.8], [9.8, 11.9], [50.1, 52.3], [52.0, 49.8], [48.9,
51.0], [90.5, 15.2], [88.7, 14.0], [91.2, 16.5]]}"
**Nota para Windows PowerShell:** Si ejecutas las pruebas desde
PowerShell, sustituye `curl` por `curl.exe --%` (como se muestra en los

bloques de PowerShell) para evitar que la consola modifique las
comillas internas del JSON.

*************models.py***********
class ArbolDecisionRequest(BaseModel):
edad: int = Field(..., ge=18, le=100)
presion_sistolica: int = Field(..., ge=80, le=220)
colesterol: int = Field(..., ge=100, le=500)
frecuencia_max: int = Field(..., ge=60, le=220)
angina_ejercicio: int = Field(..., ge=0, le=1)
class ArbolDecisionResponse(BaseModel):
estado: str
paciente_evaluado: ArbolDecisionRequest
clase_predicha: int
diagnostico_texto: str
probabilidad_riesgo: float
nivel_riesgo: str

************main.py*****************
@app.post("/arbol_decision", response_model=ArbolDecisionResponse)
def algoritmo_arbol_decision(request: ArbolDecisionRequest):
return services.evaluar_paciente_arbol(request)

************services.py********************
#3.2 Endpoint 2: ML supervisado - Árbol de decisión (POST /
arbol_decision)
def entrenar_arbol_decision():
SEED = 10
ruta_csv = os.path.join(os.path.dirname(__file__),
"dataset_cardiaco_simulado.csv")
if not os.path.exists(ruta_csv):
raise FileNotFoundError(
f"No se encontró 'dataset_cardiaco_simulado.csv' en
{os.path.dirname(__file__)}. "
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
def evaluar_paciente_arbol(paciente: ArbolDecisionRequest) ->
ArbolDecisionResponse:
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
def evaluar_paciente_arbol(paciente: ArbolDecisionRequest) ->
ArbolDecisionResponse:
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

Además se debe tener el documento dataset_cardiaco_simulado.csv para el entrenamiento correspondiente 

--En Mac / Linux:
Bash
python3 -m ven
