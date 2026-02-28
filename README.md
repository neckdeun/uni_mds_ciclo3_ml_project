# Proyecto Final MLOps - Prediccion de Riesgo de Accidente Cerebrovascular ACV

La descripcion oficial del trabajo se encuentra en [final_project_description.md](final_project_description.md).

Datos del estudiante:
- Nombre completo: Necker Vasquez Mauricio
- Correo: necker.vasquez.m@uni.pe
- Grupo: Grupo o Aula 2

## A) Definicion del problema

### Caso de uso
Este proyecto estima el riesgo de accidente cerebrovascular (ACV) (`stroke` = 0/1) a partir de variables demograficas y clinicas como edad, hipertension, enfermedad cardiaca, IMC, nivel de glucosa y habitos de tabaquismo.

### Contexto y objetivo
La deteccion temprana de personas con mayor riesgo puede apoyar decisiones preventivas en salud. El objetivo es construir un flujo reproducible de machine learning que reciba datos de un paciente y devuelva una probabilidad de riesgo.

### Restricciones
- El dataset esta desbalanceado (hay pocos casos positivos de ACV).
- Es un proyecto academico; las predicciones no reemplazan una evaluacion medica.

### Adquisicion de datos
- Fuente principal (Kaggle): [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Archivo crudo usado en el repositorio: `data/raw/healthcare-dataset-stroke-data.csv`

### Variables (features) del dataset
- `gender`: genero del paciente (`Male`, `Female`, `Other`).
- `age`: edad en anios.
- `hypertension`: presencia de hipertension (`0` no, `1` si).
- `heart_disease`: presencia de enfermedad cardiaca (`0` no, `1` si).
- `ever_married`: si estuvo casado alguna vez (`Yes`/`No`).
- `work_type`: tipo de trabajo (`Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked`).
- `Residence_type`: zona de residencia (`Urban` o `Rural`).
- `avg_glucose_level`: nivel promedio de glucosa.
- `bmi`: indice de masa corporal.
- `smoking_status`: habito de fumador (`never smoked`, `formerly smoked`, `smokes`, `Unknown`).
- `stroke` (target): variable objetivo (`0` no ACV, `1` ACV).

## B) Preparacion del proyecto

- Repositorio publico en GitHub clonado localmente.
- Desarrollo realizado en la rama `dev/final-project`.
- Estructura modular creada para datos, scripts, modelos, reportes y pruebas.

## C) Experimentacion de ML

- Se realizo division train/test con estratificacion por la variable objetivo.
- Se entreno un pipeline base de clasificacion:
  - Preprocesamiento numerico: imputacion por mediana + escalado.
  - Preprocesamiento categorico: imputacion por moda + one-hot encoding.
  - Modelo: Regresion Logistica con balanceo de clases.
- Se tomo este pipeline como modelo champion por ser reproducible, interpretable y suficiente para el alcance del curso.

Artefactos generados:
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/data_summary.json`
- `reports/inference_evidence.md`
- `experiments/model_comparison.csv`
- `experiments/model_comparison.md`

Metricas actuales (conjunto de prueba):
- Accuracy: `0.7456`
- Precision: `0.1379`
- Recall: `0.8000`
- F1-score: `0.2353`
- ROC-AUC: `0.8437`

### Resultados de prediccion (ejemplo de inferencia)
- Caso de prueba usado en la API:
  - edad `67`, hipertension `1`, glucosa `228.69`, IMC `36.6`, exfumador.
- Respuesta del servicio:
  - `prediction = 1`
  - `stroke_risk_probability = 0.872199`
- Interpretacion: el modelo asigna una probabilidad alta de riesgo para un perfil con varios factores asociados.

## D) Actividades de desarrollo ML

### Script de preparacion de datos
- Archivo: `src/data_preparation.py`
- Tareas:
  - Cargar CSV crudo.
  - Eliminar la columna `id` por no aportar al entrenamiento.
  - Dividir en train y test con estratificacion.
  - Guardar salidas en `data/training/train.csv` y `data/training/test.csv`.

### Script de entrenamiento
- Archivo: `src/train.py`
- Tareas:
  - Construir pipeline de preprocesamiento + modelo.
  - Entrenar y evaluar sobre test.
  - Serializar el pipeline entrenado en `models/stroke_model.joblib`.
  - Guardar metricas y reporte de clasificacion en `reports/`.

## E) Despliegue y serving del modelo

### Estrategia de serving
- API REST con FastAPI en `src/serving.py`.

### Endpoints
- `GET /health`: verificacion de estado del servicio.
- `POST /predict`: recibe las variables del paciente y devuelve:
  - `prediction` (0/1)
  - `stroke_risk_probability` (float)

## Instrucciones de ejecucion

### 1) Instalar dependencias
```bash
python -m pip install -r requirements.txt
```

### 2) Preparar datos
```bash
python src/data_preparation.py
```

### 3) Entrenar modelo
```bash
python src/train.py
```

### 4) Levantar API
```bash
uvicorn src.serving:app --reload
```

### 5) Probar inferencia
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"gender\":\"Male\",\"age\":67,\"hypertension\":1,\"heart_disease\":0,\"ever_married\":\"Yes\",\"work_type\":\"Private\",\"Residence_type\":\"Urban\",\"avg_glucose_level\":228.69,\"bmi\":36.6,\"smoking_status\":\"formerly smoked\"}"
```

### 6) Ejecutar comparacion de modelos (carpeta experiments)
```bash
python experiments/run_experiments.py
```

### 7) Ejecutar pruebas del API (carpeta tests)
```bash
python -m pytest tests/test_serving.py -q
```

## Conclusiones

El flujo completo se pudo implementar de forma estable: desde adquisicion de datos, preparacion, entrenamiento y evaluacion, hasta el despliegue de un endpoint de prediccion. Se logro una buena capacidad de discriminacion del modelo (ROC-AUC de `0.8437`) y un recall alto en la clase positiva (`0.8000`), que en este contexto es valioso porque prioriza detectar casos de mayor riesgo.

## Insights obtenidos

- El dataset esta claramente desbalanceado, por eso se uso `class_weight=\"balanced\"`.
- La variable `bmi` concentra los valores faltantes, lo que justifico imputacion por mediana.
- El modelo de Regresion Logistica rindio mejor en ROC-AUC que RandomForest para este conjunto y configuracion.
- La API de inferencia permite cerrar el ciclo MLOps y mostrar uso practico del modelo.

## Limitaciones

- El dataset tiene tamano moderado y no representa toda la complejidad clinica real.
- La precision es baja en la clase positiva, por lo que habria falsos positivos.
- No se incorporaron variables clinicas adicionales (antecedentes mas detallados, examenes especializados, etc.).
- Este trabajo es academico y no debe usarse como sistema de diagnostico medico.

## Mejoras futuras

- Probar tecnicas de balanceo como SMOTE o ajuste de umbral de decision.
- Hacer busqueda de hiperparametros sistematica y validacion cruzada.
- Incorporar trazabilidad de experimentos con MLflow.
- Agregar despliegue con contenedor Docker y pipeline CI/CD.
- Incluir monitoreo de drift y reentrenamiento periodico.

## Lecciones aprendidas

- Una buena estructura de repositorio facilita el trabajo y la evaluacion.
- Separar scripts de preparacion, entrenamiento y serving mejora mantenibilidad.
- Documentar cada etapa evita vacios al momento de la entrega final.
- Probar la API con casos reales da evidencia clara de funcionamiento end-to-end.

## F) Checklist de entrega

- [x] Repositorio publico con codigo, dataset, scripts y reportes.
- [x] Etapas del ciclo ML cubiertas desde datos hasta inferencia.
- [x] Modelo serializado y servido por API REST.
- [x] README usado como documentacion central.
- [x] Crear Pull Request de `dev/final-project` a `main`.
- [ ] Enviar URL del repositorio en el formulario del curso.

