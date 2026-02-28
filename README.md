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

## F) Checklist de entrega

- [x] Repositorio publico con codigo, dataset, scripts y reportes.
- [x] Etapas del ciclo ML cubiertas desde datos hasta inferencia.
- [x] Modelo serializado y servido por API REST.
- [x] README usado como documentacion central.
- [ ] Crear Pull Request de `dev/final-project` a `main`.
- [ ] Enviar URL del repositorio en el formulario del curso.

