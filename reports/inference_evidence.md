# Evidencia de inferencia

Respuesta del endpoint de salud:
- `{"status":"ok"}`

Payload enviado para prediccion:
```json
{
  "gender": "Male",
  "age": 67,
  "hypertension": 1,
  "heart_disease": 0,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

Respuesta de la prediccion:
```json
{
  "prediction": 1,
  "stroke_risk_probability": 0.872199
}
```
