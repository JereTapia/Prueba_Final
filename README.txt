# 🚀 Proyecto MLOps: Predicción de Churn de Clientes

Este proyecto implementa un flujo de trabajo MLOps completo utilizando Git para el código y DVC (Data Version Control) con DagsHub para la versionado de datos, modelos y métricas.

## Flujo de Trabajo (Pipeline DVC)

El pipeline está definido en `dvc.yaml` y consta de dos etapas principales:

1.  **Etapa `process` (Preprocesamiento):**
    * **Input:** `data/raw/telco_churn.csv`
    * **Proceso:** Script `src/process.py` (limpieza, ingeniería de características).
    * **Output:** `data/processed/processed.csv` (Datos limpios).

2.  **Etapa `train` (Entrenamiento):**
    * **Input:** `data/processed/processed.csv` y `src/train.py`
    * **Proceso:** Entrenamiento de un modelo de Regresión Logística.
    * **Output:** `model/model.pkl` (Modelo serializado) y `metrics/metrics.json` (Métricas).

## Resultados del Modelo

Las métricas finales del modelo de Regresión Logística están disponibles en `metrics/metrics.json`.

* **Accuracy:** 0.6905
* **F1 Score:** 0.5098

## Monitoreo y Mantenimiento (Próximos Pasos)

Para actualizar el modelo (Mantenimiento), simplemente:
1.  Modifica el script `src/train.py` (ej. cambiando el algoritmo o hiperparámetros).
2.  Ejecuta `dvc repro` para regenerar los datos, el modelo y las métricas.
3.  Compara las nuevas métricas en DagsHub (pestaña Experiments).