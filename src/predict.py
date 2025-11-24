# src/predict.py
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = 'model/model.pkl'

def load_model(path):
    """Carga el modelo serializado."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo cargado desde: {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Modelo no encontrado en {path}. Asegúrate de haberlo subido.")
        return None

def predict_new_data(model, new_data):
    """Hace predicciones en un DataFrame de entrada."""
    
    # 1. Aplicar el mismo preprocesamiento que en train.py
    # La codificación One-Hot debe ser la misma que la usada en el entrenamiento
    df = pd.DataFrame([new_data])
    df = pd.get_dummies(df, drop_first=True)
    
    # 2. Asegurar que las columnas coincidan con el entrenamiento (rellenar con ceros si faltan)
    # NOTA: En un despliegue real, necesitarías guardar la lista exacta de columnas de X_train.
    
    # Aquí simulamos un caso de prueba simple
    required_cols = ['age', 'gender_Male', 'region_Region A', 'contract_type_Month-to-month', 
                     'tenure_months', 'monthly_charges', 'total_charges', 
                     'internet_service_DSL', 'payment_method_Electronic check', 
                     'is_old_customer_Yes']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    X_new = df[required_cols]

    # 3. Predicción
    prediction = model.predict(X_new)[0]
    
    return "Churn" if prediction == 1 else "No Churn"

if __name__ == '__main__':
    # Ejemplo de datos de un nuevo cliente (DEBE COINCIDIR CON LAS COLUMNAS DE TU CSV)
    sample_client = {
        'age': 35,
        'gender': 'Female',
        'region': 'Region A', 
        'contract_type': 'Two year',
        'tenure_months': 45,
        'monthly_charges': 80.5,
        'total_charges': 3622.5,
        'internet_service': 'DSL',
        'payment_method': 'Credit card (automatic)',
        'is_old_customer': 'No'
    }
    
    model = load_model(MODEL_PATH)
    
    if model:
        result = predict_new_data(model, sample_client)
        print("\n--- Predicción de Cliente ---")
        print(f"Datos del cliente: {sample_client}")
        print(f"Predicción de Churn: {result}")
        print("---------------------------\n")