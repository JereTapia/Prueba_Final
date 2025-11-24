# src/train.py (CORREGIDO - Buscando la columna objetivo correcta)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import json
import pickle
import os

INPUT_FILE = 'data/processed/processed.csv'
MODEL_OUTPUT = 'model/model.pkl'
METRICS_OUTPUT = 'metrics/metrics.json'
TARGET_COL_ORIGINAL = 'churn' # Nombre original de la columna target

def train_model():
    print(f"Leyendo datos procesados desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Codificación One-Hot
    df = pd.get_dummies(df, drop_first=True)
    
    # 2. **BÚSQUEDA ROBUSTA DE LA COLUMNA TARGET BINARIA**
    # Si la columna 'churn' es categórica y tenía 'Yes/No', se transforma en 'churn_Yes'.
    # Si la columna 'churn' ya era 0/1, no se transforma, pero le ponemos un nombre para drop.
    
    if f"{TARGET_COL_ORIGINAL}_Yes" in df.columns:
        TARGET_BINARIA = f"{TARGET_COL_ORIGINAL}_Yes"
    elif TARGET_COL_ORIGINAL in df.columns:
        TARGET_BINARIA = TARGET_COL_ORIGINAL
    else:
        # Esto debería capturar el 99% de los casos. Si sigue fallando, la columna no se llama 'churn'.
        print(f"ERROR: No se encontró la columna objetivo '{TARGET_COL_ORIGINAL}' o '{TARGET_COL_ORIGINAL}_Yes'.")
        print("Columnas disponibles:", df.columns.tolist())
        return

    # Separar la columna objetivo (y) de las características (X)
    X = df.drop(columns=[TARGET_BINARIA], errors='ignore') 
    y = df[TARGET_BINARIA]
    
    # 3. División de datos (20% para prueba, 80% para entrenamiento)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenamiento del Modelo
    print("Entrenando modelo de Regresión Logística...")
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 5. Evaluación del Modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- Resultados del Entrenamiento ---")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Puntuación F1 (F1 Score): {f1:.4f}")
    
    # 6. Guardar Métricas
    metrics = {'accuracy': accuracy, 'f1_score': f1}
    with open(METRICS_OUTPUT, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {METRICS_OUTPUT}")
        
    # 7. Guardar Modelo Serializado
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {MODEL_OUTPUT}")

if __name__ == '__main__':
    # Aseguramos que las carpetas de salida existan
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    
    train_model()