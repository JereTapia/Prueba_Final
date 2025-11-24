# src/process.py
import pandas as pd
import numpy as np

# Rutas de entrada y salida
INPUT_FILE = 'data/raw/telco_churn.csv'
OUTPUT_FILE = 'data/processed/processed.csv'

# Nombre de la columna clave basado en el diagnóstico
TOTAL_CHARGES_COL = 'total_charges' 
CHURN_COL = 'churn' # Nombre de la columna objetivo

def process_data():
    print(f"Leyendo datos desde: {INPUT_FILE}")
    
    # 1. Cargar el dataset
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("ERROR: Archivo de entrada no encontrado. Asegúrate de que telco_churn.csv esté en data/raw/.")
        return
    
    # 2. **LIMPIEZA Y CORRECCIÓN DE TOTAL CHARGES**
    print(f"Procesando columna '{TOTAL_CHARGES_COL}'...")
    
    # Reemplazamos los espacios (' ') por NaN. Este dataset puede tener espacios en blanco.
    df[TOTAL_CHARGES_COL] = df[TOTAL_CHARGES_COL].replace(' ', np.nan) 
    
    # Eliminamos las filas con NaN (los pocos registros que no tienen cargos totales)
    df.dropna(subset=[TOTAL_CHARGES_COL], inplace=True) 
    
    # Convertimos la columna al tipo de dato numérico (float)
    df[TOTAL_CHARGES_COL] = pd.to_numeric(df[TOTAL_CHARGES_COL]) 

    # 3. Ingeniería de Características (Ejemplo simple)
    # Convertimos 'age' a una categoría simple (ejemplo, mayores de 50)
    df['is_old_customer'] = df['age'].apply(lambda x: 'Yes' if x > 50 else 'No')
    
    # 4. Selección de un subconjunto de columnas
    cols_to_keep = ['age', 'gender', 'region', 'contract_type', 'tenure_months', 
                    'monthly_charges', TOTAL_CHARGES_COL, 'internet_service', 
                    'payment_method', 'is_old_customer', CHURN_COL]
    df_processed = df[cols_to_keep]
    
    # 5. Guardar los datos procesados
    print(f"Guardando {len(df_processed)} filas procesadas en: {OUTPUT_FILE}")
    df_processed.to_csv(OUTPUT_FILE, index=False)
    print("Preprocesamiento completado exitosamente.")

if __name__ == '__main__':
    process_data()