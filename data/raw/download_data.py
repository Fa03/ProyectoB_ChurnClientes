"""
Script para descargar el dataset German Credit Data
Dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Ejecutar desde la raíz del proyecto:
python data/raw/download_data.py
"""

import urllib.request
import zipfile
import os
import pandas as pd

# URLs del dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

# Directorio de destino
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def download_file(url, destination):
    """Descarga un archivo desde una URL"""
    print(f"Descargando {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Guardado en: {destination}")

def process_german_credit():
    """
    Procesa el dataset German Credit Data
    El dataset original no tiene headers, hay que añadirlos manualmente
    """
    
    # Nombres de las columnas según la documentación
    column_names = [
        'checking_status', 'duration', 'credit_history', 'purpose', 
        'credit_amount', 'savings_status', 'employment', 'installment_rate',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits',
        'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class'
    ]
    
    # Leer el archivo de datos
    data_path = os.path.join(RAW_DIR, "german.data")
    
    # El dataset usa espacios como delimitador
    df = pd.read_csv(data_path, sep=' ', header=None, names=column_names)
    
    print(f"\nDataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head())
    
    # Información básica
    print("\nInformación del dataset:")
    print(df.info())
    
    # Guardar versión procesada
    processed_path = os.path.join(PROCESSED_DIR, "german_credit.csv")
    df.to_csv(processed_path, index=False)
    print(f"\nDataset guardado en: {processed_path}")
    
    return df

def main():
    """Función principal para descargar y procesar el dataset"""
    
    # Crear directorios si no existen
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("="*60)
    print("DESCARGA DE DATASET: German Credit Data")
    print("="*60)
    
    # Descargar archivos
    data_dest = os.path.join(RAW_DIR, "german.data")
    doc_dest = os.path.join(RAW_DIR, "german.doc")
    
    if not os.path.exists(data_dest):
        download_file(DATA_URL, data_dest)
    else:
        print(f"El archivo ya existe: {data_dest}")
    
    if not os.path.exists(doc_dest):
        download_file(NAMES_URL, doc_dest)
    else:
        print(f"El archivo ya existe: {doc_dest}")
    
    # Procesar datos
    print("\n" + "="*60)
    print("PROCESANDO DATASET")
    print("="*60)
    df = process_german_credit()
    
    print("\n" + "="*60)
    print("¡DESCARGA COMPLETADA!")
    print("="*60)
    print(f"\nArchivos descargados en: {RAW_DIR}/")
    print(f"Dataset procesado en: {PROCESSED_DIR}/german_credit.csv")
    print("\nPuedes empezar con el análisis exploratorio en notebooks/01_EDA_CreditRisk.ipynb")

if __name__ == "__main__":
    main()
