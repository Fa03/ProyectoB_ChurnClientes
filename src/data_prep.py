"""
Módulo de preprocesamiento de datos
Contiene funciones para limpieza, transformación y preparación de datos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))
from src import config

def load_data(filepath=None):
    """
    Carga el dataset desde un archivo CSV
    
    Args:
        filepath: Ruta al archivo CSV. Si es None, usa config.RAW_DATA_FILE
    
    Returns:
        DataFrame con los datos cargados
    """
    if filepath is None:
        filepath = config.RAW_DATA_FILE
    
    print(f"Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df

def handle_missing_values(df):
    """
    Maneja valores faltantes en el dataset
    
    Args:
        df: DataFrame con posibles valores faltantes
    
    Returns:
        DataFrame sin valores faltantes
    """
    # TODO: Implementar estrategia de manejo de valores faltantes
    # Opciones:
    # - Eliminar filas con valores faltantes
    # - Imputar con media/mediana para numéricas
    # - Imputar con moda para categóricas
    
    missing = df.isnull().sum()
    if missing.any():
        print("\nValores faltantes encontrados:")
        print(missing[missing > 0])
        # Implementar lógica aquí
    else:
        print("\nNo se encontraron valores faltantes")
    
    return df

def encode_categorical_features(df, fit=True, encoders=None):
    """
    Codifica variables categóricas usando Label Encoding
    
    Args:
        df: DataFrame con columnas categóricas
        fit: Si True, ajusta nuevos encoders. Si False, usa encoders existentes
        encoders: Diccionario de encoders pre-entrenados (si fit=False)
    
    Returns:
        DataFrame con variables codificadas, diccionario de encoders
    """
    df_encoded = df.copy()
    
    if fit:
        encoders = {}
        for col in config.CATEGORICAL_COLUMNS:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                print(f"Codificada columna '{col}': {len(le.classes_)} clases únicas")
    else:
        if encoders is None:
            raise ValueError("Debe proporcionar encoders cuando fit=False")
        for col in config.CATEGORICAL_COLUMNS:
            if col in df_encoded.columns and col in encoders:
                df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))
    
    return df_encoded, encoders

def scale_numerical_features(df, fit=True, scaler=None):
    """
    Normaliza variables numéricas usando StandardScaler
    
    Args:
        df: DataFrame con columnas numéricas
        fit: Si True, ajusta nuevo scaler. Si False, usa scaler existente
        scaler: Scaler pre-entrenado (si fit=False)
    
    Returns:
        DataFrame con variables normalizadas, scaler utilizado
    """
    df_scaled = df.copy()
    numerical_cols = [col for col in config.NUMERICAL_COLUMNS if col in df.columns]
    
    if fit:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print(f"\nNormalizadas {len(numerical_cols)} columnas numéricas")
    else:
        if scaler is None:
            raise ValueError("Debe proporcionar scaler cuando fit=False")
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df_scaled, scaler

def create_binary_target(df):
    """
    Crea variable objetivo binaria (Good/Bad credit)
    
    Args:
        df: DataFrame con columna 'class'
    
    Returns:
        Serie con target binario (0=Good, 1=Bad)
    """
    # En el dataset original: 1 = Good credit, 2 = Bad credit
    # Convertir a: 0 = Good, 1 = Bad
    target = df[config.TARGET_COLUMN].map(config.BINARY_CLASS_MAP)
    
    print("\nDistribución de clases binarias:")
    print(target.value_counts())
    print(f"Proporción clase positiva (Bad): {target.mean():.2%}")
    
    return target

def create_multiclass_target(df):
    """
    Crea variable objetivo multiclase (niveles de riesgo)
    
    Args:
        df: DataFrame con información de crédito
    
    Returns:
        Serie con target multiclase (0=Bajo, 1=Medio, 2=Alto, 3=Crítico)
    """
    # TODO: Implementar lógica de clasificación de riesgo
    # Sugerencias:
    # - Bajo riesgo: monto bajo + duración corta + buen historial
    # - Medio riesgo: características intermedias
    # - Alto riesgo: monto alto o duración larga
    # - Crítico: combinación de factores de alto riesgo
    
    # EJEMPLO (ajustar según análisis en EDA):
    # Usar cuartiles de credit_amount y duration
    risk_level = np.zeros(len(df), dtype=int)
    
    # Calcular percentiles
    amount_q25 = df['credit_amount'].quantile(0.25)
    amount_q75 = df['credit_amount'].quantile(0.75)
    duration_q25 = df['duration'].quantile(0.25)
    duration_q75 = df['duration'].quantile(0.75)
    
    # Lógica de clasificación (EJEMPLO - personalizar)
    # Bajo: monto bajo y duración corta
    mask_low = (df['credit_amount'] <= amount_q25) & (df['duration'] <= duration_q25)
    risk_level[mask_low] = 0
    
    # Alto: monto alto y duración larga
    mask_high = (df['credit_amount'] >= amount_q75) & (df['duration'] >= duration_q75)
    risk_level[mask_high] = 2
    
    # Crítico: monto muy alto Y clase original = Bad
    mask_critical = (df['credit_amount'] > df['credit_amount'].quantile(0.90)) & \
                    (df[config.TARGET_COLUMN] == 2)
    risk_level[mask_critical] = 3
    
    # Medio: todo lo demás
    mask_medium = (risk_level == 0) & ~mask_low & ~mask_high & ~mask_critical
    risk_level[mask_medium] = 1
    
    risk_series = pd.Series(risk_level, index=df.index)
    
    print("\nDistribución de niveles de riesgo:")
    for i, level in enumerate(config.RISK_LEVELS):
        count = (risk_series == i).sum()
        pct = count / len(risk_series) * 100
        print(f"{level} ({i}): {count} ({pct:.1f}%)")
    
    return risk_series

def split_data(X, y, test_size=None, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba
    
    Args:
        X: Features
        y: Target
        test_size: Proporción para test (default: config.TEST_SIZE)
        random_state: Semilla aleatoria (default: config.RANDOM_STATE)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantener proporciones de clases
    )
    
    print(f"\nDatos divididos:")
    print(f"Train: {X_train.shape[0]} muestras")
    print(f"Test: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def save_preprocessors(scaler, encoders, scaler_path=None, encoders_path=None):
    """
    Guarda objetos de preprocesamiento
    
    Args:
        scaler: StandardScaler entrenado
        encoders: Diccionario de LabelEncoders
        scaler_path: Ruta para guardar scaler
        encoders_path: Ruta para guardar encoders
    """
    if scaler_path is None:
        scaler_path = config.SCALER_PATH
    if encoders_path is None:
        encoders_path = config.LABEL_ENCODER_PATH
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoders_path)
    
    print(f"\nPreprocessors guardados:")
    print(f"Scaler: {scaler_path}")
    print(f"Encoders: {encoders_path}")

def load_preprocessors(scaler_path=None, encoders_path=None):
    """
    Carga objetos de preprocesamiento guardados
    
    Args:
        scaler_path: Ruta del scaler guardado
        encoders_path: Ruta de los encoders guardados
    
    Returns:
        scaler, encoders
    """
    if scaler_path is None:
        scaler_path = config.SCALER_PATH
    if encoders_path is None:
        encoders_path = config.LABEL_ENCODER_PATH
    
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    print(f"\nPreprocessors cargados:")
    print(f"Scaler: {scaler_path}")
    print(f"Encoders: {encoders_path}")
    
    return scaler, encoders

# Función principal de preprocesamiento
def preprocess_pipeline(df, fit=True, target_type='binary', 
                        scaler=None, encoders=None, save=True):
    """
    Pipeline completo de preprocesamiento
    
    Args:
        df: DataFrame a procesar
        fit: Si True, ajusta preprocessors. Si False, usa existentes
        target_type: 'binary' o 'multiclass'
        scaler: Scaler pre-entrenado (si fit=False)
        encoders: Encoders pre-entrenados (si fit=False)
        save: Si True, guarda preprocessors entrenados
    
    Returns:
        X_processed, y, scaler, encoders
    """
    print("="*60)
    print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
    print("="*60)
    
    # 1. Manejar valores faltantes
    df_clean = handle_missing_values(df.copy())
    
    # 2. Crear target
    if target_type == 'binary':
        y = create_binary_target(df_clean)
    elif target_type == 'multiclass':
        y = create_multiclass_target(df_clean)
    else:
        raise ValueError("target_type debe ser 'binary' o 'multiclass'")
    
    # 3. Separar features (eliminar columna target)
    X = df_clean.drop(columns=[config.TARGET_COLUMN])
    
    # 4. Codificar variables categóricas
    X_encoded, encoders = encode_categorical_features(X, fit=fit, encoders=encoders)
    
    # 5. Normalizar variables numéricas
    X_scaled, scaler = scale_numerical_features(X_encoded, fit=fit, scaler=scaler)
    
    # 6. Guardar preprocessors si es necesario
    if fit and save:
        save_preprocessors(scaler, encoders)
    
    print("\n" + "="*60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"Shape final: {X_scaled.shape}")
    print(f"Features: {X_scaled.shape[1]}")
    
    return X_scaled, y, scaler, encoders

if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de preprocesamiento de datos")
    print("Importar este módulo en notebooks o scripts de entrenamiento")
