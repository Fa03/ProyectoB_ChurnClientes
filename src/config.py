"""
Archivo de configuración del proyecto
Contiene rutas, parámetros y constantes globales
"""

import os
from pathlib import Path

# === RUTAS DEL PROYECTO ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# === ARCHIVOS DE DATOS ===
RAW_DATA_FILE = PROCESSED_DATA_DIR / "german_credit.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# === ARCHIVOS DE MODELOS ===
BINARY_MODEL_PATH = MODELS_DIR / "binary_model.h5"
MULTICLASS_MODEL_PATH = MODELS_DIR / "multiclass_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# === PARÁMETROS DE PREPROCESAMIENTO ===
TEST_SIZE = 0.2  # 20% para test
RANDOM_STATE = 42  # Semilla para reproducibilidad
VALIDATION_SPLIT = 0.2  # 20% de train para validación

# === PARÁMETROS DE ENTRENAMIENTO ===

# Modelo Binario
BINARY_EPOCHS = 100
BINARY_BATCH_SIZE = 32
BINARY_LEARNING_RATE = 0.001

# Arquitectura sugerida (ajustar según resultados)
BINARY_HIDDEN_LAYERS = [64, 32, 16]  # Neuronas por capa oculta
BINARY_DROPOUT_RATE = 0.3

# Modelo Multiclase
MULTICLASS_EPOCHS = 100
MULTICLASS_BATCH_SIZE = 32
MULTICLASS_LEARNING_RATE = 0.001

# Arquitectura sugerida
MULTICLASS_HIDDEN_LAYERS = [128, 64, 32]
MULTICLASS_DROPOUT_RATE = 0.3
NUM_CLASSES = 4  # Bajo, Medio, Alto, Crítico

# === COLUMNAS DEL DATASET ===

# Columnas categóricas que necesitan encoding
CATEGORICAL_COLUMNS = [
    'checking_status', 'credit_history', 'purpose', 'savings_status',
    'employment', 'personal_status', 'other_parties', 'property_magnitude',
    'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
]

# Columnas numéricas que necesitan normalización
NUMERICAL_COLUMNS = [
    'duration', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'num_dependents'
]

# Columna objetivo
TARGET_COLUMN = 'class'

# === MAPEO DE CLASES ===

# Para clasificación binaria
# En el dataset original: 1 = Good, 2 = Bad
BINARY_CLASS_MAP = {
    1: 0,  # Good -> 0
    2: 1   # Bad -> 1
}

# Para clasificación multiclase (a definir según análisis)
# Ejemplo: basado en credit_amount y duration
# Bajo: credit bajo y corto plazo
# Medio: credit medio
# Alto: credit alto
# Crítico: credit muy alto y largo plazo
RISK_LEVELS = ['Bajo', 'Medio', 'Alto', 'Crítico']

# === CONFIGURACIÓN DE API ===
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True  # Solo para desarrollo

# === CONFIGURACIÓN DE STREAMLIT ===
STREAMLIT_PORT = 8501

# === LOGGING ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# === EARLY STOPPING ===
EARLY_STOPPING_PATIENCE = 10  # Número de épocas sin mejora antes de parar
EARLY_STOPPING_MIN_DELTA = 0.001  # Mejora mínima considerada significativa

# === CALLBACKS ===
REDUCE_LR_PATIENCE = 5  # Reducir learning rate después de N épocas sin mejora
REDUCE_LR_FACTOR = 0.5  # Factor de reducción del learning rate

# === MÉTRICAS DE EVALUACIÓN ===
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc'
]

if __name__ == "__main__":
    print("=== CONFIGURACIÓN DEL PROYECTO ===")
    print(f"Directorio raíz: {PROJECT_ROOT}")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Directorio de modelos: {MODELS_DIR}")
    print(f"\nArchivo de datos procesados: {RAW_DATA_FILE}")
    print(f"Modelo binario: {BINARY_MODEL_PATH}")
    print(f"Modelo multiclase: {MULTICLASS_MODEL_PATH}")
