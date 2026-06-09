"""
Script para entrenar el modelo de clasificación binaria
Predice: Good credit (0) vs Bad credit (1)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import config
from src.data_prep import load_data, preprocess_pipeline, split_data

def build_binary_model(input_dim):
    """
    Construye el modelo de clasificación binaria
    
    Args:
        input_dim: Número de features de entrada
    
    Returns:
        Modelo compilado
    """
    # TODO: Personalizar arquitectura según resultados de experimentación
    
    model = keras.Sequential([
        # Capa de entrada
        layers.Input(shape=(input_dim,)),
        
        # Primera capa oculta
        layers.Dense(config.BINARY_HIDDEN_LAYERS[0], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.BINARY_DROPOUT_RATE),
        
        # Segunda capa oculta
        layers.Dense(config.BINARY_HIDDEN_LAYERS[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.BINARY_DROPOUT_RATE),
        
        # Tercera capa oculta
        layers.Dense(config.BINARY_HIDDEN_LAYERS[2], activation='relu'),
        layers.Dropout(config.BINARY_DROPOUT_RATE),
        
        # Capa de salida (binaria)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.BINARY_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo con callbacks
    
    Args:
        model: Modelo a entrenar
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
    
    Returns:
        Modelo entrenado, historia del entrenamiento
    """
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        verbose=1
    )
    
    # Entrenar
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        epochs=config.BINARY_EPOCHS,
        batch_size=config.BINARY_BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba
    
    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de prueba
    """
    print("\n" + "="*60)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("="*60)
    
    # Predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Métricas
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, 
                                target_names=['Good Credit', 'Bad Credit']))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n--- ROC-AUC Score ---")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Evaluación con método evaluate
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(X_test, y_test)
    print(f"\n--- Test Metrics ---")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"AUC: {test_auc:.4f}")

def plot_training_history(history):
    """
    Visualiza las curvas de entrenamiento
    
    Args:
        history: Historia del entrenamiento
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(config.MODELS_DIR / 'binary_model_history.png')
    print(f"\nGráficas guardadas en: {config.MODELS_DIR / 'binary_model_history.png'}")
    plt.show()

def main():
    """Función principal de entrenamiento"""
    print("="*60)
    print("ENTRENAMIENTO: MODELO DE CLASIFICACIÓN BINARIA")
    print("="*60)
    
    # 1. Cargar datos
    df = load_data()
    
    # 2. Preprocesar
    X, y, scaler, encoders = preprocess_pipeline(
        df, 
        fit=True, 
        target_type='binary',
        save=True
    )
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Construir modelo
    print("\n--- Arquitectura del Modelo ---")
    model = build_binary_model(input_dim=X_train.shape[1])
    model.summary()
    
    # 5. Entrenar
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # 6. Evaluar
    evaluate_model(model, X_test, y_test)
    
    # 7. Visualizar historia
    plot_training_history(history)
    
    # 8. Guardar modelo
    model.save(config.BINARY_MODEL_PATH)
    print(f"\n✓ Modelo guardado en: {config.BINARY_MODEL_PATH}")
    
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)

if __name__ == "__main__":
    main()
