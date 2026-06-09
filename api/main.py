"""
API REST con FastAPI para el Sistema de Predicción de Riesgo Crediticio

Endpoints:
- GET /: Información de la API
- POST /predict/binary: Predicción binaria (Good/Bad credit)
- POST /predict/risk_level: Predicción de nivel de riesgo
- GET /health: Estado del servidor

Ejecutar: uvicorn main:app --reload
Documentación: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import sys
from pathlib import Path

# TODO: Descomentar cuando los modelos estén entrenados
# from tensorflow import keras
# import joblib

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Predicción de Riesgo Crediticio",
    description="API para predicción de riesgo crediticio usando ANN",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELOS DE DATOS (SCHEMAS) ===

class CreditApplication(BaseModel):
    """Schema para solicitud de crédito"""
    # TODO: Ajustar según las columnas reales del dataset
    checking_status: str = Field(..., description="Estado de cuenta corriente")
    duration: int = Field(..., description="Duración del crédito en meses")
    credit_history: str = Field(..., description="Historial crediticio")
    purpose: str = Field(..., description="Propósito del crédito")
    credit_amount: float = Field(..., description="Monto del crédito")
    savings_status: str = Field(..., description="Estado de ahorros")
    employment: str = Field(..., description="Situación laboral")
    installment_rate: int = Field(..., description="Tasa de cuota")
    personal_status: str = Field(..., description="Estado civil y género")
    other_parties: str = Field(..., description="Otros deudores")
    residence_since: int = Field(..., description="Residencia desde")
    property_magnitude: str = Field(..., description="Propiedad")
    age: int = Field(..., description="Edad")
    other_payment_plans: str = Field(..., description="Otros planes de pago")
    housing: str = Field(..., description="Vivienda")
    existing_credits: int = Field(..., description="Créditos existentes")
    job: str = Field(..., description="Tipo de trabajo")
    num_dependents: int = Field(..., description="Número de dependientes")
    own_telephone: str = Field(..., description="Teléfono propio")
    foreign_worker: str = Field(..., description="Trabajador extranjero")
    
    class Config:
        schema_extra = {
            "example": {
                "checking_status": "A11",
                "duration": 6,
                "credit_history": "A34",
                "purpose": "A43",
                "credit_amount": 1169,
                "savings_status": "A65",
                "employment": "A75",
                "installment_rate": 4,
                "personal_status": "A93",
                "other_parties": "A101",
                "residence_since": 4,
                "property_magnitude": "A121",
                "age": 67,
                "other_payment_plans": "A143",
                "housing": "A152",
                "existing_credits": 2,
                "job": "A173",
                "num_dependents": 1,
                "own_telephone": "A192",
                "foreign_worker": "A201"
            }
        }

class BinaryPredictionResponse(BaseModel):
    """Respuesta de predicción binaria"""
    prediction: str = Field(..., description="Good o Bad")
    probability_bad: float = Field(..., description="Probabilidad de mal crédito")
    probability_good: float = Field(..., description="Probabilidad de buen crédito")
    confidence: float = Field(..., description="Confianza de la predicción")

class RiskLevelResponse(BaseModel):
    """Respuesta de nivel de riesgo"""
    risk_level: str = Field(..., description="Bajo, Medio, Alto, Crítico")
    probabilities: dict = Field(..., description="Probabilidades por nivel")
    recommendation: str = Field(..., description="Recomendación")

# === CARGAR MODELOS ===
# TODO: Descomentar y ajustar cuando los modelos estén entrenados

# binary_model = None
# multiclass_model = None
# scaler = None
# encoders = None

# try:
#     binary_model = keras.models.load_model(config.BINARY_MODEL_PATH)
#     multiclass_model = keras.models.load_model(config.MULTICLASS_MODEL_PATH)
#     scaler = joblib.load(config.SCALER_PATH)
#     encoders = joblib.load(config.LABEL_ENCODER_PATH)
#     print("✓ Modelos cargados correctamente")
# except Exception as e:
#     print(f"⚠️ Error cargando modelos: {e}")
#     print("Los endpoints de predicción no funcionarán hasta entrenar los modelos")

# === FUNCIONES AUXILIARES ===

def preprocess_input(data: CreditApplication):
    """
    Preprocesa los datos de entrada para el modelo
    
    TODO: Implementar preprocesamiento real usando scaler y encoders
    """
    # Convertir a dict
    input_dict = data.dict()
    
    # TODO: Aplicar encoding a categóricas
    # TODO: Aplicar scaling a numéricas
    # TODO: Ordenar columnas según entrenamiento
    
    # Por ahora, retornar ejemplo dummy
    processed = np.random.rand(1, 20)  # Ajustar dimensión
    return processed

# === ENDPOINTS ===

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Riesgo Crediticio",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Documentación interactiva",
            "/predict/binary": "Predicción binaria (Good/Bad)",
            "/predict/risk_level": "Predicción de nivel de riesgo",
            "/health": "Estado del servidor"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado del servidor y modelos"""
    # TODO: Verificar si los modelos están cargados
    models_loaded = False  # Cambiar cuando se carguen los modelos
    
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "binary_model": models_loaded,
        "multiclass_model": models_loaded,
        "message": "API funcionando correctamente" if models_loaded else "Entrenar modelos primero"
    }

@app.post("/predict/binary", response_model=BinaryPredictionResponse)
async def predict_binary(application: CreditApplication):
    """
    Predice si el crédito será bueno o malo
    
    Returns:
        Predicción binaria con probabilidades
    """
    # TODO: Descomentar cuando los modelos estén listos
    # if binary_model is None:
    #     raise HTTPException(status_code=503, detail="Modelo no disponible. Entrenar primero.")
    
    try:
        # Preprocesar
        # X = preprocess_input(application)
        
        # Predecir
        # proba = binary_model.predict(X)[0][0]
        
        # Respuesta dummy (eliminar cuando el modelo esté listo)
        proba = np.random.random()
        
        prediction = "Bad" if proba > 0.5 else "Good"
        confidence = max(proba, 1 - proba)
        
        return BinaryPredictionResponse(
            prediction=prediction,
            probability_bad=float(proba),
            probability_good=float(1 - proba),
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/risk_level", response_model=RiskLevelResponse)
async def predict_risk_level(application: CreditApplication):
    """
    Predice el nivel de riesgo crediticio
    
    Returns:
        Nivel de riesgo y probabilidades
    """
    # TODO: Implementar predicción real
    
    try:
        # Respuesta dummy (eliminar cuando el modelo esté listo)
        probas = np.random.dirichlet(np.ones(4))
        risk_idx = np.argmax(probas)
        risk_level = config.RISK_LEVELS[risk_idx]
        
        # Generar recomendación
        recommendations = {
            "Bajo": "Crédito aprobado con condiciones estándar",
            "Medio": "Crédito aprobado con seguimiento",
            "Alto": "Requiere garantías adicionales",
            "Crítico": "No recomendado aprobar"
        }
        
        return RiskLevelResponse(
            risk_level=risk_level,
            probabilities={
                level: float(prob) 
                for level, prob in zip(config.RISK_LEVELS, probas)
            },
            recommendation=recommendations[risk_level]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# === EJECUTAR ===
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor FastAPI...")
    print("Documentación: http://localhost:8000/docs")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
