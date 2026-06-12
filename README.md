# Proyecto B: Sistema de Predicción de Abandono de Clientes

[![Carga de datos](https://img.shields.io/badge/Cargar%20datos-CSV-4CAF50?style=for-the-badge&logo=files)]()
[![Limpieza de datos](https://img.shields.io/badge/Limpieza%20y%20EDA-Procesamiento-2196F3?style=for-the-badge&logo=databricks)]()   
[![Modelado supervisado](https://img.shields.io/badge/Modelado-Supervisado-FF9800?style=for-the-badge&logo=mlflow)]()  
[![Documentación](https://img.shields.io/badge/Documentación-Notebook-795548?style=for-the-badge&logo=jupyter)]()  
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-E91E63?style=for-the-badge&logo=streamlit)]()  
[![API](https://img.shields.io/badge/🔌_API-Disponible-blue?style=for-the-badge)]()

## 👥 Equipo
- **Integrante 1**: Sharon Obando Gómez 
- **Integrante 2**: Marco Alvarez Quiros
- **Integrante 3**: Fabián Brenes Loría

## 📋 Descripción del Proyecto
Creación de un sistema que prediga qué clientes tienen mayor probabilidad de
abandonar un servicio de telecomunicaciones e identificar clientes en
riesgo clasificandolos por nivel de urgencia para implementar estrategias de retención.

## 🎯 Objetivos
- Realizar análisis exploratorio de datos (EDA) con visualizaciones profesionales
- Implementar preprocesamiento de datos para modelos de machine learning
- Diseñar, entrenar y evaluar redes neuronales artificiales (ANN)
- Comparar múltiples modelos y seleccionar el óptimo según métricas
- Desplegar modelos en producción mediante API REST con FastAPI
- Crear interfaces de usuario con Streamlit
- Documentar proyectos de forma profesional

## 📊 Dataset
- **Fuente**: Telco Customer Churn KAGGLE
- **URL**: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Registros**: 7,043 clientes
- **Variables**: 21 (demográficas, servicios contratados, información de cuenta)
- **Variables principales**: antigüedad del cliente, tipo de contrato, método de pago, cargos
mensuales, servicios adicionales (internet, streaming, seguridad),

## 🔧 Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Pasos de Instalación
```bash
# 1. Clonar o descargar el repositorio
cd ProyectoA_RiesgoCrediticio

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset
python data/raw/download_data.py
```

## 🚀 Uso

### Ejecutar Notebooks
```bash
jupyter notebook notebooks/
```
Ejecutar en orden:
1. `01_EDA_CreditRisk.ipynb` - Análisis exploratorio
2. `02_Preprocesamiento.ipynb` - Limpieza y preparación
3. `03_ANN_BinaryClass.ipynb` - Modelo binario
4. `04_ANN_Regression.ipynb` - Modelo regresión
5. `05_Comparacion_Modelos.ipynb` - Evaluación

### Entrenar Modelos
```bash
# Entrenar modelo de clasificación binaria
python src/train/binary_classifier.py

# Entrenar modelo de clasificación multiclase
python src/train/multiclass_classifier.py
```

### Ejecutar API
```bash
cd api
uvicorn main:app --reload
```
La API estará disponible en: http://localhost:8000
Documentación automática: http://localhost:8000/docs

### Ejecutar Frontend
```bash
cd app
streamlit run Home.py
```
El frontend estará disponible en: http://localhost:8501

## 📁 Estructura del Proyecto
```
ProyectoA_RiesgoCrediticio/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                        # Archivos a ignorar en Git
│
├── data/
│   ├── raw/                          # Datos originales
│   │   └── download_data.py          # Script de descarga
│   └── processed/                    # Datos procesados
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA_CreditRisk.ipynb      # Análisis exploratorio
│   ├── 02_Preprocesamiento.ipynb     # Preprocesamiento
│   ├── 03_ANN_BinaryClass.ipynb      # Modelo binario
│   ├── 04_ANN_MultiClass.ipynb       # Modelo multiclase
│   └── 05_Comparacion_Modelos.ipynb  # Comparación
│
├── src/                              # Código fuente
│   ├── __init__.py
│   ├── data_prep.py                  # Funciones de preprocesamiento
│   ├── config.py                     # Configuraciones
│   └── train/                        # Scripts de entrenamiento
│       ├── binary_classifier.py
│       ├── Regression_classifier.py
│       └── utils.py
│
├── models/                           # Modelos entrenados
│   ├── binary_model.h5
│   ├── regression_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── api/                              # API REST
│   ├── main.py                       # Aplicación FastAPI
│   ├── schemas.py                    # Modelos Pydantic
│   └── predict.py                    # Lógica de predicción
│
└── app/                              # Frontend Streamlit
    ├── Home.py                       # Página principal
    └── pages/
        ├── 1_Prediccion_Individual.py
        ├── 2_Analisis_Batch.py
        └── 3_Metricas_Modelos.py
```

## 🧪 Modelos Implementados

### Modelo 1: Clasificación Binaria
- **Objetivo**: Predecir aprobación de crédito (Bueno/Malo)
- **Arquitectura**: [Describir arquitectura implementada]
- **Métricas**: 
  - Accuracy: [completar]
  - Precision: [completar]
  - Recall: [completar]
  - F1-Score: [completar]

### Modelo 2: Clasificación Regresión
- **Objetivo**: Clasificar nivel de riesgo (Bajo/Medio/Alto/Crítico)
- **Arquitectura**: [Describir arquitectura implementada]
- **Métricas**: 
  - Accuracy: [completar]
  - Precision macro: [completar]
  - Recall macro: [completar]
  - F1-Score macro: [completar]

## 📈 Resultados

### Comparación de Modelos
[Insertar tabla o gráfico comparativo]

### Conclusiones
[Describir conclusiones principales del proyecto]

### Recomendaciones
[Sugerencias para mejora o implementación en producción]

## 🛠️ Tecnologías Utilizadas
- **Python 3.x**
- **TensorFlow/Keras**: Redes neuronales
- **Pandas/NumPy**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Preprocesamiento y métricas
- **FastAPI**: API REST
- **Streamlit**: Frontend web
- **Uvicorn**: Servidor ASGI

## 📝 Notas de Desarrollo
[Espacio para documentar decisiones técnicas, problemas encontrados y soluciones]

## 📧 Contacto
Para consultas sobre este proyecto, contactar a: [email del grupo]

---
**Proyecto desarrollado para el curso de Inteligencia Artificial Aplicada**  
**Colegio Universitario de Cartago (CUC) - 2025**
