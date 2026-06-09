"""
Frontend con Streamlit para el Sistema de Predicción de Riesgo Crediticio

Ejecutar: streamlit run Home.py
URL: http://localhost:8501
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="Sistema de Riesgo Crediticio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ESTILOS PERSONALIZADOS ===
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1F4E78;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E75B5;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1F4E78;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("🏦 Sistema de IA")
st.sidebar.markdown("### Predicción de Riesgo Crediticio")
st.sidebar.markdown("---")

# Información del proyecto
st.sidebar.markdown("### 📊 Información del Proyecto")
st.sidebar.info("""
**Equipo:**
- Integrante 1
- Integrante 2
- Integrante 3

**Curso:** IA Aplicada - CUC  
**Año:** 2025
""")

st.sidebar.markdown("---")

# Enlaces útiles
st.sidebar.markdown("### 🔗 Enlaces")
st.sidebar.markdown("[📖 Documentación API](http://localhost:8000/docs)")
st.sidebar.markdown("[📁 GitHub del Proyecto](#)")
st.sidebar.markdown("[📊 Dataset UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)")

# === PÁGINA PRINCIPAL ===

# Header
st.markdown('<h1 class="main-header">💳 Sistema de Predicción de Riesgo Crediticio</h1>', 
            unsafe_allow_html=True)

st.markdown("""
Este sistema utiliza **Redes Neuronales Artificiales (ANN)** para predecir el riesgo crediticio 
de clientes bancarios, apoyando decisiones de aprobación de préstamos de manera inteligente y automatizada.
""")

# === TABS PRINCIPALES ===
tab1, tab2, tab3 = st.tabs(["📋 Descripción", "🎯 Modelos", "📈 Resultados"])

with tab1:
    st.markdown('<div class="sub-header">📋 Descripción del Proyecto</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objetivos")
        st.markdown("""
        - Predecir si un crédito será bueno o malo
        - Clasificar clientes por nivel de riesgo
        - Automatizar proceso de evaluación crediticia
        - Reducir tasas de default
        """)
        
        st.markdown("### 📊 Dataset")
        st.markdown("""
        **German Credit Data (UCI)**
        - 1,000 clientes
        - 20 variables predictoras
        - Variables demográficas y financieras
        """)
    
    with col2:
        st.markdown("### 🔧 Tecnologías")
        st.markdown("""
        - **TensorFlow/Keras**: Redes neuronales
        - **FastAPI**: API REST
        - **Streamlit**: Frontend interactivo
        - **Scikit-learn**: Preprocesamiento
        """)
        
        st.markdown("### 📁 Navegación")
        st.info("""
        👈 Usa el menú lateral para:
        - 📝 Realizar predicciones individuales
        - 📊 Analizar lotes de solicitudes
        - 📈 Ver métricas de los modelos
        """)

with tab2:
    st.markdown('<div class="sub-header">🎯 Modelos Implementados</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Modelo 1: Clasificación Binaria")
        st.markdown("""
        **Objetivo:** Predecir aprobación de crédito
        
        **Clases:**
        - ✅ Good Credit (Aprobar)
        - ❌ Bad Credit (Rechazar)
        
        **Arquitectura:**
        - Input Layer: [N features]
        - Hidden Layers: [64, 32, 16]
        - Output Layer: 1 neurona (sigmoid)
        
        **Métricas:**
        - Accuracy: [completar después del entrenamiento]
        - Precision: [completar]
        - Recall: [completar]
        - F1-Score: [completar]
        """)
    
    with col2:
        st.markdown("### 🟢 Modelo 2: Clasificación Multiclase")
        st.markdown("""
        **Objetivo:** Clasificar nivel de riesgo
        
        **Clases:**
        - 🟢 Riesgo Bajo
        - 🟡 Riesgo Medio
        - 🟠 Riesgo Alto
        - 🔴 Riesgo Crítico
        
        **Arquitectura:**
        - Input Layer: [N features]
        - Hidden Layers: [128, 64, 32]
        - Output Layer: 4 neuronas (softmax)
        
        **Métricas:**
        - Accuracy: [completar]
        - Precision (macro): [completar]
        - Recall (macro): [completar]
        - F1-Score (macro): [completar]
        """)

with tab3:
    st.markdown('<div class="sub-header">📈 Resultados y Conclusiones</div>', 
                unsafe_allow_html=True)
    
    st.warning("⚠️ Esta sección se completará después del entrenamiento de los modelos")
    
    # Placeholder para resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Accuracy (Binario)", value="---%", delta="---")
    
    with col2:
        st.metric(label="Accuracy (Multiclase)", value="---%", delta="---")
    
    with col3:
        st.metric(label="ROC-AUC", value="---", delta="---")
    
    st.markdown("---")
    
    st.markdown("### 🎓 Conclusiones")
    st.markdown("""
    **Hallazgos principales:**
    - [A completar después del análisis]
    - [A completar después del análisis]
    - [A completar después del análisis]
    
    **Recomendaciones:**
    - [A completar después del análisis]
    - [A completar después del análisis]
    """)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema desarrollado para el curso de Inteligencia Artificial Aplicada</p>
    <p>Colegio Universitario de Cartago (CUC) - 2025</p>
</div>
""", unsafe_allow_html=True)

# === INFORMACIÓN DE DEBUG (solo en desarrollo) ===
with st.expander("🔧 Información de Debug"):
    st.markdown("### Configuración del Sistema")
    st.json({
        "PROJECT_ROOT": str(config.PROJECT_ROOT),
        "MODELS_DIR": str(config.MODELS_DIR),
        "BINARY_MODEL": str(config.BINARY_MODEL_PATH),
        "MULTICLASS_MODEL": str(config.MULTICLASS_MODEL_PATH),
        "API_URL": f"http://localhost:{config.API_PORT}"
    })
