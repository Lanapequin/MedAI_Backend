# -*- coding: utf-8 -*-
"""
config.py
Configuración global del proyecto de Triage Asistido por IA
"""

import os
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURACIÓN DEL MODELO
# =============================================================================

# Clases de Triage (Sistema Colombiano / ESI Modificado)
TRIAGE_LEVELS = {
    1: "Resucitación/Inmediato",
    2: "Emergencia/Grave", 
    3: "Urgente",
    4: "Menos Urgente",
    5: "No Urgente"
}

# Clases críticas que requieren alto Recall
CRITICAL_CLASSES = [1, 2]

# Objetivo de Recall para clases críticas
TARGET_RECALL_CRITICAL = 0.95

# =============================================================================
# FEATURES DEL MODELO
# =============================================================================

# Signos Vitales
VITAL_SIGNS = [
    "frecuencia_cardiaca",      # latidos por minuto (bpm)
    "frecuencia_respiratoria",  # respiraciones por minuto
    "presion_sistolica",        # mmHg
    "presion_diastolica",       # mmHg
    "temperatura",              # °C
    "saturacion_oxigeno"        # % SpO2
]

# Features Neurológicas
NEUROLOGICAL_FEATURES = [
    "escala_glasgow"            # GCS: 3-15
]

# Información Demográfica
DEMOGRAPHIC_FEATURES = [
    "edad",                     # años
    "sexo"                      # 0: Femenino, 1: Masculino
]

# Antecedentes/Comorbilidades (binarias)
COMORBIDITY_FEATURES = [
    "diabetes",
    "hipertension",
    "enfermedad_cardiaca",
    "enfermedad_respiratoria",
    "inmunosupresion"
]

# Síntoma Principal (codificado)
SYMPTOM_FEATURES = [
    "sintoma_principal"         # Código categorizado
]

# Lista completa de features
ALL_FEATURES = (
    VITAL_SIGNS + 
    NEUROLOGICAL_FEATURES + 
    DEMOGRAPHIC_FEATURES + 
    COMORBIDITY_FEATURES + 
    SYMPTOM_FEATURES
)

# Features numéricas vs categóricas
NUMERICAL_FEATURES = VITAL_SIGNS + NEUROLOGICAL_FEATURES + ["edad"]
CATEGORICAL_FEATURES = ["sexo", "sintoma_principal"] + COMORBIDITY_FEATURES

# =============================================================================
# RANGOS NORMALES DE SIGNOS VITALES (para generación de datos)
# =============================================================================

NORMAL_VITAL_RANGES = {
    "frecuencia_cardiaca": (60, 100),
    "frecuencia_respiratoria": (12, 20),
    "presion_sistolica": (90, 140),
    "presion_diastolica": (60, 90),
    "temperatura": (36.0, 37.5),
    "saturacion_oxigeno": (95, 100)
}

# Rangos críticos (Nivel 1-2)
CRITICAL_VITAL_RANGES = {
    "frecuencia_cardiaca": [(0, 40), (150, 220)],
    "frecuencia_respiratoria": [(0, 8), (30, 60)],
    "presion_sistolica": [(0, 70), (200, 280)],
    "presion_diastolica": [(0, 40), (120, 160)],
    "temperatura": [(32, 35), (39.5, 42)],
    "saturacion_oxigeno": [(50, 88), None]  # Solo bajo es crítico
}

# =============================================================================
# CATEGORÍAS DE SÍNTOMAS PRINCIPALES
# =============================================================================

SYMPTOM_CATEGORIES = {
    0: "dolor_toracico",
    1: "dificultad_respiratoria",
    2: "alteracion_conciencia",
    3: "trauma",
    4: "dolor_abdominal",
    5: "fiebre",
    6: "cefalea",
    7: "sintomas_neurologicos",
    8: "hemorragia",
    9: "dolor_general",
    10: "sintomas_gastrointestinales",
    11: "sintomas_urinarios",
    12: "otros"
}

# Síntomas de alta gravedad (asociados a Nivel 1-2)
HIGH_SEVERITY_SYMPTOMS = [0, 1, 2, 3, 8]  # dolor torácico, respiratorio, alt. conciencia, trauma, hemorragia

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Parámetros de XGBoost optimizados para desbalance
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": ["mlogloss", "merror"],
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False
}

# Parámetros de LightGBM
LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": ["multi_logloss", "multi_error"],
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1
}

# =============================================================================
# CONFIGURACIÓN DE SMOTE
# =============================================================================

SMOTE_PARAMS = {
    "random_state": RANDOM_STATE,
    "k_neighbors": 5
}

# Estrategia de sampling personalizada (sobremuestrear clases críticas)
SAMPLING_STRATEGY = {
    0: "auto",  # Nivel 1 - sobremuestrear
    1: "auto",  # Nivel 2 - sobremuestrear
    2: "not minority",
    3: "not minority",
    4: "not minority"
}
