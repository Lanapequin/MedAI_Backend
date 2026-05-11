# -*- coding: utf-8 -*-
"""
main_real_data_fast.py
Entrenamiento rápido con DATASET REAL (sin SMOTE para velocidad)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Añadir path del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, TRIAGE_LEVELS
from src.preprocessing import TriagePreprocessor, calculate_class_weights
from src.model_training import TriageModelTrainer
from src.evaluation import TriageEvaluator
from src.interpretability import TriageExplainer
from sklearn.model_selection import train_test_split


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"🏥 {title}")
    print("=" * 70)


def preprocess_real_data(df: pd.DataFrame) -> tuple:
    """Preprocesa el dataset real."""
    print("🔄 Preprocesando dataset real...")
    
    df = df.copy()
    y = df['nivel_urgencia'].values - 1  # Convertir a 0-4
    
    # Features a usar
    feature_cols = [
        'edad', 'frecuencia_cardiaca', 'frecuencia_respiratoria',
        'temperatura', 'spO2', 'presion_sistolica', 'presion_diastolica', 'dolor'
    ]
    
    # Comorbilidades
    def parse_comorbidities(comorb_str):
        result = {
            'diabetes': 0, 'hipertension': 0, 'enfermedad_cardiaca': 0,
            'enfermedad_respiratoria': 0, 'inmunosupresion': 0
        }
        if pd.isna(comorb_str) or comorb_str == '':
            return result
        comorb_str = str(comorb_str).lower()
        if 'dm2' in comorb_str or 'diabetes' in comorb_str:
            result['diabetes'] = 1
        if 'hta' in comorb_str or 'hipertension' in comorb_str:
            result['hipertension'] = 1
        if 'cardia' in comorb_str or 'cardio' in comorb_str:
            result['enfermedad_cardiaca'] = 1
        if 'respira' in comorb_str or 'asma' in comorb_str or 'copd' in comorb_str:
            result['enfermedad_respiratoria'] = 1
        if 'vih' in comorb_str or 'inmunosupresi' in comorb_str or 'cancer' in comorb_str:
            result['inmunosupresion'] = 1
        return result
    
    comorb_features = df['comorbilidades'].apply(parse_comorbidities)
    comorb_df = pd.DataFrame(comorb_features.tolist())
    
    # Codificar sexo y motivo
    sexo_map = {'M': 1, 'F': 0}
    df['sexo_encoded'] = df['sexo'].map(sexo_map).fillna(0).astype(int)
    motivo_map = {m: i for i, m in enumerate(df['motivo_consulta'].unique())}
    df['motivo_encoded'] = df['motivo_consulta'].map(motivo_map).fillna(0).astype(int)
    
    # Crear X
    X = df[feature_cols + ['sexo_encoded', 'motivo_encoded']].copy()
    X = pd.concat([X, comorb_df], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    
    feature_names = X.columns.tolist()
    
    print(f"   ✅ Features: {len(feature_names)}, Muestras: {len(y)}")
    for level, count in enumerate(np.bincount(y)):
        print(f"      Nivel {level + 1}: {count} muestras")
    
    return X, y, feature_names


print_header("ENTRENAMIENTO CON DATASET REAL")
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================================================================
# PASO 1: CARGAR DATASET REAL
# =========================================================================
print_header("PASO 1: CARGAR DATASET REAL")

df = pd.read_csv(PROJECT_ROOT / "triage_choco_100k_balanceado.csv")
print(f"✅ Cargados: {df.shape[0]:,} muestras")

X, y, feature_names = preprocess_real_data(df)

# =========================================================================
# PASO 2: DIVIDIR DATOS
# =========================================================================
print_header("PASO 2: DIVIDIR DATOS")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)

print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

# =========================================================================
# PASO 3: PREPROCESAR (sin SMOTE para rapidez)
# =========================================================================
print_header("PASO 3: PREPROCESAR DATOS")

preprocessor = TriagePreprocessor(random_state=RANDOM_STATE)

X_train_proc, y_train_proc = preprocessor.fit_transform(
    X_train.reset_index(drop=True),
    pd.Series(y_train),
    apply_smote=False  # Sin SMOTE para rapidez
)

X_val_proc = preprocessor.transform(X_val.reset_index(drop=True))
X_test_proc = preprocessor.transform(X_test.reset_index(drop=True))

print(f"✅ Preprocesamiento completado")

# =========================================================================
# PASO 4: ENTRENAR MODELO
# =========================================================================
print_header("PASO 4: ENTRENAR MODELO")

trainer = TriageModelTrainer(model_type='xgboost', random_state=RANDOM_STATE)

class_weights = calculate_class_weights(y_train_proc)

training_results = trainer.train(
    X_train_proc, y_train_proc,
    X_val_proc, np.array(y_val),
    class_weights=class_weights,
    feature_names=feature_names
)

# =========================================================================
# PASO 5: EVALUAR
# =========================================================================
print_header("PASO 5: EVALUAR MODELO")

evaluator = TriageEvaluator(trainer.model)
test_results = evaluator.evaluate(X_test_proc, np.array(y_test), set_name="Test")

# =========================================================================
# PASO 6: GUARDAR MODELO
# =========================================================================
print_header("PASO 6: GUARDAR MODELO")

model_package = {
    'model': trainer.model,
    'preprocessor': preprocessor,
    'feature_names': feature_names,
    'class_weights': class_weights,
    'training_date': datetime.now().isoformat(),
    'dataset': 'triage_choco_100k_balanceado.csv (subconjunto)',
    'model_type': 'xgboost',
    'n_classes': 5,
    'n_train_samples': len(y_train_proc),
    'n_test_samples': len(y_test)
}

model_path = MODELS_DIR / "triage_model_real_data.joblib"
joblib.dump(model_package, model_path)

print(f"💾 Modelo guardado: {model_path}")

# =========================================================================
# RESUMEN
# =========================================================================
print_header("RESUMEN FINAL")
print(f"✅ Modelo entrenado con dataset REAL")
print(f"   Dataset: triage_choco_100k_balanceado.csv")
print(f"   Muestras: {len(y_train_proc):,} (train) + {len(y_test):,} (test)")
print(f"   Accuracy en test: {test_results['basic_metrics']['accuracy']*100:.1f}%")
recall_critical = test_results['critical_analysis'].get('recall_critical_avg', 0)
print(f"   Recall Crítico (1-2): {recall_critical*100:.1f}%")
print("=" * 70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 70)
