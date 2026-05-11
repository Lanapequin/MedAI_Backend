"""
train_model.py
Trains the MedAI triage classification model using the real Chocó dataset.
Saves the model package to models/triage_model.joblib for API use.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, recall_score, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.medai_preprocessor import MedAIPreprocessor  # noqa: E402
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
FEATURE_NAMES = [
    'edad', 'frecuencia_cardiaca', 'frecuencia_respiratoria',
    'temperatura', 'spO2', 'presion_sistolica', 'presion_diastolica',
    'dolor', 'sexo_encoded', 'motivo_encoded',
    'diabetes', 'hipertension', 'enfermedad_cardiaca',
    'enfermedad_respiratoria', 'inmunosupresion'
]

MOTIVO_MAP = {
    'Administrativo': 0,
    'COVID/Respiratorio': 1,
    'Cefalea': 2,
    'Dermatológico': 3,
    'Disnea': 4,
    'Dolor abdominal': 5,
    'Dolor lumbar': 6,
    'Dolor torácico': 7,
    'Fiebre': 8,
    'Hemorragia': 9,
    'Herida leve': 10,
    'Intoxicación': 11,
    'Lesión menor': 12,
    'Mareo/Síncope': 13,
    'Obstétrico': 14,
    'Otros': 15,
    'Sospecha dengue': 16,
    'Sospecha malaria': 17,
    'Trauma': 18,
    'Vómito/diarrea': 19,
}

MODO_MAP = {'Caminando': 0, 'Particular': 1, 'Ambulancia': 2, 'Policía': 3, 'Policia': 3}
SEXO_MAP = {'M': 1, 'F': 0}


def parse_comorbidities(comorb_str):
    result = {
        'diabetes': 0, 'hipertension': 0, 'enfermedad_cardiaca': 0,
        'enfermedad_respiratoria': 0, 'inmunosupresion': 0
    }
    if pd.isna(comorb_str) or comorb_str == '':
        return result
    s = str(comorb_str).lower()
    if 'dm2' in s or 'diabetes' in s:
        result['diabetes'] = 1
    if 'hta' in s or 'hipertension' in s or 'hipertensión' in s:
        result['hipertension'] = 1
    if 'cardia' in s or 'cardio' in s or 'coronaria' in s:
        result['enfermedad_cardiaca'] = 1
    if 'respira' in s or 'asma' in s or 'epoc' in s or 'copd' in s:
        result['enfermedad_respiratoria'] = 1
    if 'vih' in s or 'inmunosupresi' in s or 'cancer' in s or 'cáncer' in s:
        result['inmunosupresion'] = 1
    return result


def build_features(df):
    comorb_df = pd.DataFrame(df['comorbilidades'].apply(parse_comorbidities).tolist())

    # Encode motivo: use fixed map, fallback to Otros (15)
    motivo_encoded = df['motivo_consulta'].map(MOTIVO_MAP).fillna(15).astype(int)

    X = pd.DataFrame({
        'edad': df['edad'],
        'frecuencia_cardiaca': df['frecuencia_cardiaca'],
        'frecuencia_respiratoria': df['frecuencia_respiratoria'],
        'temperatura': df['temperatura'],
        'spO2': df['spO2'],
        'presion_sistolica': df['presion_sistolica'],
        'presion_diastolica': df['presion_diastolica'],
        'dolor': df['dolor'],
        'sexo_encoded': df['sexo'].map(SEXO_MAP).fillna(0).astype(int),
        'motivo_encoded': motivo_encoded,
    })
    X = pd.concat([X, comorb_df.reset_index(drop=True)], axis=1)
    X = X[FEATURE_NAMES]
    y = (df['nivel_urgencia'] - 1).values  # 0-indexed classes

    return X, y


def main():
    print("=" * 60)
    print("MedAI — Model Training Pipeline")
    print("=" * 60)

    # Load dataset
    csv_path = PROJECT_ROOT / "triage_choco_100k_balanceado.csv"
    print(f"\nLoading dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"  Samples: {len(df):,}  |  Columns: {len(df.columns)}")

    # Build feature matrix
    X, y = build_features(df)
    print(f"  Features: {X.shape[1]}  |  Classes: {np.unique(y)}")

    # Impute missing values (median for numerics)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_NAMES)

    # Stratified 70/15/15 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_imputed, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85,
        random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"\nSplit → Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # Fit scaler on continuous features only
    continuous_cols = ['edad', 'frecuencia_cardiaca', 'frecuencia_respiratoria',
                       'temperatura', 'spO2', 'presion_sistolica', 'presion_diastolica', 'dolor']
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # SMOTE on training set only to boost minority classes
    print("\nApplying SMOTE to training set...")
    unique, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    # Oversample Level 1 and 2 (critical) to match majority
    smote_strategy = {cls: max_count for cls in unique}
    try:
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE, k_neighbors=5)
        X_train_arr, y_train_bal = smote.fit_resample(X_train.values, y_train)
    except Exception as e:
        print(f"  SMOTE failed ({e}), using class weights only")
        X_train_arr, y_train_bal = X_train.values, y_train

    unique_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
    print(f"  After SMOTE: {dict(zip(unique_bal, counts_bal))}")

    # Class weights (additional boost for critical classes 0, 1)
    n_samples = len(y_train_bal)
    n_classes = 5
    class_weights = {}
    for cls, cnt in zip(*np.unique(y_train_bal, return_counts=True)):
        base_w = n_samples / (n_classes * cnt)
        class_weights[cls] = base_w * (1.5 if cls in [0, 1] else 1.0)

    sample_weights = np.array([class_weights[label] for label in y_train_bal])

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        max_depth=7,
        learning_rate=0.08,
        n_estimators=400,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
    )
    model.fit(
        X_train_arr, y_train_bal,
        sample_weight=sample_weights,
        eval_set=[(X_val.values, y_val)],
        verbose=50,
    )

    # Evaluate
    print("\n--- Test Set Evaluation ---")
    y_pred = model.predict(X_test.values)
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_per = recall_score(y_test, y_pred, average=None, zero_division=0)
    print(f"  Accuracy:       {acc*100:.1f}%")
    print(f"  F1 (weighted):  {f1_w*100:.1f}%")
    for i, r in enumerate(recall_per):
        flag = " ✓" if (i < 2 and r >= 0.95) else (" ⚠" if i < 2 else "")
        print(f"  Recall Level {i+1}: {r*100:.1f}%{flag}")

    preprocessor = MedAIPreprocessor(imputer, scaler, FEATURE_NAMES, continuous_cols)

    # Save model package
    model_package = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': FEATURE_NAMES,
        'class_weights': class_weights,
        'motivo_map': MOTIVO_MAP,
        'modo_map': MODO_MAP,
        'sexo_map': SEXO_MAP,
        'training_date': datetime.now().isoformat(),
        'dataset': 'triage_choco_100k_balanceado.csv',
        'model_type': 'xgboost',
        'n_classes': 5,
        'n_train_samples': len(y_train_bal),
        'n_test_samples': len(y_test),
        'test_accuracy': float(acc),
        'test_f1_weighted': float(f1_w),
        'test_recall_per_class': recall_per.tolist(),
    }

    out_path = MODELS_DIR / "triage_model.joblib"
    joblib.dump(model_package, out_path)
    print(f"\nModel saved → {out_path}")
    print("=" * 60)
    print("Training complete.")


if __name__ == '__main__':
    main()
