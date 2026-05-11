# -*- coding: utf-8 -*-
"""
preprocessing.py
Pipeline de preprocesamiento para el Sistema de Triage
Incluye manejo de valores faltantes, encoding, normalización y SMOTE
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE, SMOTE_PARAMS,
    MODELS_DIR, DATA_DIR
)


class TriagePreprocessor:
    """
    Pipeline de preprocesamiento completo para datos de triage.
    Maneja valores faltantes, normalización, encoding y desbalance de clases.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        
        # Imputadores
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Estado de ajuste
        self.is_fitted = False
        
        # Almacenar estadísticas
        self.feature_stats = {}
        
    def _separate_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separa features numéricas y categóricas."""
        numerical_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
        
        return X[numerical_cols], X[categorical_cols]
    
    def _impute_missing(
        self, 
        X_num: pd.DataFrame, 
        X_cat: pd.DataFrame, 
        fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Imputa valores faltantes."""
        if fit:
            X_num_imputed = pd.DataFrame(
                self.numerical_imputer.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
            X_cat_imputed = pd.DataFrame(
                self.categorical_imputer.fit_transform(X_cat),
                columns=X_cat.columns,
                index=X_cat.index
            )
        else:
            X_num_imputed = pd.DataFrame(
                self.numerical_imputer.transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
            X_cat_imputed = pd.DataFrame(
                self.categorical_imputer.transform(X_cat),
                columns=X_cat.columns,
                index=X_cat.index
            )
        
        return X_num_imputed, X_cat_imputed
    
    def _scale_features(
        self, 
        X_num: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """Normaliza features numéricas."""
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=X_num.columns,
                index=X_num.index
            )
        
        return X_scaled
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        apply_smote: bool = True,
        smote_strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta el preprocesador y transforma los datos.
        
        Args:
            X: Features
            y: Target
            apply_smote: Si aplicar SMOTE para balancear clases
            smote_strategy: Estrategia de SMOTE ('auto', 'minority', dict)
        
        Returns:
            X_transformed, y_transformed
        """
        print("🔧 Iniciando preprocesamiento...")
        
        # Separar features
        X_num, X_cat = self._separate_features(X)
        
        # Guardar estadísticas originales
        self.feature_stats['missing_before'] = X.isnull().sum().to_dict()
        self.feature_stats['original_shape'] = X.shape
        
        # Imputar valores faltantes
        print("   ├── Imputando valores faltantes...")
        X_num_imputed, X_cat_imputed = self._impute_missing(X_num, X_cat, fit=True)
        
        # Normalizar features numéricas
        print("   ├── Normalizando features numéricas...")
        X_num_scaled = self._scale_features(X_num_imputed, fit=True)
        
        # Combinar features
        X_processed = pd.concat([X_num_scaled, X_cat_imputed], axis=1)
        
        # Asegurar el orden correcto de columnas
        self.feature_order = X_processed.columns.tolist()
        
        # Convertir a arrays
        X_array = X_processed.values
        y_array = y.values
        
        # Aplicar SMOTE si se requiere
        if apply_smote:
            print("   ├── Aplicando SMOTE para balancear clases...")
            X_array, y_array = self._apply_smote(X_array, y_array, smote_strategy)
        
        self.is_fitted = True
        self.feature_stats['processed_shape'] = X_array.shape
        
        print(f"   └── ✅ Preprocesamiento completado")
        print(f"       Shape original: {self.feature_stats['original_shape']}")
        print(f"       Shape procesado: {self.feature_stats['processed_shape']}")
        
        return X_array, y_array
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma nuevos datos usando el preprocesador ajustado.
        
        Args:
            X: Features a transformar
        
        Returns:
            X_transformed
        """
        if not self.is_fitted:
            raise ValueError("El preprocesador no ha sido ajustado. Llama fit_transform primero.")
        
        # Separar features
        X_num, X_cat = self._separate_features(X)
        
        # Imputar
        X_num_imputed, X_cat_imputed = self._impute_missing(X_num, X_cat, fit=False)
        
        # Normalizar
        X_num_scaled = self._scale_features(X_num_imputed, fit=False)
        
        # Combinar
        X_processed = pd.concat([X_num_scaled, X_cat_imputed], axis=1)
        
        # Asegurar orden de columnas
        X_processed = X_processed[self.feature_order]
        
        return X_processed.values
    
    def _apply_smote(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        strategy: str = 'auto'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica SMOTE para sobremuestrear clases minoritarias (críticas).
        
        Estrategia personalizada para maximizar las clases 0 y 1 (Nivel 1 y 2).
        """
        # Contar clases originales
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        print(f"       Distribución original: {class_counts}")
        
        # Crear estrategia de sampling personalizada
        # Objetivo: aumentar clases críticas (0, 1) al nivel de la clase más frecuente
        max_count = max(counts)
        
        if strategy == 'custom_critical':
            # Sobremuestrear solo clases críticas
            sampling_dict = {}
            for cls in unique:
                if cls in [0, 1]:  # Clases críticas
                    sampling_dict[cls] = int(max_count * 0.8)
                else:
                    sampling_dict[cls] = class_counts[cls]
            smote = SMOTE(
                sampling_strategy=sampling_dict,
                random_state=self.random_state,
                k_neighbors=min(5, min(counts) - 1)
            )
        else:
            # SMOTE estándar
            smote = SMOTE(
                sampling_strategy=strategy,
                random_state=self.random_state,
                k_neighbors=min(5, min(counts) - 1)
            )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Mostrar nueva distribución
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"       Distribución después de SMOTE: {dict(zip(unique_new, counts_new))}")
            
        except Exception as e:
            print(f"       ⚠️ Error en SMOTE: {e}. Usando datos originales.")
            X_resampled, y_resampled = X, y
        
        return X_resampled, y_resampled
    
    def save(self, filepath: Optional[str] = None):
        """Guarda el preprocesador ajustado."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        
        joblib.dump(self, filepath)
        print(f"💾 Preprocesador guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'TriagePreprocessor':
        """Carga un preprocesador guardado."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.joblib"
        
        return joblib.load(filepath)


def prepare_data_splits(
    df: pd.DataFrame,
    target_col: str = 'nivel_triage',
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Prepara los splits de datos: train, validation, test.
    
    Args:
        df: DataFrame completo
        target_col: Nombre de la columna objetivo
        test_size: Proporción del test set
        val_size: Proporción del validation set (del train)
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario con X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Separar features y target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Split train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Split train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"📊 Splits de datos creados:")
    print(f"   Train: {X_train.shape[0]:,} muestras")
    print(f"   Validation: {X_val.shape[0]:,} muestras")
    print(f"   Test: {X_test.shape[0]:,} muestras")
    
    return {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calcula pesos de clase inversamente proporcionales a su frecuencia.
    Da mayor peso a las clases críticas (0, 1).
    """
    unique, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique)
    
    # Peso base inversamente proporcional
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = n_samples / (n_classes * count)
    
    # Aumentar peso adicional para clases críticas
    critical_boost = 1.5
    if 0 in weights:
        weights[0] *= critical_boost
    if 1 in weights:
        weights[1] *= critical_boost
    
    return weights


if __name__ == "__main__":
    # Test del preprocesador
    print("=" * 60)
    print("TEST DEL PREPROCESADOR")
    print("=" * 60)
    
    # Cargar datos si existen
    train_path = DATA_DIR / "triage_train.csv"
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        
        # Preparar splits
        splits = prepare_data_splits(df)
        
        # Crear preprocesador
        preprocessor = TriagePreprocessor()
        
        # Ajustar y transformar
        X_train_processed, y_train_processed = preprocessor.fit_transform(
            splits['X_train'],
            splits['y_train'],
            apply_smote=True,
            smote_strategy='auto'
        )
        
        # Transformar validation
        X_val_processed = preprocessor.transform(splits['X_val'])
        
        print(f"\n✅ Preprocesamiento exitoso")
        print(f"   X_train shape: {X_train_processed.shape}")
        print(f"   X_val shape: {X_val_processed.shape}")
        
        # Calcular pesos de clase
        weights = calculate_class_weights(y_train_processed)
        print(f"\n📊 Pesos de clase calculados:")
        for cls, weight in weights.items():
            print(f"   Clase {cls}: {weight:.3f}")
        
        # Guardar preprocesador
        preprocessor.save()
        
    else:
        print("⚠️ No se encontró el dataset. Ejecuta data_generator.py primero.")
