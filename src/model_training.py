# -*- coding: utf-8 -*-
"""
model_training.py
Entrenamiento del modelo de Triage con XGBoost y LightGBM
Optimizado para maximizar Recall en clases críticas (Nivel 1 y 2)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import joblib
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, f1_score, precision_score, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, RANDOM_STATE,
    TRIAGE_LEVELS, CRITICAL_CLASSES, TARGET_RECALL_CRITICAL,
    MODELS_DIR, OUTPUTS_DIR, DATA_DIR, ALL_FEATURES
)
from src.preprocessing import (
    TriagePreprocessor, prepare_data_splits, calculate_class_weights
)


class TriageModelTrainer:
    """
    Entrenador de modelos de Triage optimizado para alto Recall
    en clases críticas (Nivel 1: Resucitación, Nivel 2: Emergencia).
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        random_state: int = RANDOM_STATE
    ):
        """
        Args:
            model_type: 'xgboost' o 'lightgbm'
            random_state: Semilla aleatoria
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_threshold = None
        self.training_history = {}
        self.feature_importance = None
        
    def _create_model(self, class_weights: Optional[Dict] = None) -> Any:
        """Crea el modelo según el tipo especificado."""
        if self.model_type == 'xgboost':
            params = XGBOOST_PARAMS.copy()
            if class_weights:
                # Convertir weights a sample_weight en el entrenamiento
                pass
            return xgb.XGBClassifier(**params)
        
        elif self.model_type == 'lightgbm':
            params = LIGHTGBM_PARAMS.copy()
            if class_weights:
                params['class_weight'] = class_weights
            return lgb.LGBMClassifier(**params)
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def _compute_sample_weights(
        self, 
        y: np.ndarray, 
        class_weights: Dict[int, float]
    ) -> np.ndarray:
        """Convierte pesos de clase a pesos de muestra."""
        return np.array([class_weights[label] for label in y])
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entrena el modelo con énfasis en clases críticas.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Features de validación
            y_val: Labels de validación
            class_weights: Pesos de clase opcionales
            feature_names: Nombres de las features
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("\n" + "=" * 60)
        print(f"🚀 ENTRENAMIENTO DEL MODELO ({self.model_type.upper()})")
        print("=" * 60)
        
        # Calcular pesos si no se proporcionan
        if class_weights is None:
            class_weights = calculate_class_weights(y_train)
        
        print(f"\n📊 Datos de entrenamiento: {X_train.shape}")
        print(f"📊 Datos de validación: {X_val.shape}")
        print(f"\n⚖️ Pesos de clase:")
        for cls, weight in class_weights.items():
            level_name = TRIAGE_LEVELS.get(cls + 1, f"Clase {cls}")
            print(f"   {level_name}: {weight:.3f}")
        
        # Crear modelo
        self.model = self._create_model(class_weights)
        
        # Calcular sample weights
        sample_weights = self._compute_sample_weights(y_train, class_weights)
        
        # Entrenar
        print("\n🔄 Entrenando modelo...")
        start_time = datetime.now()
        
        if self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:  # lightgbm
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"   ⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
        
        # Evaluar en validación
        print("\n📈 Evaluando en conjunto de validación...")
        val_results = self._evaluate(X_val, y_val, "Validación")
        
        # Guardar importancia de features
        if feature_names is not None:
            self._compute_feature_importance(feature_names)
        
        # Guardar historial
        self.training_history = {
            'model_type': self.model_type,
            'training_time': training_time,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'class_weights': {str(k): v for k, v in class_weights.items()},
            'val_results': val_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return val_results
    
    def _evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        set_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo con énfasis en métricas críticas.
        """
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Métricas generales
        accuracy = accuracy_score(y, y_pred)
        
        # Recall por clase (MÉTRICA CRÍTICA)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        
        # Recall para clases críticas (0 y 1, que corresponden a Nivel 1 y 2)
        recall_critical = []
        for cls in [0, 1]:  # Clases críticas (0-indexed)
            if cls < len(recall_per_class):
                recall_critical.append(recall_per_class[cls])
        
        recall_critical_avg = np.mean(recall_critical) if recall_critical else 0
        
        # F1-Score ponderado
        f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        # Precisión por clase
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y, y_pred)
        
        # Imprimir resultados
        print(f"\n{'─' * 50}")
        print(f"📊 RESULTADOS EN {set_name.upper()}")
        print(f"{'─' * 50}")
        
        print(f"\n🎯 MÉTRICAS CRÍTICAS (Objetivo: Recall > {TARGET_RECALL_CRITICAL*100}%)")
        for cls in [0, 1]:
            if cls < len(recall_per_class):
                level_name = TRIAGE_LEVELS.get(cls + 1, f"Nivel {cls+1}")
                status = "✅" if recall_per_class[cls] >= TARGET_RECALL_CRITICAL else "⚠️"
                print(f"   {status} Recall Nivel {cls+1} ({level_name}): {recall_per_class[cls]*100:.1f}%")
        
        print(f"\n   📈 Recall Promedio Crítico: {recall_critical_avg*100:.1f}%")
        
        print(f"\n📊 MÉTRICAS GENERALES")
        print(f"   Accuracy: {accuracy*100:.1f}%")
        print(f"   F1-Score Ponderado: {f1_weighted*100:.1f}%")
        
        print(f"\n📋 RECALL POR CLASE:")
        for cls, recall in enumerate(recall_per_class):
            level_name = TRIAGE_LEVELS.get(cls + 1, f"Nivel {cls+1}")
            print(f"   Nivel {cls+1} ({level_name}): {recall*100:.1f}%")
        
        print(f"\n📋 PRECISION POR CLASE:")
        for cls, prec in enumerate(precision_per_class):
            level_name = TRIAGE_LEVELS.get(cls + 1, f"Nivel {cls+1}")
            print(f"   Nivel {cls+1} ({level_name}): {prec*100:.1f}%")
        
        print(f"\n📋 MATRIZ DE CONFUSIÓN:")
        print(cm)
        
        results = {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'recall_per_class': recall_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'recall_critical_avg': float(recall_critical_avg),
            'confusion_matrix': cm.tolist(),
            'meets_critical_threshold': recall_critical_avg >= TARGET_RECALL_CRITICAL
        }
        
        return results
    
    def _compute_feature_importance(self, feature_names: List[str]):
        """Calcula y almacena la importancia de features."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 TOP 10 FEATURES MÁS IMPORTANTES:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def optimize_threshold_for_critical(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_recall: float = TARGET_RECALL_CRITICAL
    ) -> Dict[int, float]:
        """
        Optimiza umbrales de decisión para maximizar recall en clases críticas.
        
        Permite ajustar los thresholds para que las predicciones de 
        Nivel 1 y 2 tengan mayor sensibilidad.
        """
        print("\n🔧 Optimizando umbrales para clases críticas...")
        
        y_pred_proba = self.model.predict_proba(X_val)
        
        optimized_thresholds = {}
        
        for cls in [0, 1]:  # Clases críticas
            best_threshold = 0.5
            best_recall = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                # Predecir clase si la probabilidad supera el umbral
                y_pred_cls = (y_pred_proba[:, cls] >= threshold).astype(int)
                y_true_cls = (y_val == cls).astype(int)
                
                recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
                
                if recall >= target_recall and recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
            
            optimized_thresholds[cls] = best_threshold
            print(f"   Clase {cls} (Nivel {cls+1}): Umbral={best_threshold:.2f}, Recall={best_recall*100:.1f}%")
        
        self.best_threshold = optimized_thresholds
        return optimized_thresholds
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Realiza validación cruzada estratificada.
        """
        print(f"\n🔄 Validación cruzada ({n_folds} folds)...")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Scores de recall para clases críticas
        recall_scores = []
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Entrenar modelo temporal
            class_weights = calculate_class_weights(y_fold_train)
            sample_weights = self._compute_sample_weights(y_fold_train, class_weights)
            
            temp_model = self._create_model(class_weights)
            temp_model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            # Evaluar
            y_pred = temp_model.predict(X_fold_val)
            
            recall_critical = recall_score(y_fold_val, y_pred, labels=[0, 1], average='macro', zero_division=0)
            f1 = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)
            
            recall_scores.append(recall_critical)
            f1_scores.append(f1)
            
            print(f"   Fold {fold+1}: Recall Crítico={recall_critical*100:.1f}%, F1={f1*100:.1f}%")
        
        results = {
            'recall_critical_mean': np.mean(recall_scores),
            'recall_critical_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }
        
        print(f"\n   📊 Promedio Recall Crítico: {results['recall_critical_mean']*100:.1f}% (±{results['recall_critical_std']*100:.1f}%)")
        print(f"   📊 Promedio F1: {results['f1_mean']*100:.1f}% (±{results['f1_std']*100:.1f}%)")
        
        return results
    
    def save_model(self, filepath: Optional[str] = None):
        """Guarda el modelo entrenado."""
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f"triage_model_{self.model_type}_{timestamp}.joblib"
        else:
            filepath = Path(filepath)
        
        # Guardar modelo
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'best_threshold': self.best_threshold
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Modelo guardado en: {filepath}")
        
        # Guardar también el historial en JSON
        history_path = filepath.with_suffix('.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TriageModelTrainer':
        """Carga un modelo guardado."""
        model_data = joblib.load(filepath)
        
        trainer = cls(model_type=model_data['model_type'])
        trainer.model = model_data['model']
        trainer.feature_importance = model_data['feature_importance']
        trainer.training_history = model_data['training_history']
        trainer.best_threshold = model_data['best_threshold']
        
        return trainer


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str]
) -> Dict[str, TriageModelTrainer]:
    """
    Entrena un ensemble de modelos XGBoost y LightGBM.
    """
    print("\n" + "=" * 60)
    print("🎯 ENTRENAMIENTO DE ENSEMBLE (XGBoost + LightGBM)")
    print("=" * 60)
    
    models = {}
    
    for model_type in ['xgboost', 'lightgbm']:
        trainer = TriageModelTrainer(model_type=model_type)
        trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
        trainer.optimize_threshold_for_critical(X_val, y_val)
        models[model_type] = trainer
    
    return models


def main():
    """Función principal de entrenamiento."""
    print("\n" + "=" * 60)
    print("🏥 SISTEMA DE TRIAGE ASISTIDO POR IA")
    print("   Entrenamiento del Modelo")
    print("=" * 60)
    
    # Cargar datos
    train_path = DATA_DIR / "triage_train.csv"
    
    if not train_path.exists():
        print("⚠️ No se encontró el dataset. Generando datos...")
        from src.data_generator import main as generate_data
        generate_data()
    
    print("\n📂 Cargando datos...")
    df = pd.read_csv(train_path)
    
    # Preparar splits
    splits = prepare_data_splits(df)
    
    # Preprocesar datos
    preprocessor = TriagePreprocessor()
    
    X_train, y_train = preprocessor.fit_transform(
        splits['X_train'],
        splits['y_train'],
        apply_smote=True,
        smote_strategy='auto'
    )
    
    X_val = preprocessor.transform(splits['X_val'])
    y_val = splits['y_val'].values
    
    X_test = preprocessor.transform(splits['X_test'])
    y_test = splits['y_test'].values
    
    # Guardar preprocesador
    preprocessor.save()
    
    # Obtener nombres de features
    feature_names = preprocessor.feature_order
    
    # Entrenar modelo principal (XGBoost)
    trainer = TriageModelTrainer(model_type='xgboost')
    trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    # Optimizar umbrales
    trainer.optimize_threshold_for_critical(X_val, y_val)
    
    # Validación cruzada
    trainer.cross_validate(X_train, y_train)
    
    # Evaluar en test
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN FINAL EN TEST SET")
    print("=" * 60)
    trainer._evaluate(X_test, y_test, "Test")
    
    # Guardar modelo
    model_path = trainer.save_model(MODELS_DIR / "triage_model_final.joblib")
    
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   Modelo guardado: {model_path}")
    print(f"   Preprocesador: {MODELS_DIR / 'preprocessor.joblib'}")
    
    return trainer, preprocessor


if __name__ == "__main__":
    trainer, preprocessor = main()
