# src/__init__.py
"""
Módulos del Sistema de Triage Asistido por IA
"""

# from .data_generator import TriageDataGenerator  # OBSOLETO: Solo se usa dataset real
from .preprocessing import TriagePreprocessor, prepare_data_splits
from .model_training import TriageModelTrainer
from .evaluation import TriageEvaluator
from .interpretability import TriageExplainer
from .inference import TriageInferenceEngine, PatientData, TriageResult

__all__ = [
    # 'TriageDataGenerator',  # OBSOLETO
    'TriagePreprocessor',
    'prepare_data_splits',
    'TriageModelTrainer',
    'TriageEvaluator',
    'TriageExplainer',
    'TriageInferenceEngine',
    'PatientData',
    'TriageResult'
]
