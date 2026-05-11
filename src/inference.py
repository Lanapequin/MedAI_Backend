# -*- coding: utf-8 -*-
"""
inference.py
Pipeline de inferencia para el Sistema de Triage
Diseñado para integración en aplicaciones de tele-triage o e-salud
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRIAGE_LEVELS, VITAL_SIGNS, NEUROLOGICAL_FEATURES,
    DEMOGRAPHIC_FEATURES, COMORBIDITY_FEATURES, SYMPTOM_FEATURES,
    ALL_FEATURES, MODELS_DIR, TARGET_RECALL_CRITICAL,
    NORMAL_VITAL_RANGES, SYMPTOM_CATEGORIES
)


class TriageLevel(Enum):
    """Niveles de Triage según el Sistema Colombiano/ESI"""
    RESUCITACION = 1
    EMERGENCIA = 2
    URGENTE = 3
    MENOS_URGENTE = 4
    NO_URGENTE = 5


@dataclass
class PatientData:
    """Estructura de datos del paciente para triage."""
    # Signos vitales
    frecuencia_cardiaca: float
    frecuencia_respiratoria: float
    presion_sistolica: float
    presion_diastolica: float
    temperatura: float
    saturacion_oxigeno: float
    
    # Neurológico
    escala_glasgow: int
    
    # Demográfico
    edad: int
    sexo: int  # 0: Femenino, 1: Masculino
    
    # Síntoma principal (código)
    sintoma_principal: int
    
    # Comorbilidades (opcionales)
    diabetes: int = 0
    hipertension: int = 0
    enfermedad_cardiaca: int = 0
    enfermedad_respiratoria: int = 0
    inmunosupresion: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'frecuencia_cardiaca': self.frecuencia_cardiaca,
            'frecuencia_respiratoria': self.frecuencia_respiratoria,
            'presion_sistolica': self.presion_sistolica,
            'presion_diastolica': self.presion_diastolica,
            'temperatura': self.temperatura,
            'saturacion_oxigeno': self.saturacion_oxigeno,
            'escala_glasgow': self.escala_glasgow,
            'edad': self.edad,
            'sexo': self.sexo,
            'sintoma_principal': self.sintoma_principal,
            'diabetes': self.diabetes,
            'hipertension': self.hipertension,
            'enfermedad_cardiaca': self.enfermedad_cardiaca,
            'enfermedad_respiratoria': self.enfermedad_respiratoria,
            'inmunosupresion': self.inmunosupresion
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte a DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class TriageResult:
    """Resultado de la clasificación de triage."""
    nivel: int
    nombre_nivel: str
    descripcion: str
    probabilidades: Dict[str, float]
    confianza: float
    factores_riesgo: List[Dict]
    factores_protectores: List[Dict]
    alertas: List[str]
    recomendaciones: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'nivel': self.nivel,
            'nombre_nivel': self.nombre_nivel,
            'descripcion': self.descripcion,
            'probabilidades': self.probabilidades,
            'confianza': self.confianza,
            'factores_riesgo': self.factores_riesgo,
            'factores_protectores': self.factores_protectores,
            'alertas': self.alertas,
            'recomendaciones': self.recomendaciones,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Serializa a JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class TriageInferenceEngine:
    """
    Motor de inferencia para clasificación de triage.
    Diseñado para ser integrado en aplicaciones de tele-triage.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None
    ):
        """
        Args:
            model_path: Ruta al modelo serializado
            preprocessor_path: Ruta al preprocesador serializado
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
        if model_path and preprocessor_path:
            self.load(model_path, preprocessor_path)
    
    def load(
        self, 
        model_path: Optional[str] = None, 
        preprocessor_path: Optional[str] = None
    ):
        """
        Carga el modelo y preprocesador.
        """
        if model_path is None:
            model_path = MODELS_DIR / "triage_model_final.joblib"
        if preprocessor_path is None:
            preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        
        model_path = Path(model_path)
        preprocessor_path = Path(preprocessor_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocesador no encontrado: {preprocessor_path}")
        
        # Cargar modelo
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        
        # Cargar preprocesador
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = self.preprocessor.feature_order
        
        self.is_loaded = True
        print(f"✅ Motor de inferencia cargado")
        print(f"   Modelo: {model_path.name}")
        print(f"   Features: {len(self.feature_names)}")
    
    def _validate_input(self, patient: PatientData) -> List[str]:
        """
        Valida los datos de entrada y genera alertas.
        """
        alertas = []
        data = patient.to_dict()
        
        # Validar rangos de signos vitales
        vital_validations = {
            'frecuencia_cardiaca': (20, 250, 'FC'),
            'frecuencia_respiratoria': (4, 60, 'FR'),
            'presion_sistolica': (40, 280, 'PAS'),
            'presion_diastolica': (20, 180, 'PAD'),
            'temperatura': (32, 42, 'Temp'),
            'saturacion_oxigeno': (50, 100, 'SpO2')
        }
        
        for vital, (min_val, max_val, name) in vital_validations.items():
            val = data.get(vital)
            if val is not None:
                if val < min_val or val > max_val:
                    alertas.append(f"⚠️ {name} fuera de rango fisiológico: {val}")
        
        # Validar Glasgow
        glasgow = data.get('escala_glasgow')
        if glasgow is not None and (glasgow < 3 or glasgow > 15):
            alertas.append(f"⚠️ Glasgow inválido: {glasgow} (debe ser 3-15)")
        
        # Alertas por valores críticos
        if data.get('saturacion_oxigeno', 100) < 90:
            alertas.append("🚨 ALERTA: Hipoxemia severa (SpO2 < 90%)")
        
        if data.get('presion_sistolica', 120) < 90:
            alertas.append("🚨 ALERTA: Hipotensión (PAS < 90 mmHg)")
        
        if data.get('escala_glasgow', 15) <= 8:
            alertas.append("🚨 ALERTA: Alteración severa de conciencia (Glasgow ≤ 8)")
        
        if data.get('frecuencia_cardiaca', 80) > 150:
            alertas.append("🚨 ALERTA: Taquicardia severa (FC > 150)")
        
        if data.get('temperatura', 37) > 39.5:
            alertas.append("🚨 ALERTA: Fiebre alta (Temp > 39.5°C)")
        
        return alertas
    
    def _generate_recommendations(self, nivel: int, alertas: List[str]) -> List[str]:
        """
        Genera recomendaciones basadas en el nivel de triage.
        """
        recomendaciones = {
            1: [
                "🔴 ACTIVAR PROTOCOLO DE RESUCITACIÓN",
                "Preparar equipos de soporte vital avanzado",
                "Notificar equipo de emergencias",
                "Preparar traslado inmediato a centro de mayor complejidad",
                "Monitoreo continuo de signos vitales",
                "Establecer acceso venoso"
            ],
            2: [
                "🟠 ATENCIÓN MÉDICA INMEDIATA",
                "Evaluación por médico en menos de 10 minutos",
                "Preparar área de observación",
                "Considerar referencia según evolución",
                "Monitoreo frecuente de signos vitales (cada 15 min)"
            ],
            3: [
                "🟡 ATENCIÓN PRIORITARIA",
                "Evaluación médica en menos de 30 minutos",
                "Iniciar manejo sintomático según protocolo",
                "Reevaluación si hay deterioro",
                "Monitoreo de signos vitales (cada 30 min)"
            ],
            4: [
                "🟢 ATENCIÓN PROGRAMADA",
                "Evaluación según disponibilidad",
                "Puede esperar en sala de espera",
                "Reevaluación si hay cambio en síntomas"
            ],
            5: [
                "🔵 ATENCIÓN NO URGENTE",
                "Derivación a consulta externa si aplica",
                "Educación sobre signos de alarma",
                "Seguimiento ambulatorio"
            ]
        }
        
        base_recs = recomendaciones.get(nivel, [])
        
        # Añadir recomendaciones específicas según alertas
        if any("Hipoxemia" in a for a in alertas):
            base_recs.insert(1, "Administrar oxígeno suplementario")
        
        if any("Hipotensión" in a for a in alertas):
            base_recs.insert(1, "Iniciar reanimación con fluidos")
        
        return base_recs
    
    def predict(
        self, 
        patient: Union[PatientData, Dict, pd.DataFrame],
        explain: bool = True
    ) -> TriageResult:
        """
        Realiza predicción de triage para un paciente.
        
        Args:
            patient: Datos del paciente (PatientData, dict, o DataFrame)
            explain: Si incluir explicación de factores
        
        Returns:
            TriageResult con la clasificación y recomendaciones
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llama load() primero.")
        
        # Convertir a PatientData si es necesario
        if isinstance(patient, dict):
            patient = PatientData(**patient)
        elif isinstance(patient, pd.DataFrame):
            patient = PatientData(**patient.iloc[0].to_dict())
        
        # Validar entrada
        alertas = self._validate_input(patient)
        
        # Convertir a DataFrame
        df = patient.to_dataframe()
        
        # Preprocesar
        X = self.preprocessor.transform(df)
        
        # Predecir
        pred_proba = self.model.predict_proba(X)[0]
        pred_class = np.argmax(pred_proba)
        
        # Nivel de triage (1-indexed)
        nivel = pred_class + 1
        nombre_nivel = TRIAGE_LEVELS.get(nivel, f"Nivel {nivel}")
        
        # Descripción del nivel
        descripciones = {
            1: "Requiere atención inmediata. Riesgo vital inminente.",
            2: "Emergencia grave. Atención prioritaria requerida.",
            3: "Urgencia moderada. Atención relativamente pronta.",
            4: "Urgencia menor. Puede esperar atención estándar.",
            5: "No urgente. Atención programada o ambulatoria."
        }
        
        # Probabilidades formateadas
        probabilidades = {
            TRIAGE_LEVELS.get(i+1, f"Nivel {i+1}"): float(p)
            for i, p in enumerate(pred_proba)
        }
        
        # Factores de riesgo y protectores (simplificado sin SHAP para velocidad)
        factores_riesgo = []
        factores_protectores = []
        
        if explain and self.feature_importance is not None:
            # Usar feature importance del modelo
            patient_dict = patient.to_dict()
            top_features = self.feature_importance.head(10)['feature'].tolist()
            
            for feat in top_features[:5]:
                val = patient_dict.get(feat)
                if val is not None:
                    normal_range = NORMAL_VITAL_RANGES.get(feat)
                    if normal_range:
                        if val < normal_range[0] or val > normal_range[1]:
                            factores_riesgo.append({
                                'feature': feat,
                                'value': val,
                                'descripcion': f"{feat} fuera de rango normal"
                            })
                        else:
                            factores_protectores.append({
                                'feature': feat,
                                'value': val,
                                'descripcion': f"{feat} en rango normal"
                            })
        
        # Generar recomendaciones
        recomendaciones = self._generate_recommendations(nivel, alertas)
        
        # Crear resultado
        result = TriageResult(
            nivel=nivel,
            nombre_nivel=nombre_nivel,
            descripcion=descripciones.get(nivel, ""),
            probabilidades=probabilidades,
            confianza=float(pred_proba[pred_class]),
            factores_riesgo=factores_riesgo,
            factores_protectores=factores_protectores,
            alertas=alertas,
            recomendaciones=recomendaciones,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def predict_batch(
        self, 
        patients: List[Union[PatientData, Dict]],
        explain: bool = False
    ) -> List[TriageResult]:
        """
        Realiza predicciones para múltiples pacientes.
        """
        return [self.predict(p, explain=explain) for p in patients]
    
    def get_prediction_summary(self, result: TriageResult) -> str:
        """
        Genera un resumen textual de la predicción.
        """
        lines = [
            "=" * 50,
            "RESULTADO DE CLASIFICACIÓN DE TRIAGE",
            "=" * 50,
            f"\n📋 NIVEL: {result.nivel} - {result.nombre_nivel}",
            f"📝 {result.descripcion}",
            f"🎯 Confianza: {result.confianza*100:.1f}%",
        ]
        
        if result.alertas:
            lines.append("\n⚠️ ALERTAS:")
            for alerta in result.alertas:
                lines.append(f"   {alerta}")
        
        lines.append("\n📊 PROBABILIDADES:")
        for nivel, prob in result.probabilidades.items():
            bar = "█" * int(prob * 20)
            lines.append(f"   {nivel[:20]:<20} {bar} {prob*100:.1f}%")
        
        lines.append("\n✅ RECOMENDACIONES:")
        for rec in result.recomendaciones[:5]:
            lines.append(f"   • {rec}")
        
        lines.append("\n" + "=" * 50)
        lines.append("⚠️ Esta es una recomendación asistida por IA.")
        lines.append("La decisión final debe ser tomada por personal médico.")
        lines.append("=" * 50)
        
        return "\n".join(lines)


def create_sample_patient() -> PatientData:
    """Crea un paciente de ejemplo para pruebas."""
    return PatientData(
        frecuencia_cardiaca=95,
        frecuencia_respiratoria=22,
        presion_sistolica=130,
        presion_diastolica=85,
        temperatura=37.8,
        saturacion_oxigeno=96,
        escala_glasgow=15,
        edad=45,
        sexo=1,
        sintoma_principal=5,  # Fiebre
        diabetes=0,
        hipertension=1,
        enfermedad_cardiaca=0,
        enfermedad_respiratoria=0,
        inmunosupresion=0
    )


def create_critical_patient() -> PatientData:
    """Crea un paciente crítico de ejemplo."""
    return PatientData(
        frecuencia_cardiaca=140,
        frecuencia_respiratoria=32,
        presion_sistolica=75,
        presion_diastolica=45,
        temperatura=39.8,
        saturacion_oxigeno=82,
        escala_glasgow=7,
        edad=68,
        sexo=0,
        sintoma_principal=1,  # Dificultad respiratoria
        diabetes=1,
        hipertension=1,
        enfermedad_cardiaca=1,
        enfermedad_respiratoria=1,
        inmunosupresion=0
    )


# API simplificada para integración
def quick_triage(
    frecuencia_cardiaca: float,
    frecuencia_respiratoria: float,
    presion_sistolica: float,
    presion_diastolica: float,
    temperatura: float,
    saturacion_oxigeno: float,
    escala_glasgow: int,
    edad: int,
    sexo: int,
    sintoma_principal: int,
    **comorbilidades
) -> Dict[str, Any]:
    """
    API simplificada para clasificación rápida de triage.
    
    Ejemplo:
        result = quick_triage(
            frecuencia_cardiaca=95,
            frecuencia_respiratoria=22,
            presion_sistolica=130,
            presion_diastolica=85,
            temperatura=37.8,
            saturacion_oxigeno=96,
            escala_glasgow=15,
            edad=45,
            sexo=1,
            sintoma_principal=5
        )
    """
    engine = TriageInferenceEngine()
    engine.load()
    
    patient = PatientData(
        frecuencia_cardiaca=frecuencia_cardiaca,
        frecuencia_respiratoria=frecuencia_respiratoria,
        presion_sistolica=presion_sistolica,
        presion_diastolica=presion_diastolica,
        temperatura=temperatura,
        saturacion_oxigeno=saturacion_oxigeno,
        escala_glasgow=escala_glasgow,
        edad=edad,
        sexo=sexo,
        sintoma_principal=sintoma_principal,
        **comorbilidades
    )
    
    result = engine.predict(patient)
    return result.to_dict()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏥 SISTEMA DE TRIAGE - DEMO DE INFERENCIA")
    print("=" * 60)
    
    try:
        # Crear motor de inferencia
        engine = TriageInferenceEngine()
        engine.load()
        
        # Caso 1: Paciente moderado
        print("\n" + "─" * 60)
        print("📋 CASO 1: Paciente con síntomas moderados")
        print("─" * 60)
        
        patient1 = create_sample_patient()
        result1 = engine.predict(patient1)
        print(engine.get_prediction_summary(result1))
        
        # Caso 2: Paciente crítico
        print("\n" + "─" * 60)
        print("📋 CASO 2: Paciente en estado crítico")
        print("─" * 60)
        
        patient2 = create_critical_patient()
        result2 = engine.predict(patient2)
        print(engine.get_prediction_summary(result2))
        
        # Mostrar JSON
        print("\n" + "─" * 60)
        print("📄 SALIDA JSON (para integración API):")
        print("─" * 60)
        print(result2.to_json())
        
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        print("Ejecuta primero el pipeline de entrenamiento (main.py)")
