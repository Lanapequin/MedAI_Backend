# -*- coding: utf-8 -*-
"""
data_generator.py
Generador de datos sintéticos realistas para el Sistema de Triage
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRIAGE_LEVELS, VITAL_SIGNS, NORMAL_VITAL_RANGES, CRITICAL_VITAL_RANGES,
    SYMPTOM_CATEGORIES, HIGH_SEVERITY_SYMPTOMS, ALL_FEATURES,
    COMORBIDITY_FEATURES, DATA_DIR, RANDOM_STATE
)


class TriageDataGenerator:
    """
    Generador de datos sintéticos de triage clínico.
    Simula datos realistas basados en conocimiento médico.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _generate_vital_signs(self, triage_level: int, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Genera signos vitales basados en el nivel de triage.
        Niveles 1-2 tienen valores más extremos.
        """
        vitals = {}
        
        for vital, normal_range in NORMAL_VITAL_RANGES.items():
            if triage_level in [1, 2]:
                # Generar valores críticos
                critical_ranges = CRITICAL_VITAL_RANGES[vital]
                values = []
                for _ in range(n_samples):
                    # 70% probabilidad de valores críticos para niveles 1-2
                    if np.random.random() < 0.7 and critical_ranges[0] is not None:
                        # Seleccionar rango bajo o alto
                        if critical_ranges[1] is not None and np.random.random() < 0.5:
                            val = np.random.uniform(*critical_ranges[1])
                        else:
                            val = np.random.uniform(*critical_ranges[0])
                    else:
                        # Valores borderline
                        if np.random.random() < 0.5:
                            val = np.random.uniform(normal_range[0] - 10, normal_range[0])
                        else:
                            val = np.random.uniform(normal_range[1], normal_range[1] + 15)
                    values.append(val)
                vitals[vital] = np.array(values)
                
            elif triage_level == 3:
                # Valores moderadamente alterados
                mean = (normal_range[0] + normal_range[1]) / 2
                std = (normal_range[1] - normal_range[0]) / 2
                values = np.random.normal(mean, std * 1.5, n_samples)
                vitals[vital] = values
                
            else:  # Niveles 4-5
                # Valores normales o ligeramente alterados
                values = np.random.uniform(
                    normal_range[0] - 5,
                    normal_range[1] + 5,
                    n_samples
                )
                vitals[vital] = values
        
        # Ajustar rangos físicamente posibles
        vitals["frecuencia_cardiaca"] = np.clip(vitals["frecuencia_cardiaca"], 20, 250)
        vitals["frecuencia_respiratoria"] = np.clip(vitals["frecuencia_respiratoria"], 4, 60)
        vitals["presion_sistolica"] = np.clip(vitals["presion_sistolica"], 40, 280)
        vitals["presion_diastolica"] = np.clip(vitals["presion_diastolica"], 20, 180)
        vitals["temperatura"] = np.clip(vitals["temperatura"], 32, 42)
        vitals["saturacion_oxigeno"] = np.clip(vitals["saturacion_oxigeno"], 50, 100)
        
        return vitals
    
    def _generate_glasgow(self, triage_level: int, n_samples: int) -> np.ndarray:
        """
        Genera puntaje de Escala de Coma de Glasgow (3-15).
        """
        if triage_level == 1:
            # Severo: 3-8
            return np.random.randint(3, 9, n_samples)
        elif triage_level == 2:
            # Moderado-Severo: 6-12
            return np.random.randint(6, 13, n_samples)
        elif triage_level == 3:
            # Leve-Moderado: 10-15
            return np.random.randint(10, 16, n_samples)
        else:
            # Normal: 14-15
            return np.random.choice([14, 15], n_samples, p=[0.2, 0.8])
    
    def _generate_demographics(self, triage_level: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera edad y sexo.
        Niveles críticos más frecuentes en extremos de edad.
        """
        if triage_level in [1, 2]:
            # Distribución bimodal: más jóvenes (trauma) y mayores (cardio)
            n_young = n_samples // 3
            n_old = n_samples - n_young
            ages = np.concatenate([
                np.random.normal(25, 10, n_young),
                np.random.normal(70, 12, n_old)
            ])
            np.random.shuffle(ages)
        else:
            # Distribución más uniforme
            ages = np.random.normal(45, 20, n_samples)
        
        ages = np.clip(ages, 0, 105).astype(int)
        sexo = np.random.binomial(1, 0.5, n_samples)  # 0: F, 1: M
        
        return ages, sexo
    
    def _generate_symptoms(self, triage_level: int, n_samples: int) -> np.ndarray:
        """
        Genera síntoma principal codificado.
        """
        n_categories = len(SYMPTOM_CATEGORIES)
        
        if triage_level in [1, 2]:
            # Alta probabilidad de síntomas graves
            probs = np.ones(n_categories) * 0.02
            for idx in HIGH_SEVERITY_SYMPTOMS:
                probs[idx] = 0.15
            probs = probs / probs.sum()
        elif triage_level == 3:
            # Probabilidad mixta
            probs = np.ones(n_categories) * 0.05
            for idx in HIGH_SEVERITY_SYMPTOMS:
                probs[idx] = 0.08
            probs = probs / probs.sum()
        else:
            # Síntomas menos graves más frecuentes
            probs = np.ones(n_categories) * 0.1
            for idx in HIGH_SEVERITY_SYMPTOMS:
                probs[idx] = 0.02
            probs = probs / probs.sum()
        
        return np.random.choice(range(n_categories), n_samples, p=probs)
    
    def _generate_comorbidities(self, ages: np.ndarray, triage_level: int) -> Dict[str, np.ndarray]:
        """
        Genera comorbilidades basadas en edad y nivel de triage.
        """
        n_samples = len(ages)
        comorbidities = {}
        
        # Probabilidad base aumenta con la edad
        age_factor = np.clip(ages / 100, 0.1, 0.8)
        
        # Probabilidades base por comorbilidad
        base_probs = {
            "diabetes": 0.15,
            "hipertension": 0.25,
            "enfermedad_cardiaca": 0.12,
            "enfermedad_respiratoria": 0.10,
            "inmunosupresion": 0.05
        }
        
        for comorb, base_prob in base_probs.items():
            # Ajustar por nivel de triage (más comorbilidades en niveles críticos)
            triage_factor = 1.5 if triage_level in [1, 2] else 1.0
            probs = np.minimum(base_prob * age_factor * triage_factor, 0.9)
            comorbidities[comorb] = np.random.binomial(1, probs)
        
        return comorbidities
    
    def _introduce_missing_values(self, df: pd.DataFrame, missing_rate: float = 0.05) -> pd.DataFrame:
        """
        Introduce valores faltantes de manera realista.
        Los valores críticos tienen menor tasa de missing.
        """
        df_missing = df.copy()
        
        # No introducir missing en la variable objetivo
        features_with_missing = [col for col in df.columns if col != 'nivel_triage']
        
        for col in features_with_missing:
            # Menor missing rate para signos vitales críticos
            if col in ['saturacion_oxigeno', 'presion_sistolica', 'frecuencia_cardiaca']:
                col_missing_rate = missing_rate * 0.5
            else:
                col_missing_rate = missing_rate
            
            mask = np.random.random(len(df)) < col_missing_rate
            df_missing.loc[mask, col] = np.nan
        
        return df_missing
    
    def generate_dataset(
        self,
        n_samples: int = 10000,
        class_distribution: Optional[Dict[int, float]] = None,
        include_missing: bool = True,
        missing_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Genera un dataset completo de triage.
        
        Args:
            n_samples: Número total de muestras
            class_distribution: Distribución de clases {nivel: proporción}
            include_missing: Si incluir valores faltantes
            missing_rate: Tasa de valores faltantes
        
        Returns:
            DataFrame con todos los datos generados
        """
        # Distribución realista de triage (desbalanceada)
        if class_distribution is None:
            class_distribution = {
                1: 0.03,   # 3% - Resucitación (muy raro)
                2: 0.08,   # 8% - Emergencia
                3: 0.22,   # 22% - Urgente
                4: 0.35,   # 35% - Menos urgente
                5: 0.32    # 32% - No urgente
            }
        
        all_data = []
        
        for level, proportion in class_distribution.items():
            n_level = int(n_samples * proportion)
            
            # Generar componentes
            vitals = self._generate_vital_signs(level, n_level)
            glasgow = self._generate_glasgow(level, n_level)
            ages, sexo = self._generate_demographics(level, n_level)
            symptoms = self._generate_symptoms(level, n_level)
            comorbidities = self._generate_comorbidities(ages, level)
            
            # Construir DataFrame para este nivel
            level_data = pd.DataFrame({
                **vitals,
                "escala_glasgow": glasgow,
                "edad": ages,
                "sexo": sexo,
                "sintoma_principal": symptoms,
                **comorbidities,
                "nivel_triage": level
            })
            
            all_data.append(level_data)
        
        # Combinar todos los niveles
        df = pd.concat(all_data, ignore_index=True)
        
        # Mezclar datos
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Introducir valores faltantes
        if include_missing:
            df = self._introduce_missing_values(df, missing_rate)
        
        # Redondear valores para realismo
        df["frecuencia_cardiaca"] = df["frecuencia_cardiaca"].round(0)
        df["frecuencia_respiratoria"] = df["frecuencia_respiratoria"].round(0)
        df["presion_sistolica"] = df["presion_sistolica"].round(0)
        df["presion_diastolica"] = df["presion_diastolica"].round(0)
        df["temperatura"] = df["temperatura"].round(1)
        df["saturacion_oxigeno"] = df["saturacion_oxigeno"].round(0)
        
        # Ajustar el nivel de triage para que sea 0-indexed (para sklearn)
        df["nivel_triage"] = df["nivel_triage"] - 1
        
        return df


def main():
    """Genera y guarda el dataset de entrenamiento."""
    print("=" * 60)
    print("GENERADOR DE DATOS DE TRIAGE - ZONAS DE ALTO RIESGO COLOMBIA")
    print("=" * 60)
    
    generator = TriageDataGenerator(random_state=RANDOM_STATE)
    
    # Generar dataset principal
    print("\n📊 Generando dataset de entrenamiento (10,000 muestras)...")
    df_train = generator.generate_dataset(
        n_samples=10000,
        include_missing=True,
        missing_rate=0.05
    )
    
    # Generar dataset de prueba
    print("📊 Generando dataset de prueba (2,000 muestras)...")
    df_test = generator.generate_dataset(
        n_samples=2000,
        include_missing=True,
        missing_rate=0.03
    )
    
    # Guardar datasets
    train_path = DATA_DIR / "triage_train.csv"
    test_path = DATA_DIR / "triage_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"\n✅ Dataset de entrenamiento guardado: {train_path}")
    print(f"✅ Dataset de prueba guardado: {test_path}")
    
    # Mostrar estadísticas
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL DATASET DE ENTRENAMIENTO")
    print("=" * 60)
    
    print("\n📈 Distribución de clases:")
    class_counts = df_train["nivel_triage"].value_counts().sort_index()
    for level, count in class_counts.items():
        level_name = TRIAGE_LEVELS[level + 1]
        percentage = count / len(df_train) * 100
        print(f"   Nivel {level + 1} ({level_name}): {count:,} ({percentage:.1f}%)")
    
    print(f"\n📊 Dimensiones: {df_train.shape[0]:,} filas × {df_train.shape[1]} columnas")
    print(f"📋 Features: {len(ALL_FEATURES)}")
    
    print("\n🔍 Valores faltantes:")
    missing = df_train.isnull().sum()
    missing_pct = (missing / len(df_train) * 100).round(2)
    for col in df_train.columns:
        if missing[col] > 0:
            print(f"   {col}: {missing[col]} ({missing_pct[col]}%)")
    
    print("\n📊 Estadísticas descriptivas de signos vitales:")
    print(df_train[VITAL_SIGNS].describe().round(2).to_string())
    
    return df_train, df_test


if __name__ == "__main__":
    main()
