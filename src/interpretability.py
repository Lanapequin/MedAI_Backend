# -*- coding: utf-8 -*-
"""
interpretability.py
Módulo de interpretabilidad usando SHAP para explicar predicciones
Crítico para la validación clínica del modelo de Triage
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRIAGE_LEVELS, VITAL_SIGNS, NEUROLOGICAL_FEATURES,
    DEMOGRAPHIC_FEATURES, COMORBIDITY_FEATURES, SYMPTOM_FEATURES,
    OUTPUTS_DIR, ALL_FEATURES
)


class TriageExplainer:
    """
    Clase para explicar predicciones del modelo de Triage usando SHAP.
    Proporciona explicaciones a nivel global y local (por paciente).
    """
    
    def __init__(
        self, 
        model,
        feature_names: List[str],
        model_type: str = 'tree'
    ):
        """
        Args:
            model: Modelo entrenado (XGBoost, LightGBM)
            feature_names: Nombres de las features
            model_type: Tipo de modelo ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.base_values = None
        
        # Categorización de features para visualización
        self.feature_categories = self._categorize_features()
        
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categoriza features para mejor visualización."""
        categories = {
            'Signos Vitales': [],
            'Neurológico': [],
            'Demográfico': [],
            'Comorbilidades': [],
            'Síntomas': [],
            'Otros': []
        }
        
        for feat in self.feature_names:
            if feat in VITAL_SIGNS:
                categories['Signos Vitales'].append(feat)
            elif feat in NEUROLOGICAL_FEATURES:
                categories['Neurológico'].append(feat)
            elif feat in DEMOGRAPHIC_FEATURES:
                categories['Demográfico'].append(feat)
            elif feat in COMORBIDITY_FEATURES:
                categories['Comorbilidades'].append(feat)
            elif feat in SYMPTOM_FEATURES:
                categories['Síntomas'].append(feat)
            else:
                categories['Otros'].append(feat)
        
        return {k: v for k, v in categories.items() if v}
    
    def fit(self, X_background: np.ndarray, max_samples: int = 500):
        """
        Inicializa el explainer SHAP con datos de background.
        
        Args:
            X_background: Datos para el background (subset de entrenamiento)
            max_samples: Máximo de muestras para el background
        """
        print("🔧 Inicializando SHAP Explainer...")
        
        # Limitar tamaño del background para eficiencia
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                X_background
            )
        
        print(f"   ✅ Explainer inicializado con {len(X_background)} muestras de background")
    
    def compute_shap_values(
        self, 
        X: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Calcula valores SHAP para un conjunto de datos.
        
        Args:
            X: Features para explicar
            check_additivity: Verificar propiedad aditiva de SHAP
        
        Returns:
            Array de valores SHAP
        """
        print(f"📊 Calculando valores SHAP para {len(X)} muestras...")
        
        if self.explainer is None:
            raise ValueError("Explainer no inicializado. Llama fit() primero.")
        
        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        # Para TreeExplainer multiclase, shap_values es una lista
        if isinstance(self.shap_values, list):
            print(f"   ✅ SHAP values calculados para {len(self.shap_values)} clases")
        else:
            print(f"   ✅ SHAP values shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def get_global_importance(self) -> pd.DataFrame:
        """
        Calcula importancia global de features basada en SHAP.
        
        Returns:
            DataFrame con importancia de cada feature
        """
        if self.shap_values is None:
            raise ValueError("SHAP values no calculados. Llama compute_shap_values() primero.")
        
        # Para shap_values 3D (samples, features, classes), promediar en samples y classes
        if len(self.shap_values.shape) == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_abs_mean = np.abs(self.shap_values).mean(axis=(0, 2))
        elif isinstance(self.shap_values, list):
            # Si es una lista de arrays por clase
            shap_abs_mean = np.mean([
                np.abs(sv).mean(axis=0) for sv in self.shap_values
            ], axis=0)
        else:
            shap_abs_mean = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_abs_mean
        }).sort_values('importance', ascending=False)
        
        # Añadir categoría
        def get_category(feat):
            for cat, feats in self.feature_categories.items():
                if feat in feats:
                    return cat
            return 'Otros'
        
        importance_df['category'] = importance_df['feature'].apply(get_category)
        
        return importance_df
    
    def get_class_importance(self, class_idx: int) -> pd.DataFrame:
        """
        Obtiene importancia de features para una clase específica.
        
        Args:
            class_idx: Índice de la clase (0-4)
        
        Returns:
            DataFrame con importancia para esa clase
        """
        if self.shap_values is None:
            raise ValueError("SHAP values no calculados.")
        
        # Para shap_values 3D (samples, features, classes)
        if len(self.shap_values.shape) == 3:
            class_shap = self.shap_values[:, :, class_idx]
        elif isinstance(self.shap_values, list):
            class_shap = self.shap_values[class_idx]
        else:
            class_shap = self.shap_values
        
        importance = np.abs(class_shap).mean(axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def explain_prediction(
        self, 
        X_single: np.ndarray,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explica una predicción individual (para un paciente).
        
        Args:
            X_single: Features de un solo paciente (1D array)
            patient_id: Identificador del paciente
        
        Returns:
            Diccionario con la explicación completa
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        # Predicción
        pred_proba = self.model.predict_proba(X_single)[0]
        pred_class = np.argmax(pred_proba)
        
        # SHAP values para esta muestra
        sample_shap = self.explainer.shap_values(X_single)
        
        # Preparar explicación para la clase predicha
        if isinstance(sample_shap, list):
            shap_for_pred = sample_shap[pred_class][0]
        else:
            shap_for_pred = sample_shap[0, :, pred_class]
        
        # Ordenar contribuciones
        sorted_idx = np.argsort(np.abs(shap_for_pred))[::-1]
        
        contributions = []
        for idx in sorted_idx[:10]:  # Top 10 contribuciones
            contributions.append({
                'feature': self.feature_names[idx],
                'value': float(X_single[0, idx]),
                'shap_value': float(shap_for_pred[idx]),
                'direction': 'aumenta' if shap_for_pred[idx] > 0 else 'disminuye'
            })
        
        explanation = {
            'patient_id': patient_id,
            'predicted_class': int(pred_class),
            'predicted_level': TRIAGE_LEVELS.get(pred_class + 1, f"Nivel {pred_class + 1}"),
            'probabilities': {
                TRIAGE_LEVELS.get(i + 1, f"Nivel {i+1}"): float(p) 
                for i, p in enumerate(pred_proba)
            },
            'confidence': float(pred_proba[pred_class]),
            'top_contributions': contributions,
            'risk_factors': [c for c in contributions if c['shap_value'] > 0][:5],
            'protective_factors': [c for c in contributions if c['shap_value'] < 0][:5]
        }
        
        return explanation
    
    def plot_summary(
        self, 
        X: np.ndarray,
        class_idx: Optional[int] = None,
        max_display: int = 15,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Genera gráfico de resumen SHAP (beeswarm plot).
        
        Args:
            X: Datos para visualizar
            class_idx: Clase específica (None para todas)
            max_display: Número máximo de features a mostrar
            figsize: Tamaño de la figura
            save_path: Ruta para guardar
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=figsize)
        
        if class_idx is not None and isinstance(self.shap_values, list):
            shap_to_plot = self.shap_values[class_idx]
            title = f"SHAP Summary - Nivel {class_idx + 1} ({TRIAGE_LEVELS.get(class_idx + 1, '')})"
        else:
            if isinstance(self.shap_values, list):
                # Promediar para todas las clases
                shap_to_plot = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
            else:
                shap_to_plot = self.shap_values
            title = "SHAP Summary - Todas las Clases"
        
        shap.summary_plot(
            shap_to_plot, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SHAP summary guardado: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Gráfico de barras de importancia global de features.
        """
        importance_df = self.get_global_importance().head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colores por categoría
        category_colors = {
            'Signos Vitales': '#e74c3c',
            'Neurológico': '#9b59b6',
            'Demográfico': '#3498db',
            'Comorbilidades': '#e67e22',
            'Síntomas': '#2ecc71',
            'Otros': '#95a5a6'
        }
        
        colors = [category_colors.get(cat, '#95a5a6') for cat in importance_df['category']]
        
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importancia SHAP (valor absoluto medio)')
        ax.set_title('Importancia Global de Features (SHAP)\nContribución a la clasificación de Triage', fontsize=14)
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) 
                         for cat, color in category_colors.items() 
                         if cat in importance_df['category'].values]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance guardado: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_critical_class_analysis(
        self,
        X: np.ndarray,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ):
        """
        Análisis detallado de features para clases críticas (Nivel 1 y 2).
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for idx, class_idx in enumerate([0, 1]):
            # Importancia para esta clase
            class_importance = self.get_class_importance(class_idx).head(10)
            
            # Gráfico de barras
            ax_bar = axes[idx, 0]
            ax_bar.barh(range(len(class_importance)), class_importance['importance'], 
                       color='#e74c3c' if class_idx == 0 else '#e67e22')
            ax_bar.set_yticks(range(len(class_importance)))
            ax_bar.set_yticklabels(class_importance['feature'])
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel('Importancia SHAP')
            level_name = TRIAGE_LEVELS.get(class_idx + 1, f"Nivel {class_idx + 1}")
            ax_bar.set_title(f'Top Features - Nivel {class_idx + 1}\n({level_name})', fontsize=12)
            
            # SHAP summary para esta clase
            ax_summary = axes[idx, 1]
            plt.sca(ax_summary)
            
            if isinstance(self.shap_values, list):
                shap.summary_plot(
                    self.shap_values[class_idx], 
                    X, 
                    feature_names=self.feature_names,
                    max_display=10,
                    show=False,
                    plot_type='dot'
                )
            ax_summary.set_title(f'Distribución SHAP - Nivel {class_idx + 1}', fontsize=12)
        
        plt.suptitle('Análisis de Features para Clases Críticas\n(Claves para detectar emergencias)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Análisis crítico guardado: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_single_explanation(
        self,
        X_single: np.ndarray,
        patient_id: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualiza la explicación de una predicción individual.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        explanation = self.explain_prediction(X_single, patient_id)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Probabilidades por clase
        ax_prob = axes[0]
        classes = list(explanation['probabilities'].keys())
        probs = list(explanation['probabilities'].values())
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']
        
        bars = ax_prob.bar(range(len(classes)), probs, color=colors)
        ax_prob.set_xticks(range(len(classes)))
        ax_prob.set_xticklabels([f"N{i+1}" for i in range(len(classes))], rotation=45)
        ax_prob.set_ylabel('Probabilidad')
        ax_prob.set_title(f'Probabilidades de Triage\nPredicción: {explanation["predicted_level"]}', fontsize=12)
        ax_prob.set_ylim(0, 1)
        
        # Destacar predicción
        pred_idx = explanation['predicted_class']
        bars[pred_idx].set_edgecolor('black')
        bars[pred_idx].set_linewidth(3)
        
        # Top contribuciones
        ax_contrib = axes[1]
        contributions = explanation['top_contributions'][:8]
        
        features = [c['feature'] for c in contributions]
        shap_vals = [c['shap_value'] for c in contributions]
        colors_contrib = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals]
        
        ax_contrib.barh(range(len(features)), shap_vals, color=colors_contrib)
        ax_contrib.set_yticks(range(len(features)))
        ax_contrib.set_yticklabels([f"{f} = {contributions[i]['value']:.1f}" 
                                    for i, f in enumerate(features)])
        ax_contrib.invert_yaxis()
        ax_contrib.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax_contrib.set_xlabel('Contribución SHAP')
        ax_contrib.set_title('Factores que influyen en la predicción\n(Rojo: aumenta gravedad, Azul: disminuye)', 
                            fontsize=12)
        
        patient_str = f" - Paciente: {patient_id}" if patient_id else ""
        plt.suptitle(f'Explicación de Predicción de Triage{patient_str}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Explicación individual guardada: {save_path}")
        
        plt.show()
        
        return fig, explanation
    
    def generate_clinical_report(
        self,
        X_single: np.ndarray,
        patient_id: Optional[str] = None,
        original_features: Optional[Dict] = None
    ) -> str:
        """
        Genera un reporte clínico textual para una predicción.
        Diseñado para ser comprensible por personal médico.
        """
        explanation = self.explain_prediction(X_single, patient_id)
        
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE CLASIFICACIÓN DE TRIAGE ASISTIDO POR IA")
        report.append("=" * 60)
        
        if patient_id:
            report.append(f"\nPaciente ID: {patient_id}")
        
        report.append(f"\n{'─' * 40}")
        report.append("📋 RESULTADO DE CLASIFICACIÓN")
        report.append(f"{'─' * 40}")
        report.append(f"\n   NIVEL ASIGNADO: {explanation['predicted_class'] + 1}")
        report.append(f"   CATEGORÍA: {explanation['predicted_level']}")
        report.append(f"   CONFIANZA: {explanation['confidence']*100:.1f}%")
        
        report.append(f"\n{'─' * 40}")
        report.append("📊 PROBABILIDADES POR NIVEL")
        report.append(f"{'─' * 40}")
        for level, prob in explanation['probabilities'].items():
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report.append(f"   {level[:25]:<25} [{bar}] {prob*100:5.1f}%")
        
        report.append(f"\n{'─' * 40}")
        report.append("🚨 FACTORES DE RIESGO (aumentan gravedad)")
        report.append(f"{'─' * 40}")
        for factor in explanation['risk_factors']:
            report.append(f"   ⚠️  {factor['feature']}: {factor['value']:.1f}")
            report.append(f"       → Contribución: +{factor['shap_value']:.3f}")
        
        report.append(f"\n{'─' * 40}")
        report.append("✅ FACTORES PROTECTORES (disminuyen gravedad)")
        report.append(f"{'─' * 40}")
        for factor in explanation['protective_factors']:
            report.append(f"   ✓  {factor['feature']}: {factor['value']:.1f}")
            report.append(f"       → Contribución: {factor['shap_value']:.3f}")
        
        report.append(f"\n{'─' * 40}")
        report.append("⚕️  RECOMENDACIÓN")
        report.append(f"{'─' * 40}")
        
        pred_class = explanation['predicted_class']
        if pred_class == 0:
            report.append("   🔴 ATENCIÓN INMEDIATA REQUERIDA")
            report.append("   → Activar protocolo de resucitación")
            report.append("   → Preparar traslado urgente a centro de mayor complejidad")
        elif pred_class == 1:
            report.append("   🟠 EMERGENCIA - Atención prioritaria")
            report.append("   → Evaluación médica inmediata")
            report.append("   → Considerar traslado según evolución")
        elif pred_class == 2:
            report.append("   🟡 URGENTE - Atención en menos de 30 minutos")
            report.append("   → Monitoreo de signos vitales")
        elif pred_class == 3:
            report.append("   🟢 MENOS URGENTE - Atención según disponibilidad")
            report.append("   → Puede esperar atención estándar")
        else:
            report.append("   🔵 NO URGENTE - Atención programada")
            report.append("   → Derivar a consulta externa si es necesario")
        
        report.append(f"\n{'─' * 40}")
        report.append("⚠️  AVISO IMPORTANTE")
        report.append(f"{'─' * 40}")
        report.append("   Esta es una RECOMENDACIÓN asistida por IA.")
        report.append("   La decisión final debe ser tomada por")
        report.append("   personal médico calificado.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_explanations(
        self,
        X: np.ndarray,
        output_dir: Optional[str] = None
    ):
        """
        Guarda todas las visualizaciones de interpretabilidad.
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("📊 GENERANDO VISUALIZACIONES SHAP")
        print("=" * 60)
        
        # Calcular SHAP si no se ha hecho
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Feature importance global
        self.plot_feature_importance(
            save_path=output_dir / "shap_feature_importance.png"
        )
        
        # Summary plot
        self.plot_summary(
            X,
            save_path=output_dir / "shap_summary.png"
        )
        
        # Análisis de clases críticas
        self.plot_critical_class_analysis(
            X,
            save_path=output_dir / "shap_critical_analysis.png"
        )
        
        # Guardar importancia en CSV
        importance_df = self.get_global_importance()
        importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
        
        print(f"\n✅ Visualizaciones SHAP guardadas en: {output_dir}")


if __name__ == "__main__":
    print("Módulo de interpretabilidad - Ejecutar desde el pipeline principal")
