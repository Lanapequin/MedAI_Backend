# -*- coding: utf-8 -*-
"""
evaluation.py
Módulo de evaluación completa del modelo de Triage
Incluye visualizaciones, métricas detalladas y análisis de errores
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, f1_score, precision_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TRIAGE_LEVELS, CRITICAL_CLASSES, TARGET_RECALL_CRITICAL,
    OUTPUTS_DIR, MODELS_DIR
)


class TriageEvaluator:
    """
    Evaluador completo para el modelo de Triage.
    Genera métricas, visualizaciones y análisis de errores.
    """
    
    def __init__(self, model, class_names: Optional[List[str]] = None):
        """
        Args:
            model: Modelo entrenado con métodos predict y predict_proba
            class_names: Nombres de las clases para visualización
        """
        self.model = model
        self.class_names = class_names or [
            f"Nivel {i+1}: {TRIAGE_LEVELS[i+1]}" 
            for i in range(len(TRIAGE_LEVELS))
        ]
        self.n_classes = len(self.class_names)
        self.evaluation_results = {}
        
    def evaluate(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray,
        set_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Realiza evaluación completa del modelo.
        
        Args:
            X: Features
            y_true: Labels verdaderas
            set_name: Nombre del conjunto (para reportes)
        
        Returns:
            Diccionario con todas las métricas
        """
        print("\n" + "=" * 70)
        print(f"📊 EVALUACIÓN COMPLETA - {set_name.upper()}")
        print("=" * 70)
        
        # Predicciones
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Métricas básicas
        metrics = self._compute_basic_metrics(y_true, y_pred)
        
        # Métricas por clase
        class_metrics = self._compute_class_metrics(y_true, y_pred)
        
        # Análisis de clases críticas
        critical_analysis = self._analyze_critical_classes(y_true, y_pred, y_pred_proba)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Compilar resultados
        self.evaluation_results = {
            'set_name': set_name,
            'basic_metrics': metrics,
            'class_metrics': class_metrics,
            'critical_analysis': critical_analysis,
            'confusion_matrix': cm,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Imprimir reporte
        self._print_report()
        
        return self.evaluation_results
    
    def _compute_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas básicas."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _compute_class_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Dict]:
        """Calcula métricas por clase."""
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Contar muestras por clase
        unique, counts = np.unique(y_true, return_counts=True)
        support = dict(zip(unique, counts))
        
        class_metrics = {}
        for cls in range(self.n_classes):
            class_metrics[cls] = {
                'recall': recall_per_class[cls] if cls < len(recall_per_class) else 0,
                'precision': precision_per_class[cls] if cls < len(precision_per_class) else 0,
                'f1': f1_per_class[cls] if cls < len(f1_per_class) else 0,
                'support': support.get(cls, 0)
            }
        
        return class_metrics
    
    def _analyze_critical_classes(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Análisis detallado de clases críticas (Nivel 1 y 2).
        """
        critical_classes = [0, 1]  # 0-indexed
        
        analysis = {}
        
        for cls in critical_classes:
            # Máscara de verdaderos positivos de esta clase
            true_mask = (y_true == cls)
            pred_mask = (y_pred == cls)
            
            # True Positives, False Negatives
            tp = np.sum(true_mask & pred_mask)
            fn = np.sum(true_mask & ~pred_mask)
            fp = np.sum(~true_mask & pred_mask)
            tn = np.sum(~true_mask & ~pred_mask)
            
            # Recall (Sensibilidad) - CRÍTICO
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Precisión
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Especificidad
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Análisis de falsos negativos (CRÍTICO)
            fn_indices = np.where(true_mask & ~pred_mask)[0]
            fn_predicted_classes = y_pred[fn_indices]
            
            analysis[cls] = {
                'true_positives': int(tp),
                'false_negatives': int(fn),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'recall': float(recall),
                'precision': float(precision),
                'specificity': float(specificity),
                'fn_misclassified_as': dict(zip(*np.unique(fn_predicted_classes, return_counts=True))) if len(fn_predicted_classes) > 0 else {},
                'meets_target': recall >= TARGET_RECALL_CRITICAL
            }
        
        # Recall promedio de clases críticas
        analysis['critical_recall_avg'] = np.mean([
            analysis[cls]['recall'] for cls in critical_classes
        ])
        analysis['meets_overall_target'] = analysis['critical_recall_avg'] >= TARGET_RECALL_CRITICAL
        
        return analysis
    
    def _print_report(self):
        """Imprime el reporte de evaluación."""
        results = self.evaluation_results
        
        # Métricas básicas
        print(f"\n{'─' * 70}")
        print("📈 MÉTRICAS GENERALES")
        print(f"{'─' * 70}")
        
        metrics = results['basic_metrics']
        print(f"   Accuracy:           {metrics['accuracy']*100:.2f}%")
        print(f"   F1-Score Weighted:  {metrics['f1_weighted']*100:.2f}%")
        print(f"   F1-Score Macro:     {metrics['f1_macro']*100:.2f}%")
        print(f"   Precision Weighted: {metrics['precision_weighted']*100:.2f}%")
        print(f"   Recall Weighted:    {metrics['recall_weighted']*100:.2f}%")
        
        # Análisis crítico
        print(f"\n{'─' * 70}")
        print(f"🚨 ANÁLISIS DE CLASES CRÍTICAS (Objetivo: Recall ≥ {TARGET_RECALL_CRITICAL*100}%)")
        print(f"{'─' * 70}")
        
        critical = results['critical_analysis']
        
        for cls in [0, 1]:
            cls_analysis = critical[cls]
            level_name = TRIAGE_LEVELS.get(cls + 1, f"Nivel {cls+1}")
            status = "✅ CUMPLE" if cls_analysis['meets_target'] else "❌ NO CUMPLE"
            
            print(f"\n   📋 NIVEL {cls+1} ({level_name}) - {status}")
            print(f"      Recall (Sensibilidad): {cls_analysis['recall']*100:.2f}%")
            print(f"      Precisión:             {cls_analysis['precision']*100:.2f}%")
            print(f"      Especificidad:         {cls_analysis['specificity']*100:.2f}%")
            print(f"      True Positives:  {cls_analysis['true_positives']}")
            print(f"      False Negatives: {cls_analysis['false_negatives']} ⚠️")
            print(f"      False Positives: {cls_analysis['false_positives']}")
            
            if cls_analysis['fn_misclassified_as']:
                print(f"      Falsos Negativos clasificados como:")
                for pred_cls, count in cls_analysis['fn_misclassified_as'].items():
                    pred_name = TRIAGE_LEVELS.get(int(pred_cls) + 1, f"Nivel {int(pred_cls)+1}")
                    print(f"         → Nivel {int(pred_cls)+1} ({pred_name}): {count}")
        
        # Resumen crítico
        print(f"\n{'─' * 70}")
        overall_status = "✅" if critical['meets_overall_target'] else "❌"
        print(f"   {overall_status} RECALL PROMEDIO CRÍTICO: {critical['critical_recall_avg']*100:.2f}%")
        print(f"{'─' * 70}")
        
        # Métricas por clase
        print(f"\n{'─' * 70}")
        print("📊 MÉTRICAS POR CLASE")
        print(f"{'─' * 70}")
        
        class_metrics = results['class_metrics']
        print(f"\n   {'Nivel':<35} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Support':>10}")
        print(f"   {'-'*75}")
        
        for cls in range(self.n_classes):
            if cls in class_metrics:
                m = class_metrics[cls]
                level_name = f"Nivel {cls+1}: {TRIAGE_LEVELS.get(cls+1, 'Unknown')[:20]}"
                print(f"   {level_name:<35} {m['recall']*100:>9.1f}% {m['precision']*100:>9.1f}% {m['f1']*100:>9.1f}% {m['support']:>10}")
    
    def plot_confusion_matrix(
        self, 
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Genera visualización de matriz de confusión.
        """
        cm = self.evaluation_results['confusion_matrix']
        
        # Normalizar por filas (recall)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Matriz de confusión absoluta
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[f"N{i+1}" for i in range(self.n_classes)],
            yticklabels=[f"N{i+1}" for i in range(self.n_classes)],
            ax=axes[0]
        )
        axes[0].set_title('Matriz de Confusión (Conteos)', fontsize=14)
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Valor Real')
        
        # Destacar clases críticas
        for i in [0, 1]:
            axes[0].add_patch(plt.Rectangle((0, i), self.n_classes, 1, 
                                            fill=False, edgecolor='red', linewidth=3))
        
        # Matriz normalizada
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='RdYlGn',
            xticklabels=[f"N{i+1}" for i in range(self.n_classes)],
            yticklabels=[f"N{i+1}" for i in range(self.n_classes)],
            ax=axes[1],
            vmin=0, vmax=1
        )
        axes[1].set_title('Matriz de Confusión Normalizada (Recall)', fontsize=14)
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Valor Real')
        
        # Destacar clases críticas
        for i in [0, 1]:
            axes[1].add_patch(plt.Rectangle((0, i), self.n_classes, 1, 
                                            fill=False, edgecolor='red', linewidth=3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de confusión guardada: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_class_metrics(
        self, 
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualiza métricas por clase con énfasis en clases críticas.
        """
        class_metrics = self.evaluation_results['class_metrics']
        
        classes = list(class_metrics.keys())
        recalls = [class_metrics[c]['recall'] for c in classes]
        precisions = [class_metrics[c]['precision'] for c in classes]
        f1s = [class_metrics[c]['f1'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, recalls, width, label='Recall', color='#2ecc71')
        bars2 = ax.bar(x, precisions, width, label='Precisión', color='#3498db')
        bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#9b59b6')
        
        # Línea de objetivo para recall
        ax.axhline(y=TARGET_RECALL_CRITICAL, color='red', linestyle='--', 
                   linewidth=2, label=f'Objetivo Recall ({TARGET_RECALL_CRITICAL*100}%)')
        
        # Destacar clases críticas
        for i in [0, 1]:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.2, color='red')
        
        ax.set_xlabel('Nivel de Triage')
        ax.set_ylabel('Score')
        ax.set_title('Métricas por Clase de Triage\n(Áreas rojas: Clases Críticas)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'N{c+1}' for c in classes])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        
        # Añadir valores sobre las barras
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Métricas por clase guardadas: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_roc_curves(
        self, 
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Genera curvas ROC para cada clase (One-vs-Rest).
        """
        y_true = self.evaluation_results['predictions']['y_true']
        y_pred_proba = self.evaluation_results['predictions']['y_pred_proba']
        
        # Binarizar labels
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']
        
        for i in range(self.n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            level_name = TRIAGE_LEVELS.get(i + 1, f"Nivel {i+1}")
            linestyle = '-' if i in [0, 1] else '--'
            linewidth = 3 if i in [0, 1] else 1.5
            
            ax.plot(fpr, tpr, color=colors[i], linestyle=linestyle, linewidth=linewidth,
                   label=f'N{i+1} ({level_name[:15]}) AUC={roc_auc:.3f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        ax.set_title('Curvas ROC por Clase\n(Líneas sólidas: Clases Críticas)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Curvas ROC guardadas: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_precision_recall_curves(
        self, 
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Genera curvas Precision-Recall para cada clase.
        """
        y_true = self.evaluation_results['predictions']['y_true']
        y_pred_proba = self.evaluation_results['predictions']['y_pred_proba']
        
        # Binarizar labels
        y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']
        
        for i in range(self.n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            level_name = TRIAGE_LEVELS.get(i + 1, f"Nivel {i+1}")
            linestyle = '-' if i in [0, 1] else '--'
            linewidth = 3 if i in [0, 1] else 1.5
            
            ax.plot(recall, precision, color=colors[i], linestyle=linestyle, linewidth=linewidth,
                   label=f'N{i+1} ({level_name[:15]}) AP={ap:.3f}')
        
        # Línea de objetivo de recall
        ax.axvline(x=TARGET_RECALL_CRITICAL, color='red', linestyle=':', 
                   linewidth=2, label=f'Objetivo Recall ({TARGET_RECALL_CRITICAL*100}%)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensibilidad)')
        ax.set_ylabel('Precisión')
        ax.set_title('Curvas Precision-Recall por Clase\n(Líneas sólidas: Clases Críticas)', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Curvas P-R guardadas: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_full_report(
        self, 
        output_dir: Optional[str] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Genera reporte completo con todas las visualizaciones.
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("📄 GENERANDO REPORTE COMPLETO")
        print("=" * 70)
        
        # Generar visualizaciones
        if save_plots:
            self.plot_confusion_matrix(
                save_path=output_dir / "confusion_matrix.png"
            )
            
            self.plot_class_metrics(
                save_path=output_dir / "class_metrics.png"
            )
            
            self.plot_roc_curves(
                save_path=output_dir / "roc_curves.png"
            )
            
            self.plot_precision_recall_curves(
                save_path=output_dir / "pr_curves.png"
            )
        
        # Guardar métricas en JSON
        import json
        
        metrics_to_save = {
            'basic_metrics': self.evaluation_results['basic_metrics'],
            'critical_analysis': {
                k: {kk: vv for kk, vv in v.items() if kk != 'fn_misclassified_as'} 
                if isinstance(v, dict) else v
                for k, v in self.evaluation_results['critical_analysis'].items()
            },
            'class_metrics': {
                str(k): v for k, v in self.evaluation_results['class_metrics'].items()
            }
        }
        
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        print(f"📄 Métricas guardadas: {metrics_path}")
        
        print(f"\n✅ Reporte generado en: {output_dir}")
        
        return self.evaluation_results


def evaluate_model(model, X_test, y_test, output_dir=None):
    """
    Función auxiliar para evaluar un modelo rápidamente.
    """
    evaluator = TriageEvaluator(model)
    results = evaluator.evaluate(X_test, y_test, "Test")
    
    if output_dir:
        evaluator.generate_full_report(output_dir, save_plots=True)
    
    return evaluator, results


if __name__ == "__main__":
    print("Módulo de evaluación - Ejecutar desde el pipeline principal")
