"""
Módulo de evaluación del modelo
Genera matriz de confusión y métricas detalladas
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
import pandas as pd
import os
from config import CATEGORIAS, RESULTS_DIR
from train_model import TextClassifierTrainer


class ModelEvaluator:
    """Evaluador del modelo"""
    
    def __init__(self, model_path=None):
        self.trainer = TextClassifierTrainer()
        self.model, self.tokenizer = self.trainer.cargar_modelo(model_path)
        self.trainer.model = self.model
        self.trainer.tokenizer = self.tokenizer
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo en el conjunto de prueba"""
        print("\n📊 Evaluando modelo en conjunto de prueba...")
        
        # Realizar predicciones
        predictions, probabilities = self.trainer.predecir(X_test, return_probs=True)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # Métricas por clase
        precision_per_class = precision_score(y_test, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, predictions, average=None, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, predictions)
        
        # Reporte de clasificación
        class_names = [CATEGORIAS[i] for i in range(len(CATEGORIAS))]
        report = classification_report(
            y_test, predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Mostrar resultados
        print("\n" + "="*60)
        print("RESULTADOS DE EVALUACIÓN")
        print("="*60)
        print(f"\nMétricas Generales:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nMétricas por Categoría:")
        for i, categoria in enumerate(class_names):
            print(f"\n  {categoria}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1-Score:  {f1_per_class[i]:.4f}")
        
        print("\n" + "="*60)
        
        # Verificar criterio de aceptación
        if f1 >= 0.8:
            print(f"✅ Criterio de aceptación cumplido: F1-Score = {f1:.4f} >= 0.8")
        else:
            print(f"⚠️  Criterio de aceptación NO cumplido: F1-Score = {f1:.4f} < 0.8")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def generar_matriz_confusion(self, y_test, predictions, save_path=None):
        """Genera y guarda la matriz de confusión"""
        cm = confusion_matrix(y_test, predictions)
        class_names = [CATEGORIAS[i] for i in range(len(CATEGORIAS))]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Cantidad'}
        )
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Etiqueta Predicha', fontsize=12)
        plt.tight_layout()
        
        # Guardar
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, 'matriz_confusion.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Matriz de confusión guardada en: {save_path}")
        plt.close()
    
    def generar_reporte_completo(self, y_test, predictions, save_path=None):
        """Genera un reporte completo en CSV"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, 'reporte_evaluacion.csv')
        
        class_names = [CATEGORIAS[i] for i in range(len(CATEGORIAS))]
        
        # Métricas por clase
        precision_per_class = precision_score(y_test, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, predictions, average=None, zero_division=0)
        
        # Crear DataFrame
        report_data = {
            'Categoría': class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class
        }
        
        df = pd.DataFrame(report_data)
        
        # Agregar métricas generales
        general_metrics = pd.DataFrame({
            'Categoría': ['PROMEDIO PONDERADO'],
            'Precision': [precision_score(y_test, predictions, average='weighted', zero_division=0)],
            'Recall': [recall_score(y_test, predictions, average='weighted', zero_division=0)],
            'F1-Score': [f1_score(y_test, predictions, average='weighted', zero_division=0)]
        })
        
        df = pd.concat([df, general_metrics], ignore_index=True)
        
        # Guardar
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"💾 Reporte guardado en: {save_path}")
        
        return df


if __name__ == "__main__":
    from database import DatabaseManager
    from preprocessing import preparar_datos
    
    # Preparar datos
    print("📊 Preparando datos de prueba...")
    db_manager = DatabaseManager()
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(db_manager, balancear=False)
    
    # Evaluar modelo
    evaluator = ModelEvaluator()
    results = evaluator.evaluar(X_test, y_test)
    
    # Generar visualizaciones
    evaluator.generar_matriz_confusion(y_test, results['predictions'])
    evaluator.generar_reporte_completo(y_test, results['predictions'])
    
    print("\n✅ Evaluación completada!")



