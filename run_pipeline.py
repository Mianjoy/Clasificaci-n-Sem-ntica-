"""
Script principal para ejecutar todo el pipeline del proyecto
"""
import os
import sys

def main():
    print("="*60)
    print("PIPELINE DE CLASIFICACIÓN DE TEXTOS CLÁSICOS")
    print("="*60)
    
    # Paso 1: Importar datos a base de datos
    print("\n📊 PASO 1: Importando datos desde Excel a base de datos...")
    from database import DatabaseManager
    db_manager = DatabaseManager()
    total = db_manager.importar_todos_excel()
    print(f"✅ {total} registros importados")
    db_manager.estadisticas()
    
    if total == 0:
        print("\n❌ No se importaron datos. Verifique que los archivos Excel existan.")
        return
    
    # Paso 2: Preprocesar y balancear datos
    print("\n📊 PASO 2: Preprocesando y balanceando datos...")
    from preprocessing import preparar_datos
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(db_manager)
    
    # Paso 3: Entrenar modelo
    print("\n📊 PASO 3: Entrenando modelo...")
    from train_model import TextClassifierTrainer
    trainer = TextClassifierTrainer()
    trainer_model, eval_results = trainer.entrenar(X_train, y_train, X_val, y_val)
    
    # Paso 4: Evaluar modelo
    print("\n📊 PASO 4: Evaluando modelo en conjunto de prueba...")
    from evaluate_model import ModelEvaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluar(X_test, y_test)
    
    # Generar visualizaciones
    evaluator.generar_matriz_confusion(y_test, results['predictions'])
    evaluator.generar_reporte_completo(y_test, results['predictions'])
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO")
    print("="*60)
    print("\nPara iniciar la aplicación web, ejecute:")
    print("  python app.py")
    print("\nO use gunicorn para producción:")
    print("  gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    
    db_manager.cerrar()


if __name__ == "__main__":
    main()



