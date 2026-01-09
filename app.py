"""
Aplicación web Flask para clasificación de textos clásicos
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from train_model import TextClassifierTrainer
from preprocessing import TextPreprocessor
from config import CATEGORIAS, MODEL_SAVE_PATH
import os

app = Flask(__name__)
CORS(app)

# Cargar modelo al iniciar
print("🔄 Cargando modelo...")
try:
    trainer = TextClassifierTrainer()
    model, tokenizer = trainer.cargar_modelo()
    trainer.model = model
    trainer.tokenizer = tokenizer
    preprocessor = TextPreprocessor()
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    trainer = None
    preprocessor = None


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    if trainer is None:
        return jsonify({
            'error': 'Modelo no disponible. Por favor, entrena el modelo primero.'
        }), 500
    
    try:
        data = request.get_json()
        texto = data.get('texto', '').strip()
        
        if not texto:
            return jsonify({
                'error': 'Por favor, proporciona un texto para clasificar.'
            }), 400
        
        # Preprocesar texto
        texto_limpio = preprocessor.limpiar_texto(texto)
        
        if not texto_limpio:
            return jsonify({
                'error': 'El texto no pudo ser procesado correctamente.'
            }), 400
        
        # Realizar predicción
        predictions, probabilities = trainer.predecir([texto_limpio], return_probs=True)
        
        categoria_idx = int(predictions[0])
        categoria = CATEGORIAS[categoria_idx]
        confianza = float(probabilities[0][categoria_idx]) * 100
        
        # Obtener probabilidades de todas las categorías
        probabilidades = {}
        for idx, cat in CATEGORIAS.items():
            probabilidades[cat] = float(probabilities[0][idx]) * 100
        
        return jsonify({
            'categoria': categoria,
            'confianza': round(confianza, 2),
            'probabilidades': probabilidades,
            'texto_procesado': texto_limpio
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la predicción: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de salud del servicio"""
    return jsonify({
        'status': 'ok',
        'modelo_cargado': trainer is not None
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



