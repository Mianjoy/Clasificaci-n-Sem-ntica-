"""
Configuración del proyecto
"""
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Base de datos
DATABASE_PATH = os.path.join(DATA_DIR, "textos_clasicos.db")

# Modelo
MODEL_NAME = "distilbert-base-uncased"  # Modelo más ligero para fine-tuning
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "clasificador_textos_clasicos")

# Asegurar que el directorio del modelo existe
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Categorías
CATEGORIAS = {
    0: "Areté",
    1: "Poder y Política",
    2: "Relación entre Humanos y Dioses"
}

# Parámetros de entrenamiento
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)



