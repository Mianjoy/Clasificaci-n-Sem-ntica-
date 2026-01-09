# Clasificador Semántico de Textos Clásicos mediante Modelos Masivos de Lenguaje

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-blue)](https://USERNAME.github.io/clasificador-textos-clasicos/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Sistema de Inteligencia Artificial para la clasificación automática de fragmentos de textos clásicos en tres categorías temáticas y filosóficas: **Areté**, **Poder y Política**, y **Relación entre Humanos y Dioses**.

🌐 **[Ver página en GitHub Pages](https://USERNAME.github.io/clasificador-textos-clasicos/)**

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de clasificación de textos utilizando técnicas de Deep Learning y Fine-Tuning sobre modelos de lenguaje pre-entrenados. El sistema está diseñado para ser accesible a investigadores en humanidades sin conocimientos técnicos previos.

## 🎯 Características Principales

- ✅ **Base de Datos Estructurada**: Migración de datos Excel a SQLite
- ✅ **Preprocesamiento Avanzado**: Limpieza y normalización de textos
- ✅ **Balanceo de Clases**: Técnica SMOTE+Tomek para balancear el dataset
- ✅ **Fine-Tuning de LLM**: Entrenamiento con DistilBERT
- ✅ **Evaluación Completa**: Matriz de confusión y métricas detalladas
- ✅ **Interfaz Web Moderna**: GUI accesible y estéticamente diseñada
- ✅ **Interpretabilidad**: Muestra probabilidades por categoría

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Descargar recursos de NLTK** (se descargan automáticamente, pero si hay problemas):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📖 Uso

### 🔄 Clonar el Repositorio

**⚠️ Importante**: Este repositorio usa Git LFS para archivos grandes. Asegúrate de tener Git LFS instalado antes de clonar:

```bash
# Instalar Git LFS (si no lo tienes)
git lfs install

# Clonar el repositorio
git clone https://github.com/USERNAME/clasificador-textos-clasicos.git
cd clasificador-textos-clasicos
```

Si ya clonaste el repositorio sin Git LFS instalado, ejecuta:
```bash
git lfs install
git lfs pull
```

### 1. Preparar los Datos

**Opción A: Usar la base de datos incluida (Recomendado)**
El repositorio incluye la base de datos `data/textos_clasicos.db` con los datos ya importados. Puedes saltarte el paso de importación.

**Opción B: Importar desde Excel**
Si prefieres usar tus propios datos, asegúrese de que los archivos Excel (`0.xlsx`, `1.xlsx`, ..., `6.xlsx`) estén en el directorio raíz del proyecto y ejecute:
```bash
python database.py
```

### 2. Usar el Modelo Entrenado (Incluido en el Repositorio)

**Si clonas este repositorio desde GitHub**, el modelo ya está entrenado y listo para usar. Puedes iniciar directamente la aplicación web:

```bash
python app.py
```

O usar el script de inicio:

```bash
# Windows
INICIAR_APP.bat

# Linux/Mac
python app.py
```

### 3. Entrenar un Modelo Nuevo (Opcional)

Si deseas reentrenar el modelo desde cero:

**Opción A: Script Python (Recomendado)**
```bash
python run_pipeline.py
```

Este script ejecuta:
- Importación de datos desde Excel a base de datos SQLite (si no existe la BD)
- Preprocesamiento y balanceo de clases
- Entrenamiento del modelo con fine-tuning
- Evaluación del modelo y generación de métricas

### 4. Ejecutar Componentes Individuales

#### Importar datos a base de datos:
```bash
python database.py
```

#### Preprocesar datos:
```bash
python preprocessing.py
```

#### Entrenar modelo:
```bash
python train_model.py
```

#### Evaluar modelo:
```bash
python evaluate_model.py
```

### 5. Iniciar la Aplicación Web

#### Desarrollo:
```bash
python app.py
```

#### Producción (con Gunicorn):
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Luego, abra su navegador en: `http://localhost:5000`

## 📁 Estructura del Proyecto

```
.
├── app.py                  # Aplicación web Flask
├── config.py               # Configuración del proyecto
├── database.py             # Gestión de base de datos
├── preprocessing.py        # Preprocesamiento y balanceo
├── train_model.py          # Entrenamiento del modelo
├── evaluate_model.py       # Evaluación del modelo
├── run_pipeline.py         # Script principal
├── requirements.txt        # Dependencias
├── README.md              # Este archivo
├── templates/
│   └── index.html         # Interfaz web
├── data/
│   └── textos_clasicos.db # Base de datos SQLite (incluida en el repo)
├── models/
│   └── clasificador_textos_clasicos/  # Modelo entrenado (incluido en el repo)
│       ├── pytorch_model.bin          # Modelo entrenado final
│       ├── config.json                # Configuración del modelo
│       ├── tokenizer*.json            # Archivos del tokenizer
│       └── ...                        # Otros archivos necesarios
├── results/               # Resultados de evaluación (se generan al evaluar)
└── logs/                  # Logs de entrenamiento (excluidos del repo)
```

## 🔧 Configuración

Puede modificar los parámetros en `config.py`:

- `MODEL_NAME`: Modelo base a usar (por defecto: `distilbert-base-uncased`)
- `MAX_LENGTH`: Longitud máxima de tokens (por defecto: 512)
- `BATCH_SIZE`: Tamaño de lote (por defecto: 16)
- `LEARNING_RATE`: Tasa de aprendizaje (por defecto: 2e-5)
- `NUM_EPOCHS`: Número de épocas (por defecto: 3)

## 📊 Métricas de Evaluación

El sistema genera:

- **Matriz de Confusión**: Visualización de aciertos y errores
- **Precisión (Precision)**: Por clase y promedio ponderado
- **Sensibilidad (Recall)**: Por clase y promedio ponderado
- **F1-Score**: Por clase y promedio ponderado
- **Accuracy**: Precisión general

**Criterio de Aceptación**: F1-Score ≥ 0.8

Los resultados se guardan en:
- `results/matriz_confusion.png`
- `results/reporte_evaluacion.csv`

## 🎨 Interfaz de Usuario

La interfaz web ofrece:

- Diseño moderno y limpio
- Entrada de texto intuitiva
- Visualización clara de resultados
- Probabilidades por categoría con barras de progreso
- Diseño responsive (adaptable a móviles)

## 🔬 Metodología

### 1. Ingesta y Gestión de Datos
- Migración de Excel a SQLite
- Estructura normalizada con integridad referencial
- Consultas optimizadas

### 2. Preprocesamiento
- Limpieza de texto (normalización, eliminación de caracteres especiales)
- Tokenización y lematización
- Balanceo con SMOTE+Tomek Links

### 3. Modelado
- Fine-tuning de DistilBERT (modelo ligero y eficiente)
- Transfer Learning desde modelo pre-entrenado
- Entrenamiento con validación temprana

### 4. Evaluación
- División train/validation/test (70/10/20)
- Métricas estándar de clasificación
- Visualizaciones profesionales

## 📝 Notas Técnicas

- **Modelo Base**: DistilBERT es una versión ligera de BERT, ideal para tareas de clasificación de texto
- **Balanceo**: SMOTE+Tomek combina oversampling sintético con limpieza de ejemplos ambiguos
- **GPU**: El sistema detecta automáticamente si hay GPU disponible y usa FP16 para acelerar el entrenamiento

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
**Si clonaste desde GitHub**: El modelo ya está incluido. Si aparece este error, verifica que exista `models/clasificador_textos_clasicos/pytorch_model.bin`

**Si estás entrenando desde cero**: Ejecute `python train_model.py` o `python run_pipeline.py`

### Error: "Base de datos vacía"
**Si clonaste desde GitHub**: La base de datos ya está incluida en `data/textos_clasicos.db`. Si aparece este error, verifica que el archivo existe.

**Si estás usando tus propios datos**: Ejecute `python database.py` para importar los datos desde Excel

### Error al instalar dependencias
Asegúrese de tener Python 3.8+ y actualice pip:
```bash
pip install --upgrade pip
```

## 🌐 GitHub Pages

Este proyecto incluye una página web estática en [GitHub Pages](https://USERNAME.github.io/clasificador-textos-clasicos/) que muestra información sobre el proyecto, características, instalación y uso.

**Para activar GitHub Pages:**
1. Ve a Settings → Pages en tu repositorio de GitHub
2. Selecciona la fuente: `Deploy from a branch`
3. Selecciona la rama: `main` o `master`
4. Selecciona la carpeta: `/docs`
5. Haz clic en Save
6. Tu página estará disponible en: `https://USERNAME.github.io/clasificador-textos-clasicos/`

## 🚀 Subir el Proyecto a GitHub

Si deseas publicar este proyecto en GitHub, consulta las guías:

- **[GUIA_RAPIDA_GITHUB.md](GUIA_RAPIDA_GITHUB.md)** - Guía rápida paso a paso ⚡
- **[GITHUB_SETUP.md](GITHUB_SETUP.md)** - Guía detallada completa 📖

**Opción 1: Usar el script automático (Recomendado)**

```powershell
# Windows PowerShell
.\subir_a_github.ps1
```

Este script configurará Git LFS, preparará los archivos y te guiará paso a paso.

**Opción 2: Comandos manuales**

```bash
# 1. Inicializar Git LFS
git lfs install

# 2. Agregar archivos
git add .
git commit -m "Initial commit: Clasificador de Textos Clásicos con modelo entrenado"

# 3. Configurar repositorio remoto (reemplaza USERNAME)
git remote add origin https://github.com/USERNAME/clasificador-textos-clasicos.git
git branch -M main

# 4. Subir a GitHub
git push -u origin main
```

**⚠️ Nota importante**: El modelo entrenado es grande (>200MB), por lo que se requiere Git LFS para subirlo a GitHub. Los scripts incluidos configuran esto automáticamente.

**Nota importante**: El repositorio incluye:
- ✅ Base de datos entrenada (`data/textos_clasicos.db`)
- ✅ Modelo entrenado completo (`models/clasificador_textos_clasicos/`)
- ✅ Todos los archivos necesarios para usar el sistema inmediatamente

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autor

Desarrollado como proyecto académico para clasificación semántica de textos clásicos.

## 🙏 Agradecimientos

- Hugging Face por los modelos pre-entrenados
- Comunidad de código abierto por las librerías utilizadas

---

**Versión**: 1.0.0  
**Última actualización**: 2024

