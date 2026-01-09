"""
Módulo de preprocesamiento y balanceo de clases
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from config import CATEGORIAS, TEST_SIZE, VAL_SIZE

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Preprocesador de texto para textos clásicos"""
    
    def __init__(self, language='spanish'):
        self.language = language
        self.stop_words = set(stopwords.words('spanish')) if language == 'spanish' else set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def limpiar_texto(self, texto):
        """
        Limpia y normaliza el texto
        """
        if pd.isna(texto) or texto == '':
            return ''
        
        texto = str(texto)
        
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Eliminar caracteres especiales pero mantener acentos y puntuación básica
        texto = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ.,;:!?¿¡]', '', texto)
        
        # Normalizar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto)
        
        # Eliminar espacios al inicio y final
        texto = texto.strip()
        
        return texto
    
    def preprocesar(self, textos):
        """
        Preprocesa una lista de textos
        """
        textos_limpios = [self.limpiar_texto(texto) for texto in textos]
        return textos_limpios


class DataBalancer:
    """Balanceador de clases para el dataset"""
    
    def __init__(self, estrategia='oversample'):
        """
        Args:
            estrategia: 'oversample' (oversampling aleatorio), 'undersample' (undersampling), 
                      'none' (sin balanceo)
        Nota: SMOTE no funciona directamente con texto, usamos oversampling aleatorio
        """
        self.estrategia = estrategia
        self.balancer = None
        
        if estrategia == 'oversample':
            self.balancer = 'oversample'
        elif estrategia == 'undersample':
            self.balancer = RandomUnderSampler(random_state=42)
        else:
            self.balancer = None
    
    def balancear(self, X, y):
        """
        Balancea las clases del dataset
        """
        if self.balancer is None:
            print("⚠️  No se aplicará balanceo de clases")
            return X, y
        
        print(f"📊 Aplicando balanceo: {self.estrategia}")
        print(f"   Distribución antes: {pd.Series(y).value_counts().to_dict()}")
        
        if self.balancer == 'oversample':
            # Oversampling aleatorio (replicar ejemplos de clases minoritarias)
            X_list = X.tolist() if isinstance(X, np.ndarray) else list(X)
            y_list = y.tolist() if isinstance(y, np.ndarray) else list(y)
            
            df = pd.DataFrame({'texto': X_list, 'categoria': y_list})
            counts = df['categoria'].value_counts()
            max_count = counts.max()
            
            X_balanced = []
            y_balanced = []
            
            for categoria in df['categoria'].unique():
                categoria_df = df[df['categoria'] == categoria]
                current_count = len(categoria_df)
                
                # Agregar todos los ejemplos originales
                X_balanced.extend(categoria_df['texto'].tolist())
                y_balanced.extend(categoria_df['categoria'].tolist())
                
                # Oversample si es necesario
                if current_count < max_count:
                    samples_needed = max_count - current_count
                    oversampled = categoria_df.sample(
                        n=samples_needed, 
                        replace=True, 
                        random_state=42
                    )
                    X_balanced.extend(oversampled['texto'].tolist())
                    y_balanced.extend(oversampled['categoria'].tolist())
            
            X_balanced = np.array(X_balanced)
            y_balanced = np.array(y_balanced)
        else:
            # Undersampling
            X_balanced, y_balanced = self.balancer.fit_resample(
                np.array(X).reshape(-1, 1) if isinstance(X, list) else X.reshape(-1, 1),
                y
            )
            X_balanced = X_balanced.flatten()
        
        print(f"   Distribución después: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def justificar_estrategia(self):
        """
        Justifica la estrategia de balanceo seleccionada
        """
        justificaciones = {
            'oversample': """
            Random Over-Sampling (Recomendado para texto):
            - Replica ejemplos de las clases minoritarias aleatoriamente
            - Mantiene toda la información de las clases mayoritarias
            - Simple y efectivo para datos de texto
            - No requiere representación numérica como SMOTE
            - Preserva la integridad semántica del texto original
            """,
            'undersample': """
            Random Under-Sampling:
            - Reduce el tamaño de las clases mayoritarias
            - Puede perder información valiosa
            - Útil cuando hay muchos datos y desequilibrio extremo
            """,
            'none': """
            Sin balanceo:
            - Se mantiene la distribución original
            - Útil si el desequilibrio es natural y aceptable
            """
        }
        
        return justificaciones.get(self.estrategia, "Estrategia no documentada")


def preparar_datos(db_manager, test_size=None, val_size=None, balancear=True):
    """
    Prepara los datos para entrenamiento
    
    Args:
        db_manager: Instancia de DatabaseManager
        test_size: Tamaño del conjunto de prueba (default: 0.2)
        val_size: Tamaño del conjunto de validación (default: 0.1)
        balancear: Si aplicar balanceo de clases
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Usar valores por defecto si no se especifican
    if test_size is None:
        test_size = TEST_SIZE
    if val_size is None:
        val_size = VAL_SIZE
    
    # Obtener datos de la base de datos
    df = db_manager.obtener_dataframe()
    
    if df.empty:
        raise ValueError(
            "❌ La base de datos está vacía.\n"
            "   Ejecuta 'python database.py' para importar los datos desde Excel.\n"
            "   O ejecuta 'python verificar_datos.py' para diagnosticar el problema."
        )
    
    # Preprocesar textos
    preprocessor = TextPreprocessor()
    df['texto_limpio'] = preprocessor.preprocesar(df['texto'].tolist())
    
    # Filtrar textos vacíos
    df = df[df['texto_limpio'].str.len() > 0]
    
    # Mapear categorías a números
    categoria_to_num = {cat: idx for idx, cat in CATEGORIAS.items()}
    df['categoria_num'] = df['categoria'].map(categoria_to_num)
    
    # Eliminar categorías no reconocidas
    df = df.dropna(subset=['categoria_num'])
    
    # Asegurar que las etiquetas sean enteros
    df['categoria_num'] = df['categoria_num'].astype(int)
    
    X = df['texto_limpio'].values
    y = df['categoria_num'].values.astype(int)  # Asegurar que sean enteros
    
    print(f"\n📊 Datos originales: {len(X)} textos")
    print(f"   Distribución: {pd.Series(y).value_counts().to_dict()}")
    
    # Validar que hay datos
    if len(X) == 0:
        raise ValueError(
            "❌ No hay datos válidos después del preprocesamiento.\n"
            "   Posibles causas:\n"
            "   1. La base de datos está vacía\n"
            "   2. Las categorías no coinciden con las esperadas\n"
            "   3. Los textos están vacíos después de la limpieza\n"
            "   Ejecuta 'python database.py' para verificar la importación."
        )
    
    # Validar que hay al menos una muestra por categoría
    if len(np.unique(y)) < len(CATEGORIAS):
        print(f"⚠️  Advertencia: Solo se encontraron {len(np.unique(y))} categorías de {len(CATEGORIAS)} esperadas")
        print(f"   Categorías encontradas: {[CATEGORIAS[int(c)] for c in np.unique(y)]}")
    
    # Balancear si es necesario
    if balancear and len(X) > 0:
        balancer = DataBalancer(estrategia='oversample')
        print("\n" + balancer.justificar_estrategia())
        X, y = balancer.balancear(X, y)
    
    # Validar que hay suficientes datos para dividir
    if len(X) < 10:
        raise ValueError(
            f"❌ No hay suficientes datos para entrenar. Se encontraron {len(X)} textos.\n"
            "   Se requieren al menos 10 textos para dividir en train/val/test."
        )
    
    # Dividir en train/val/test
    # Si hay muy pocos datos, ajustar los tamaños
    if len(X) < 30:
        test_size = 0.1
        val_size = 0.1
        print(f"⚠️  Pocos datos detectados. Ajustando test_size a {test_size} y val_size a {val_size}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\n✅ División de datos:")
    print(f"   Train: {len(X_train)} textos")
    print(f"   Val:   {len(X_val)} textos")
    print(f"   Test:  {len(X_test)} textos")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    from database import DatabaseManager
    
    db_manager = DatabaseManager()
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(db_manager)
    
    print("\n✅ Preprocesamiento completado")

