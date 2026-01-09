"""
Módulo para gestión de base de datos SQLite
Migra los datos de Excel a una base de datos estructurada
"""
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from config import DATABASE_PATH, BASE_DIR

Base = declarative_base()


class TextoClasico(Base):
    """Modelo de tabla para textos clásicos"""
    __tablename__ = 'textos_clasicos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    texto = Column(Text, nullable=False)
    categoria = Column(String(100), nullable=False)
    archivo_origen = Column(String(50))
    confianza_original = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<TextoClasico(id={self.id}, categoria='{self.categoria}')>"


class DatabaseManager:
    """Gestor de base de datos"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def importar_excel(self, archivo_excel):
        """
        Importa datos desde un archivo Excel a la base de datos
        """
        try:
            # Leer Excel - intentar diferentes hojas
            df = None
            for sheet_name in [0, 1, 2, 'Sheet1', 'Sheet2', 'Sheet3']:
                try:
                    df = pd.read_excel(archivo_excel, sheet_name=sheet_name)
                    if df is not None and not df.empty:
                        break
                except:
                    continue
            
            if df is None or df.empty:
                print(f"⚠️  No se pudo leer {archivo_excel}")
                return 0
            
            # Detectar columnas de texto y categoría
            # Asumimos que la última columna es la categoría y las anteriores son texto
            texto_cols = []
            categoria_col = None
            
            # Buscar columna de categoría (última o con nombre relacionado)
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['categoria', 'category', 'clase', 'label', 'etiqueta']):
                    categoria_col = col
                else:
                    texto_cols.append(col)
            
            # Si no encontramos categoría explícita, usar última columna
            if categoria_col is None and len(df.columns) > 1:
                categoria_col = df.columns[-1]
                texto_cols = df.columns[:-1].tolist()
            elif categoria_col is None:
                # Solo una columna - asumir que es texto sin categoría
                texto_cols = df.columns.tolist()
                print(f"⚠️  {archivo_excel}: No se encontró columna de categoría")
                return 0
            
            # Combinar columnas de texto
            if len(texto_cols) > 1:
                df['texto_combinado'] = df[texto_cols].apply(
                    lambda x: ' '.join(x.astype(str).dropna()), axis=1
                )
                texto_col = 'texto_combinado'
            else:
                texto_col = texto_cols[0] if texto_cols else df.columns[0]
            
            # Limpiar y normalizar categorías
            df['categoria_limpia'] = df[categoria_col].astype(str).str.strip()
            
            # Mapear categorías a las estándar
            categoria_mapping = {
                'areté': 'Areté',
                'arete': 'Areté',
                'areté': 'Areté',
                'poder y política': 'Poder y Política',
                'poder y politica': 'Poder y Política',
                'poder y política': 'Poder y Política',
                'relación entre humanos y dioses': 'Relación entre Humanos y Dioses',
                'relacion entre humanos y dioses': 'Relación entre Humanos y Dioses',
                'relación entre humanos y dioses': 'Relación entre Humanos y Dioses',
            }
            
            # Normalizar categorías
            def normalizar_categoria(cat):
                if pd.isna(cat) or str(cat).strip() == '':
                    return None
                cat_lower = str(cat).strip().lower()
                # Buscar coincidencias parciales
                for key, value in categoria_mapping.items():
                    if key in cat_lower or cat_lower in key:
                        return value
                # Si contiene palabras clave
                if 'aret' in cat_lower or 'virtud' in cat_lower or 'excelencia' in cat_lower:
                    return 'Areté'
                elif 'poder' in cat_lower and ('politic' in cat_lower or 'gobierno' in cat_lower):
                    return 'Poder y Política'
                elif 'dios' in cat_lower or 'divin' in cat_lower or 'humano' in cat_lower:
                    return 'Relación entre Humanos y Dioses'
                return cat.title()  # Capitalizar como fallback
            
            df['categoria_limpia'] = df['categoria_limpia'].apply(normalizar_categoria)
            
            # Insertar en base de datos
            registros_insertados = 0
            textos_rechazados = 0
            
            for _, row in df.iterrows():
                texto = str(row[texto_col]).strip()
                categoria = row['categoria_limpia']
                
                # Validar texto
                if not texto or texto == 'nan' or texto == '' or len(texto) < 10:
                    textos_rechazados += 1
                    continue
                
                # Validar categoría
                if pd.isna(categoria) or not categoria or str(categoria).strip() == '' or str(categoria).strip() == 'nan':
                    textos_rechazados += 1
                    continue
                
                categoria = str(categoria).strip()
                
                # Solo insertar si la categoría es una de las esperadas
                categorias_validas = ['Areté', 'Poder y Política', 'Relación entre Humanos y Dioses']
                if categoria not in categorias_validas:
                    textos_rechazados += 1
                    continue
                
                texto_obj = TextoClasico(
                    texto=texto,
                    categoria=categoria,
                    archivo_origen=os.path.basename(archivo_excel)
                )
                self.session.add(texto_obj)
                registros_insertados += 1
            
            if textos_rechazados > 0:
                print(f"   ⚠️  {textos_rechazados} textos rechazados (texto vacío o categoría inválida)")
            
            self.session.commit()
            print(f"✅ {archivo_excel}: {registros_insertados} registros insertados")
            return registros_insertados
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error importando {archivo_excel}: {e}")
            return 0
    
    def importar_todos_excel(self):
        """Importa todos los archivos Excel del directorio base"""
        archivos = [f'{i}.xlsx' for i in range(7)]
        total = 0
        
        for archivo in archivos:
            archivo_path = os.path.join(BASE_DIR, archivo)
            if os.path.exists(archivo_path):
                total += self.importar_excel(archivo_path)
        
        return total
    
    def obtener_datos(self, categoria=None):
        """Obtiene datos de la base de datos"""
        query = self.session.query(TextoClasico)
        if categoria:
            query = query.filter(TextoClasico.categoria == categoria)
        return query.all()
    
    def obtener_dataframe(self):
        """Obtiene todos los datos como DataFrame"""
        query = "SELECT texto, categoria FROM textos_clasicos"
        return pd.read_sql(query, self.engine)
    
    def estadisticas(self):
        """Muestra estadísticas de la base de datos"""
        df = self.obtener_dataframe()
        if df.empty:
            print("⚠️  La base de datos está vacía")
            return
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE LA BASE DE DATOS")
        print("="*60)
        print(f"Total de registros: {len(df)}")
        print(f"\nDistribución por categoría:")
        print(df['categoria'].value_counts())
        print(f"\nPorcentajes:")
        print(df['categoria'].value_counts(normalize=True) * 100)
        print("="*60)
    
    def cerrar(self):
        """Cierra la sesión"""
        self.session.close()


if __name__ == "__main__":
    # Crear base de datos e importar datos
    db_manager = DatabaseManager()
    
    print("📊 Importando datos desde archivos Excel...")
    total = db_manager.importar_todos_excel()
    print(f"\n✅ Total de registros importados: {total}")
    
    db_manager.estadisticas()
    db_manager.cerrar()



