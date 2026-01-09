"""
Módulo de entrenamiento del modelo LLM con fine-tuning
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
try:
    from datasets import Dataset
except ImportError:
    # Fallback si datasets no está instalado
    print("⚠️  datasets no está instalado. Instalando...")
    import subprocess
    import sys
    try:
        # Intentar con python -m pip (más confiable en Windows)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from datasets import Dataset
        print("✅ datasets instalado correctamente")
    except Exception as e:
        print(f"❌ Error instalando datasets: {e}")
        print("Por favor, instale manualmente ejecutando:")
        print("  python -m pip install datasets")
        raise ImportError("El módulo 'datasets' es requerido. Por favor, instálelo manualmente.")
import os
from config import (
    MODEL_NAME, MODEL_SAVE_PATH, MAX_LENGTH, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, CATEGORIAS
)


class TextClassifierTrainer:
    """Entrenador del clasificador de textos"""
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.num_labels = len(CATEGORIAS)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Función de pérdida personalizada para clasificación multiclase"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Usar CrossEntropyLoss para clasificación multiclase
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def tokenizar_datos(self, textos, max_length=MAX_LENGTH):
        """Tokeniza los textos"""
        return self.tokenizer(
            textos,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def preparar_dataset(self, textos, etiquetas):
        """Prepara el dataset para entrenamiento"""
        # Asegurar que las etiquetas sean enteros
        if isinstance(etiquetas, np.ndarray):
            etiquetas = etiquetas.astype(int)
        else:
            etiquetas = np.array(etiquetas, dtype=int)
        
        # Tokenizar
        encodings = self.tokenizer(
            textos.tolist() if isinstance(textos, np.ndarray) else textos,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )
        
        # Crear dataset de HuggingFace
        # Asegurar que las etiquetas sean una lista de enteros
        labels_list = etiquetas.tolist() if isinstance(etiquetas, np.ndarray) else list(etiquetas)
        labels_list = [int(label) for label in labels_list]  # Convertir a enteros explícitamente
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels_list
        })
        
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def entrenar(self, X_train, y_train, X_val, y_val):
        """Entrena el modelo"""
        print(f"\n🚀 Iniciando entrenamiento con {self.model_name}")
        print(f"   Número de clases: {self.num_labels}")
        print(f"   Datos de entrenamiento: {len(X_train)}")
        print(f"   Datos de validación: {len(X_val)}")
        
        # Preparar datasets
        train_dataset = self.preparar_dataset(X_train, y_train)
        val_dataset = self.preparar_dataset(X_val, y_val)
        
        # Cargar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Configurar el tipo de problema explícitamente
        if hasattr(self.model.config, 'problem_type'):
            self.model.config.problem_type = "single_label_classification"
        
        # Asegurar que use CrossEntropyLoss en lugar de BCE
        # Esto se hace automáticamente cuando num_labels > 2 y problem_type está configurado
        print(f"   Configuración del modelo:")
        print(f"   - num_labels: {self.model.config.num_labels}")
        print(f"   - problem_type: {getattr(self.model.config, 'problem_type', 'No configurado')}")
        
        # Asegurar que el directorio existe
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        log_dir = os.path.join(MODEL_SAVE_PATH, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar argumentos de entrenamiento
        # Deshabilitar accelerate si hay problemas de compatibilidad
        use_cpu = not torch.cuda.is_available()
        
        training_args = TrainingArguments(
            output_dir=MODEL_SAVE_PATH,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            logging_dir=log_dir,
            logging_steps=10,
            evaluation_strategy="epoch",  # Cambiado de eval_strategy a evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=False if use_cpu else torch.cuda.is_available(),  # Usar FP16 solo si hay GPU
            dataloader_pin_memory=False,  # Evitar problemas de memoria en Windows
            no_cuda=use_cpu,  # Forzar CPU si no hay GPU
            report_to=[],  # Deshabilitar TensorBoard para evitar problemas
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Verificar formato de etiquetas antes de crear el trainer
        print(f"\n🔍 Verificando formato de etiquetas:")
        print(f"   Tipo de y_train: {type(y_train)}")
        if isinstance(y_train, np.ndarray):
            print(f"   Dtype de y_train: {y_train.dtype}")
            print(f"   Shape de y_train: {y_train.shape}")
            print(f"   Valores únicos: {np.unique(y_train)}")
            print(f"   Primeros valores: {y_train[:5]}")
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Sobrescribir compute_loss para usar CrossEntropyLoss correctamente
        original_compute_loss = trainer.compute_loss
        def custom_compute_loss(model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Usar CrossEntropyLoss para clasificación multiclase
            loss_fct = nn.CrossEntropyLoss()
            # Asegurar que labels sean LongTensor
            if isinstance(labels, torch.Tensor):
                labels = labels.long()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        
        trainer.compute_loss = custom_compute_loss
        
        # Entrenar
        print("\n📚 Entrenando modelo...")
        trainer.train()
        
        # Guardar modelo
        print(f"\n💾 Guardando modelo en {MODEL_SAVE_PATH}")
        trainer.save_model()
        self.tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        # Evaluación final
        print("\n📊 Evaluando modelo final...")
        eval_results = trainer.evaluate()
        
        print("\n✅ Resultados de validación:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
        
        return trainer, eval_results
    
    def cargar_modelo(self, model_path=None):
        """Carga un modelo entrenado"""
        path = model_path or MODEL_SAVE_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado en {path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.eval()
        return self.model, self.tokenizer
    
    def predecir(self, textos, return_probs=False):
        """Realiza predicciones"""
        if self.model is None:
            raise ValueError("Modelo no cargado. Usa cargar_modelo() primero.")
        
        # Tokenizar
        encodings = self.tokenizer(
            textos if isinstance(textos, list) else [textos],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        if return_probs:
            return predictions.numpy(), probs.numpy()
        return predictions.numpy()


if __name__ == "__main__":
    from database import DatabaseManager
    from preprocessing import preparar_datos
    
    # Preparar datos
    print("📊 Preparando datos...")
    db_manager = DatabaseManager()
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(db_manager)
    
    # Entrenar modelo
    trainer = TextClassifierTrainer()
    trainer_model, eval_results = trainer.entrenar(X_train, y_train, X_val, y_val)
    
    print("\n✅ Entrenamiento completado!")

