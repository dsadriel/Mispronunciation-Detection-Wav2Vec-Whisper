"""
Script principal de treinamento.
Configura e executa o fine-tuning de um modelo acústico (Wav2Vec 2.0) utilizando 
a biblioteca Transformers da Hugging Face, com o objetivo de realizar o reconhecimento 
de fonemas através da perda CTC (Connectionist Temporal Classification).
"""

import os
# Define o modo de fallback para operações MPS não implementadas (como ctc_loss)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import pandas as pd
import torch
import librosa
import numpy as np
import evaluate
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from src.utils.data_loader import parse_textgrid

# 1. Configurações e Caminhos
VOCAB_FILE = "data/vocab.json"
TRAIN_CSV = "data/train_split.csv"
TEST_CSV = "data/test_split.csv"
OUTPUT_DIR = "./wav2vec2-l2arctic-phonemes"

import re

def simplify_phone(p):
    """
    Simplifica um fonema removendo acentos tônicos (números), símbolos especiais,
    padronizando para letras maiúsculas e ignorando marcações de silêncio/ruído.
    
    Args:
        p (str): O fonema original a ser simplificado.
        
    Returns:
        str ou None: O fonema simplificado, ou None se for um silêncio/ruído ou vazio.
    """
    # 1. Remover números (acentos tônicos)
    p = re.sub(r'\d+', '', p)
    # 2. Remover símbolos especiais
    p = p.replace('*', '').replace(')', '').replace('_', '').replace('`', '').replace('(', '')
    # 3. Padronizar para maiúsculas
    p = p.upper()
    # 4. Ignorar silêncios e lixo
    if p in ['SIL', 'SP', 'SPN', '']:
        return None
    return p

def load_data(csv_path):
    """
    Carrega os dados a partir de um arquivo CSV, processa as anotações do TextGrid 
    e formata o conjunto de dados em um formato compatível para treinamento, combinando 
    caminhos de áudio com os textos alvos (fonemas separados pelo caractere '|').
    
    Args:
        csv_path (str): Caminho para o arquivo CSV de entrada.
        
    Returns:
        Dataset: Um objeto Dataset do Hugging Face contendo os dados processados.
    """
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        try:
            annotations = parse_textgrid(row['ann_path'])
            # Extrair e simplificar os fonemas
            produced_phones = []
            for ann in annotations:
                simple = simplify_phone(ann['produced'])
                if simple:
                    produced_phones.append(simple)
            
            if not produced_phones:
                continue

            target_text = "|".join(produced_phones)
            data.append({
                "audio": row['wav_path'],
                "target_text": target_text
            })
        except Exception as e:
            print(f"Error reading {row['ann_path']}: {e}")
            continue
    return Dataset.from_list(data)

# 2. Data Collator para CTC
@dataclass
class DataCollatorCTCWithPadding:
    """
    Agrupador de dados (Data Collator) customizado para modelos baseados em CTC.
    Ele preenche (pad) dinamicamente os arrays de áudio e as sequências de texto alvo
    de modo que possam ser processados em lotes de tamanhos variáveis.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Processa e preenche uma lista de features para construir um lote (batch).
        
        Args:
            features (List[Dict]): Lista de features contendo os inputs do áudio e os rótulos.
            
        Returns:
            Dict[str, torch.Tensor]: O dicionário processado e empacotado como tensores PyTorch.
        """
        # Separar entradas e rótulos já que eles têm comprimentos diferentes 
        # e precisam de métodos de padding diferentes
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Substituir os preenchimentos (pad) por -100 para ignorá-los corretamente no cálculo da perda (loss)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    """
    Função de pré-processamento que carrega o arquivo de áudio, extrai suas features usando 
    o processador e converte os fonemas-alvo em identificadores (labels).
    
    Args:
        batch (Dict): Um dicionário contendo os caminhos do 'audio' e do 'target_text'.
        processor (Wav2Vec2Processor): O processador a ser utilizado.
        
    Returns:
        Dict: O batch contendo as 'input_values' (áudio carregado e processado) e os 'labels'.
    """
    audio = batch["audio"]
    # Carregar e reamostrar
    speech, sr = librosa.load(audio, sr=16000)
    batch["input_values"] = processor(speech, sampling_rate=sr).input_values[0]
    
    batch["labels"] = processor(text=batch["target_text"]).input_ids
    return batch

def main():
    """
    Função principal que gerencia o fluxo de carregamento dos dados, processamento,
    inicialização do modelo, configurações de métricas (WER) e treinamento (fine-tuning)
    do modelo Wav2Vec 2.0.
    """
    # 3. Carregar o Processador
    tokenizer = Wav2Vec2CTCTokenizer(VOCAB_FILE, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 4. Carregar Datasets
    print("Loading datasets...")
    train_dataset = load_data(TRAIN_CSV)
    test_dataset = load_data(TEST_CSV)

    print("Preprocessing datasets (this might take a while)...")
    # Mapear o pré-processamento usando múltiplos núcleos de CPU para maior velocidade
    num_cores = 8 
    train_dataset = train_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=train_dataset.column_names, num_proc=num_cores)
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=test_dataset.column_names, num_proc=num_cores)

    # 5. Carregar o Modelo
    with open(VOCAB_FILE, "r") as f:
        vocab = json.load(f)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # Congelar o codificador de features
    model.freeze_feature_encoder()

    # 6. Argumentos de Treinamento
    training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      per_device_train_batch_size=16,
      gradient_accumulation_steps=2,
      dataloader_num_workers=4,
      eval_strategy="steps",
      num_train_epochs=60, # Aumentado para ~3 noites de treino
      fp16=False, 
      save_steps=100,
      eval_steps=100,
      logging_steps=25,
      learning_rate=5e-5, # Resetado para permitir progresso além do checkpoint atual
      max_grad_norm=0.5,
      weight_decay=0.005,
      warmup_steps=500,
      save_total_limit=10, # Mantém mais backups por segurança
    )

    # 7. Inicializar o Trainer
    print("Initializing Data Collator...")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    print("Loading WER metric...")
    try:
        wer_metric = evaluate.load("wer")
        print("WER metric loaded successfully.")
    except Exception as e:
        print(f"Error loading WER metric: {e}")
        print("Falling back to a dummy metric or skipping WER...")
        wer_metric = None

    def compute_metrics(pred):
        """
        Calcula as métricas de avaliação (WER - Word Error Rate) a partir das predições.
        
        Args:
            pred: Objeto contendo as predições e os rótulos reais.
            
        Returns:
            Dict: Um dicionário com o valor de WER computado.
        """
        if wer_metric is None:
            return {"wer": 0.0}
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
    )
    print("Trainer initialized.")

    # 8. Treinamento
    print("Starting training...")
    # Verificar se um checkpoint existe para ser retomado
    last_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"Resuming training from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 9. Salvar o modelo final e o processador
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    # Checar se os arquivos existem antes de rodar
    if os.path.exists(TRAIN_CSV) and os.path.exists(VOCAB_FILE):
        main()
    else:
        print("Required files (CSV or Vocab) not found. Run prepare_fine_tuning.py first.")
