"""
Módulo para realizar inferência acústica de fonemas.

Este script carrega o modelo Wav2Vec2 ajustado e processa arquivos de áudio
para extrair sequências de fonemas em formato Arpabet simplificado.
Também fornece funções auxiliares para tratar e simplificar os rótulos de fonemas.
"""

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import pandas as pd
import re
from src.utils.data_loader import parse_textgrid

MODEL_DIR = "./wav2vec2-l2arctic-phonemes"

def simplify_phone(p):
    """
    Simplifica uma string de fonema (Arpabet).
    
    Remove os acentos tônicos (números), limpa símbolos especiais, padroniza
    para maiúsculas e ignora silêncios/marcadores indesejados.

    Args:
        p (str): A string do fonema original.

    Returns:
        str ou None: O fonema simplificado, ou None se for vazio/silêncio.
    """
    if not p:
        return None
    # 1. Remover números (acentos tônicos)
    p = re.sub(r'\d+', '', p)
    # 2. Remover símbolos especiais
    p = p.replace('*', '').replace(')', '').replace('_', '').replace('`', '').replace('(', '')
    # 3. Padronizar para maiúsculas
    p = p.upper()
    # 4. Ignorar silêncios e marcadores inválidos
    if p in ['SIL', 'SP', 'SPN', '']:
        return None
    return p

def get_latest_checkpoint(base_dir):
    """
    Obtém o diretório do checkpoint mais recente do modelo salvo.

    Args:
        base_dir (str): O diretório base onde os checkpoints estão salvos.

    Returns:
        str ou None: O caminho para o último checkpoint encontrado, ou None.
    """
    if not os.path.exists(base_dir):
        return None
    checkpoints = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return base_dir
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

def load_model_and_processor(model_id):
    """
    Carrega o modelo Wav2Vec2 e seu processador a partir do diretório informado.

    Args:
        model_id (str): O caminho ou identificador do modelo.

    Returns:
        tuple: (processor, model) O processador e o modelo Wav2Vec2 carregados.
    """
    print(f"Carregando processador e modelo de {model_id}...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    return processor, model

def run_inference(audio_path, processor, model, device="cpu"):
    """
    Executa a inferência no arquivo de áudio utilizando o modelo Wav2Vec2 para CTC.

    Args:
        audio_path (str): O caminho do arquivo de áudio a ser inferido.
        processor (Wav2Vec2Processor): O processador de áudio do modelo.
        model (Wav2Vec2ForCTC): O modelo Wav2Vec2 ajustado.
        device (str, opcional): O dispositivo no qual rodar a inferência (e.g. 'cpu', 'cuda', 'mps').

    Returns:
        str: A transcrição fonética extraída do áudio.
    """
    # Carregar e pré-processar o áudio
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    # Mover para o dispositivo
    input_values = input_values.to(device)
    model.to(device)

    # Inferência
    with torch.no_grad():
        logits = model(input_values).logits

    # Decodificar
    predicted_ids = torch.argmax(logits, dim=-1)
    # O token | é usado como separador, então nós o substituímos por espaço para melhor legibilidade
    transcription = processor.batch_decode(predicted_ids)[0].replace("|", " ")
    
    return transcription

def get_ground_truth(ann_path):
    """
    Obtém a transcrição de referência real (ground truth) a partir do TextGrid.

    Args:
        ann_path (str): O caminho para o arquivo de anotação (TextGrid).

    Returns:
        str: Sequência de fonemas correspondente à produção real do falante.
    """
    annotations = parse_textgrid(ann_path)
    # O modelo foi treinado em rótulos 'produced' simplificados
    simplified = [simplify_phone(ann['produced']) for ann in annotations]
    return " ".join([p for p in simplified if p])

if __name__ == "__main__":
    MODEL_ID = get_latest_checkpoint(MODEL_DIR)
    if not MODEL_ID or not os.path.exists(MODEL_ID):
        print(f"Modelo não encontrado em {MODEL_DIR}")
        exit(1)

    print(f"Usando pesos do modelo de: {MODEL_ID}")
    processor, model = load_model_and_processor(MODEL_ID)

    test_csv = "data/test_split.csv"
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        # Selecionar algumas amostras do conjunto de teste
        samples = df.sample(min(5, len(df)))
        
        for _, sample_row in samples.iterrows():
            audio_file = sample_row['wav_path']
            ann_file = sample_row['ann_path']
            
            print(f"\n--- Testando amostra: {sample_row['file_id']} (Falante: {sample_row['speaker']}) ---")
            
            prediction = run_inference(audio_file, processor, model)
            ground_truth = get_ground_truth(ann_file)
            
            print(f"Fonemas Preditos:\n\t'{prediction}'")
            print(f"Verdade Terrestre (Produzido):\n\t'{ground_truth}'")
    else:
        print("CSV de teste não encontrado. Por favor, execute src/prepare_fine_tuning.py primeiro.")
