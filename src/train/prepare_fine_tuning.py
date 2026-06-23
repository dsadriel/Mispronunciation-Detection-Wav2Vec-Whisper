"""
Script responsável por preparar os dados para o fine-tuning do modelo.
Realiza a divisão do conjunto de dados em treino e teste (estratificado por locutor)
e gera o vocabulário de fonemas a partir das anotações do conjunto de treino.
"""

import json
import os
import pandas as pd
from src.utils.data_loader import get_dataset, parse_textgrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def generate_vocab_and_split(data_dir: str, output_dir: str = "data"):
    """
    Gera o vocabulário de fonemas e divide o dataset em conjuntos de treino e teste.
    
    Args:
        data_dir (str): Caminho para o diretório raiz do dataset.
        output_dir (str): Caminho para o diretório onde os arquivos de saída serão salvos (padrão: "data").
        
    Returns:
        dict: O vocabulário de fonemas gerado contendo os mapeamentos de fonema para índice.
    """
    print("Loading dataset inventory...")
    df = get_dataset(data_dir)
    
    # 1. Divisão do dataset (Estratificada por locutor)
    print("Splitting dataset into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['speaker']
    )
    
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    print(f"Split complete: {len(train_df)} train, {len(test_df)} test.")
    
    # 2. Geração do Vocabulário
    print("Generating phoneme vocabulary from training set...")
    phonemes = set()
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        annotations = parse_textgrid(row['ann_path'])
        for ann in annotations:
            # Utilizamos o fonema 'produzido' para treinar o modelo acústico
            # Uma normalização pode ser necessária posteriormente
            phone = ann['produced'].strip()
            if phone:
                phonemes.add(phone)
    
    # Adicionar tokens especiais
    vocab = {p: i for i, p in enumerate(sorted(list(phonemes)))}
    
    # Garantir que tokens padrão CTC estejam presentes ou mapeados
    # | é frequentemente utilizado como separador de palavras no Wav2vec 2.0
    if '|' not in vocab:
        vocab['|'] = len(vocab)
    if '[PAD]' not in vocab:
        vocab['[PAD]'] = len(vocab)
    if '[UNK]' not in vocab:
        vocab['[UNK]'] = len(vocab)
        
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=4)
        
    print(f"Vocabulary saved with {len(vocab)} tokens.")
    return vocab

if __name__ == "__main__":
    data_path = "data/l2arctic_release_v5.0"
    if os.path.exists(data_path):
        generate_vocab_and_split(data_path)
    else:
        print(f"Data directory not found: {data_path}")
