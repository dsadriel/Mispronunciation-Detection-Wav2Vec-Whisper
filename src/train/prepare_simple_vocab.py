"""
Script responsável por gerar um vocabulário simplificado de fonemas para o treinamento.
Ele extrai os fonemas do conjunto de treino, remove numerais e caracteres especiais, e gera 
um arquivo JSON contendo o mapeamento desses fonemas para IDs, adicionando tokens especiais.
"""

import pandas as pd
import json
import re
import os
from src.utils.data_loader import parse_textgrid
from tqdm import tqdm

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

def main():
    """
    Função principal que lê o conjunto de dados de treino, extrai as anotações
    de cada arquivo de áudio, simplifica os fonemas e constrói um novo vocabulário
    salvando o resultado em um arquivo JSON.
    """
    df = pd.read_csv("data/train_split.csv")
    unique_phones = set()
    
    print("Analisando fonemas no dataset...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.exists(row['ann_path']):
            try:
                ann = parse_textgrid(row['ann_path'])
                for a in ann:
                    simple = simplify_phone(a['produced'])
                    if simple:
                        unique_phones.add(simple)
            except:
                continue
    
    # Criar novo vocabulário
    sorted_phones = sorted(list(unique_phones))
    vocab = {phone: i for i, phone in enumerate(sorted_phones)}
    
    # Adicionar tokens especiais
    idx = len(vocab)
    vocab["|"] = idx
    vocab["[PAD]"] = idx + 1
    vocab["[UNK]"] = idx + 2
    
    with open("data/vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)
    
    print(f"\nNovo vocabulário simplificado criado com {len(vocab)} tokens.")
    print("Fonemas inclusos:", sorted_phones)

if __name__ == "__main__":
    main()
