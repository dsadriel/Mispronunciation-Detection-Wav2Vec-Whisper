
import pandas as pd
import json
import re
import os
from src.data_loader import parse_textgrid
from tqdm import tqdm

def simplify_phone(p):
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
