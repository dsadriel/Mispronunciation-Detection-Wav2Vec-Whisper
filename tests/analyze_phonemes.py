
import pandas as pd
from src.data_loader import parse_textgrid
import os
from tqdm import tqdm
import json

def analyze_phones(csv_path):
    df = pd.read_csv(csv_path)
    all_phones = set()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.exists(row['ann_path']):
            try:
                ann = parse_textgrid(row['ann_path'])
                for a in ann:
                    all_phones.add(a['produced'])
            except:
                pass
    
    print("\nUnique phones in dataset:")
    print(sorted(list(all_phones)))
    
    with open("data/vocab.json", "r") as f:
        vocab = json.load(f)
    
    missing = [p for p in all_phones if p not in vocab]
    print("\nPhones missing from vocab.json:")
    print(missing)

if __name__ == "__main__":
    analyze_phones("data/train_split.csv")
