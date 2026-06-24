import pandas as pd
from src.utils.data_loader import parse_textgrid

df = pd.read_csv("data/test_split.csv")
count = 0
for idx, row in df.iterrows():
    annotations = parse_textgrid(row['ann_path'])
    for ann in annotations:
        if ann['error_type'] == 'a':
            print(f"File: {row['file_id']}, Locutor: {row['speaker']}")
            print(f"Adição (Inserção) - Alvo (geralmente vazio/esp): '{ann['phone']}', Produzido: '{ann['produced']}'")
            count += 1
            if count >= 2:
                exit(0)
