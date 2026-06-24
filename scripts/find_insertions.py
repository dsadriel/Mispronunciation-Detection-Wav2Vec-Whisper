import pandas as pd
from src.utils.data_loader import parse_textgrid

df = pd.read_csv("data/test_split.csv")
count = 0
for idx, row in df.iterrows():
    annotations = parse_textgrid(row['ann_path'])
    for ann in annotations:
        if ann['error_type'] == 'addition':
            print(f"File: {row['file_id']}")
            print(f"Target: {ann['phone']}, Produced: {ann['produced']}, Error: {ann['error_type']}")
            count += 1
            if count >= 3:
                exit(0)
