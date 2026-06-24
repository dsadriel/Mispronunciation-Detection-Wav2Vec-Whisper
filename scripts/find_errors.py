import pandas as pd
from src.utils.data_loader import parse_textgrid

df = pd.read_csv("data/test_split.csv")
errors = set()
for idx, row in df.iterrows():
    annotations = parse_textgrid(row['ann_path'])
    for ann in annotations:
        if ann['error_type']:
            errors.add(ann['error_type'])
print("Unique error types:", errors)
