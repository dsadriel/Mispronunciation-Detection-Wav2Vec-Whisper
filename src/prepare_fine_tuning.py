import json
import os
import pandas as pd
from src.data_loader import get_dataset, parse_textgrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def generate_vocab_and_split(data_dir: str, output_dir: str = "data"):
    print("Loading dataset inventory...")
    df = get_dataset(data_dir)
    
    # 1. Dataset Split (Stratified by speaker)
    print("Splitting dataset into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['speaker']
    )
    
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    print(f"Split complete: {len(train_df)} train, {len(test_df)} test.")
    
    # 2. Vocabulary Generation
    print("Generating phoneme vocabulary from training set...")
    phonemes = set()
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        annotations = parse_textgrid(row['ann_path'])
        for ann in annotations:
            # We use the 'produced' phoneme for training the acoustic model
            # Normalization might be needed later
            phone = ann['produced'].strip()
            if phone:
                phonemes.add(phone)
    
    # Add special tokens
    vocab = {p: i for i, p in enumerate(sorted(list(phonemes)))}
    
    # Ensure standard CTC tokens are present or mapped
    # | is often used as a word separator in Wav2vec 2.0
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
