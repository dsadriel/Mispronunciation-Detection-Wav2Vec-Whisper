
import pandas as pd
import librosa
from src.data_loader import parse_textgrid
import os
from tqdm import tqdm

def check_dataset(csv_path):
    df = pd.read_csv(csv_path)
    counts = {'short': 0, 'empty': 0, 'ok': 0}
    bad_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            ann = parse_textgrid(row['ann_path'])
            phones = [a['produced'] for a in ann if a['produced'] not in ['sil', 'sp']]
            if not phones:
                counts['empty'] += 1
                bad_files.append((row['wav_path'], "empty"))
                continue
            
            speech, _ = librosa.load(row['wav_path'], sr=16000)
            # Wav2Vec2 downsamples by 320
            n_frames = len(speech) // 320
            
            if n_frames < len(phones):
                counts['short'] += 1
                bad_files.append((row['wav_path'], f"short: {n_frames} frames vs {len(phones)} phones"))
            else:
                counts['ok'] += 1
        except Exception as e:
            print(f"Error processing {row['wav_path']}: {e}")
            
    print(f"\nSummary for {csv_path}:")
    print(counts)
    if bad_files:
        print("\nFirst 5 bad files:")
        for f in bad_files[:5]:
            print(f)

if __name__ == "__main__":
    check_dataset("data/train_split.csv")
