import os
import pandas as pd
from tqdm import tqdm
import jiwer
import textgrid
import librosa
from src.predict.reference_generator import ReferenceGenerator
from src.predict.mispronunciation_detector import MispronunciationDetector
import re

def main():
    print("Loading test split...")
    df = pd.read_csv("data/test_split.csv")
    
    sample_df = df.sample(150, random_state=42)
    ref_gen = ReferenceGenerator("base")
    detector = MispronunciationDetector(load_whisper=True)
    
    ground_truths = []
    transcriptions = []
    
    print("Calculating Whisper WER...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        ann_path = row['ann_path']
        wav_path = row['wav_path']
        
        try:
            tg = textgrid.TextGrid.fromFile(ann_path)
            words_tier = next((t for t in tg if t.name == 'words'), None)
            if not words_tier: continue
            
            gt_words = [interval.mark.strip() for interval in words_tier if interval.mark.strip()]
            gt_text = " ".join(gt_words).lower()
            
            pred_text = ref_gen.transcribe(wav_path).lower()
            
            pred_text = re.sub(r'[^a-z\s]', '', pred_text)
            gt_text = re.sub(r'[^a-z\s]', '', gt_text)
            
            ground_truths.append(gt_text)
            transcriptions.append(pred_text)
        except Exception as e:
            print(f"Error {wav_path}: {e}")
            
    wer = jiwer.wer(ground_truths, transcriptions)
    print(f"\nWhisper Base WER (estimated on 150 samples): {wer:.4f} ({wer*100:.2f}%)")
    
    print("\nExtracting Insertion Examples...")
    count = 0
    for idx, row in df.head(50).iterrows():
        try:
            acoustic_pred, canonical_ref, transcription, aligned_pairs = detector.detect(row['wav_path'])
            insertions = [p for p in aligned_pairs if p['type'] == 'insertion']
            if len(insertions) > 0 and len(insertions) < 3: 
                print(f"\nExample found in {row['wav_path']}:")
                print(f"Transcription: {transcription}")
                print(f"Acoustic:  {acoustic_pred}")
                print(f"Canonical: {canonical_ref}")
                print("Insertions:", insertions)
                
                # We can format the string to show insertion clearly
                out = jiwer.process_words(canonical_ref, acoustic_pred)
                print(jiwer.visualize_alignment(out))
                
                count += 1
                if count >= 2:
                    break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
