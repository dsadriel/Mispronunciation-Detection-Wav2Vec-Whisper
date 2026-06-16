import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import pandas as pd
import re
from src.data_loader import parse_textgrid

MODEL_DIR = "./wav2vec2-l2arctic-phonemes"

def simplify_phone(p):
    if not p:
        return None
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

def get_latest_checkpoint(base_dir):
    if not os.path.exists(base_dir):
        return None
    checkpoints = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return base_dir
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

def load_model_and_processor(model_id):
    print(f"Loading processor and model from {model_id}...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    return processor, model

def run_inference(audio_path, processor, model, device="cpu"):
    # Load and preprocess audio
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    # Move to device
    input_values = input_values.to(device)
    model.to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    # The | token is used as a separator, so we replace it with space for readability
    transcription = processor.batch_decode(predicted_ids)[0].replace("|", " ")
    
    return transcription

def get_ground_truth(ann_path):
    annotations = parse_textgrid(ann_path)
    # The model was trained on simplified 'produced' labels
    simplified = [simplify_phone(ann['produced']) for ann in annotations]
    return " ".join([p for p in simplified if p])

if __name__ == "__main__":
    MODEL_ID = get_latest_checkpoint(MODEL_DIR)
    if not MODEL_ID or not os.path.exists(MODEL_ID):
        print(f"Model not found at {MODEL_DIR}")
        exit(1)

    print(f"Using model weights from: {MODEL_ID}")
    processor, model = load_model_and_processor(MODEL_ID)

    test_csv = "data/test_split.csv"
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        # Select some samples from the test set
        samples = df.sample(min(5, len(df)))
        
        for _, sample_row in samples.iterrows():
            audio_file = sample_row['wav_path']
            ann_file = sample_row['ann_path']
            
            print(f"\n--- Testing sample: {sample_row['file_id']} (Speaker: {sample_row['speaker']}) ---")
            
            prediction = run_inference(audio_file, processor, model)
            ground_truth = get_ground_truth(ann_file)
            
            print(f"Predicted Phonemes:\n\t'{prediction}'")
            print(f"Ground Truth (Produced):\n\t'{ground_truth}'")
    else:
        print("Test CSV not found. Please run src/prepare_fine_tuning.py first.")
