
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

MODEL_ID = "./wav2vec2-l2arctic-phonemes"

def run_inference(audio_path):
    # 1. Load processor and model
    print(f"Loading model from {MODEL_ID}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    # 2. Load and preprocess audio
    print(f"Loading audio: {audio_path}")
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

    # 3. Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # 4. Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

if __name__ == "__main__":
    import pandas as pd
    train_csv = "data/train_split.csv"
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        samples = df.sample(3)
        
        print(f"Loading model from {MODEL_ID}...")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

        for _, sample_row in samples.iterrows():
            audio_file = sample_row['wav_path']
            print(f"\n--- Testing on TRAINING sample: {audio_file} ---")
            speech, sr = librosa.load(audio_file, sr=16000)
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            print(f"Predicted Phonemes: '{transcription}'")
    else:
        print("Train CSV not found.")
