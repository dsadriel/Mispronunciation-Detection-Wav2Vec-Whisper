import os
# Set fallback for MPS ops not implemented (like ctc_loss)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import pandas as pd
import torch
import librosa
import numpy as np
import evaluate
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from src.data_loader import parse_textgrid

# 1. Configuration and Paths
VOCAB_FILE = "data/vocab.json"
TRAIN_CSV = "data/train_split.csv"
TEST_CSV = "data/test_split.csv"
OUTPUT_DIR = "./wav2vec2-l2arctic-phonemes"

import re

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

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        try:
            annotations = parse_textgrid(row['ann_path'])
            # Extrair e simplificar os fonemas
            produced_phones = []
            for ann in annotations:
                simple = simplify_phone(ann['produced'])
                if simple:
                    produced_phones.append(simple)
            
            if not produced_phones:
                continue

            target_text = "|".join(produced_phones)
            data.append({
                "audio": row['wav_path'],
                "target_text": target_text
            })
        except Exception as e:
            print(f"Error reading {row['ann_path']}: {e}")
            continue
    return Dataset.from_list(data)

# 2. Data Collator for CTC
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    # Load and resample
    speech, sr = librosa.load(audio, sr=16000)
    batch["input_values"] = processor(speech, sampling_rate=sr).input_values[0]
    
    batch["labels"] = processor(text=batch["target_text"]).input_ids
    return batch

def main():
    # 3. Load Processor
    tokenizer = Wav2Vec2CTCTokenizer(VOCAB_FILE, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 4. Load Datasets
    print("Loading datasets...")
    train_dataset = load_data(TRAIN_CSV)
    test_dataset = load_data(TEST_CSV)

    print("Preprocessing datasets (this might take a while)...")
    # Map preprocessing using multiple CPU cores for speed
    num_cores = 8 
    train_dataset = train_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=train_dataset.column_names, num_proc=num_cores)
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=test_dataset.column_names, num_proc=num_cores)

    # 5. Load Model
    with open(VOCAB_FILE, "r") as f:
        vocab = json.load(f)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # Freeze feature extractor
    model.freeze_feature_encoder()

    # 6. Training Arguments
    training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      per_device_train_batch_size=16,
      gradient_accumulation_steps=2,
      dataloader_num_workers=4,
      eval_strategy="steps",
      num_train_epochs=60, # Aumentado para ~3 noites de treino
      fp16=False, 
      save_steps=100,
      eval_steps=100,
      logging_steps=25,
      learning_rate=5e-5, # Resetado para permitir progresso além do checkpoint atual
      max_grad_norm=0.5,
      weight_decay=0.005,
      warmup_steps=500,
      save_total_limit=10, # Mantém mais backups por segurança
    )

    # 7. Initialize Trainer
    print("Initializing Data Collator...")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    print("Loading WER metric...")
    try:
        wer_metric = evaluate.load("wer")
        print("WER metric loaded successfully.")
    except Exception as e:
        print(f"Error loading WER metric: {e}")
        print("Falling back to a dummy metric or skipping WER...")
        wer_metric = None

    def compute_metrics(pred):
        if wer_metric is None:
            return {"wer": 0.0}
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
    )
    print("Trainer initialized.")

    # 8. Train
    print("Starting training...")
    # Check if a checkpoint exists to resume from
    last_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            print(f"Resuming training from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 9. Save final model and processor
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    # Check if files exist before running
    if os.path.exists(TRAIN_CSV) and os.path.exists(VOCAB_FILE):
        main()
    else:
        print("Required files (CSV or Vocab) not found. Run prepare_fine_tuning.py first.")
