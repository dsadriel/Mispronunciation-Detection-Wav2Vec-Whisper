## Reference Generator
# This module defines the ReferenceGenerator class, which provides functionality to:
# 1. Transcribe audio files into text using OpenAI's Whisper model.
# 2. Convert the transcribed text into Arpabet phonemes using the g2p_en library.

import whisper
from g2p_en import G2p
import os
from typing import List
import ssl
import urllib.request
import nltk

# Bypass SSL verification for model and nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Pre-download NLTK resources needed by g2p_en
nltk.download('cmudict', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class ReferenceGenerator:
    def __init__(self, whisper_model_size: str = "base"):
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.g2p = G2p()
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribes audio using Whisper."""
        result = self.whisper_model.transcribe(audio_path)
        return result['text'].strip()
    
    def to_phonemes(self, text: str) -> List[str]:
        """Converts text to Arpabet phonemes using g2p_en."""
        phonemes = self.g2p(text)
        return [p for p in phonemes if p.strip()]

    def generate(self, audio_path: str) -> List[str]:
        """Full pipeline: Audio -> Text -> Phonemes."""
        text = self.transcribe(audio_path)
        phonemes = self.to_phonemes(text)
        return phonemes

if __name__ == "__main__":
    print("Testing ReferenceGenerator with a sample audio file (arctic_a0003.wav)...")
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    
    if os.path.exists(sample_wav):
        generator = ReferenceGenerator()
        
        print(f"\nProcessing: {sample_wav}")
        text = generator.transcribe(sample_wav)
        print(f"Transcription: {text}")
        
        phonemes = generator.to_phonemes(text)
        print(f"Phonemes: {' '.join(phonemes)}")
        
        from data_loader import parse_textgrid
        ann_path = "data/l2arctic_release_v5.0/ABA/annotation/arctic_a0003.TextGrid"
        if os.path.exists(ann_path):
            gt_ann = parse_textgrid(ann_path)
            gt_phones = [ann['phone'] for ann in gt_ann if ann['phone'] not in ['sil', 'sp']]
            print(f"Ground Truth: {' '.join(gt_phones)}")
    else:
        print(f"File not found: {sample_wav}")
