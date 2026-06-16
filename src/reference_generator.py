## Reference Generator
# This module defines the ReferenceGenerator class, which provides functionality to:
# 1. Transcribe audio files into text using OpenAI's Whisper model.
# 2. Convert the transcribed text into Arpabet phonemes using the g2p_en library.

import whisper
from g2p_en import G2p
import os
import re
from typing import List
import ssl
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

def simplify_phone(p):
    if not p:
        return None
    # 1. Remover números (acentos tônicos)
    p = re.sub(r'\d+', '', p)
    # 2. Remover símbolos especiais
    p = p.replace('*', '').replace(')', '').replace('_', '').replace('`', '').replace('(', '')
    # 3. Padronizar para maiúsculas
    p = p.upper()
    # 4. Ignorar silêncios, espaços e lixo
    if p in ['SIL', 'SP', 'SPN', ' ', '']:
        return None
    return p

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
        """Converts text to simplified Arpabet phonemes using g2p_en."""
        raw_phonemes = self.g2p(text)
        simplified = [simplify_phone(p) for p in raw_phonemes]
        return [p for p in simplified if p]

    def generate(self, audio_path: str) -> str:
        """Full pipeline: Audio -> Text -> Simplified Phoneme String."""
        text = self.transcribe(audio_path)
        phonemes = self.to_phonemes(text)
        return " ".join(phonemes)

if __name__ == "__main__":
    import sys
    from src.data_loader import parse_textgrid

    print("Testing ReferenceGenerator with a sample audio file...")
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    ann_path = "data/l2arctic_release_v5.0/ABA/annotation/arctic_a0003.TextGrid"
    
    if os.path.exists(sample_wav):
        generator = ReferenceGenerator()
        
        print(f"\nProcessing: {sample_wav}")
        text = generator.transcribe(sample_wav)
        print(f"Whisper Transcription: '{text}'")
        
        canonical_phonemes = generator.generate(sample_wav)
        print(f"Canonical Phonemes: '{canonical_phonemes}'")
        
        if os.path.exists(ann_path):
            gt_ann = parse_textgrid(ann_path)
            # Use 'phone' (target) for ground truth comparison
            gt_phones = [simplify_phone(ann['phone']) for ann in gt_ann]
            gt_phones_str = " ".join([p for p in gt_phones if p])
            print(f"Target Ground Truth: '{gt_phones_str}'")
    else:
        print(f"File not found: {sample_wav}")
