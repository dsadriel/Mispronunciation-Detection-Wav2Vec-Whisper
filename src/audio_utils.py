import librosa
import soundfile as sf
import os
from typing import Tuple
import numpy as np

def load_and_resample(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples it to the target sampling rate.
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def ensure_resampled_exists(file_path: str, output_dir: str, target_sr: int = 16000) -> str:
    """
    Checks if a resampled version of the file exists, otherwise creates it.
    Returns the path to the resampled file.
    """
    base_name = os.path.basename(file_path)
    speaker_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    
    target_dir = os.path.join(output_dir, speaker_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    output_path = os.path.join(target_dir, base_name)
    
    if not os.path.exists(output_path):
        audio, sr = load_and_resample(file_path, target_sr)
        sf.write(output_path, audio, target_sr)
        
    return output_path

if __name__ == "__main__":
    # Test with a sample file
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    output_base = "data/processed_audio"
    
    if os.path.exists(sample_wav):
        out_path = ensure_resampled_exists(sample_wav, output_base)
        print(f"Resampled file saved at: {out_path}")
    else:
        print(f"File not found: {sample_wav}")
