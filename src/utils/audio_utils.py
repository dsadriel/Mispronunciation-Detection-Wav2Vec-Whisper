"""
Módulo utilitário para manipulação e processamento de arquivos de áudio.
Contém funções para carregar e reamostrar áudios, garantindo que os dados
estejam no formato correto para os modelos de reconhecimento de fonemas.
"""

import librosa
import soundfile as sf
import os
from typing import Tuple
import numpy as np

def load_and_resample(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Carrega um arquivo de áudio e o reamostra para a taxa de amostragem alvo.

    Parâmetros:
        file_path (str): Caminho para o arquivo de áudio a ser carregado.
        target_sr (int, opcional): Taxa de amostragem desejada. O padrão é 16000.

    Retorna:
        Tuple[np.ndarray, int]: Uma tupla contendo o sinal de áudio como um array numpy e a taxa de amostragem.
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def ensure_resampled_exists(file_path: str, output_dir: str, target_sr: int = 16000) -> str:
    """
    Verifica se uma versão reamostrada do arquivo já existe, caso contrário, cria uma nova.

    Parâmetros:
        file_path (str): Caminho do arquivo de áudio original.
        output_dir (str): Diretório base de saída onde o arquivo reamostrado será salvo.
        target_sr (int, opcional): Taxa de amostragem desejada para o novo arquivo. O padrão é 16000.

    Retorna:
        str: Caminho completo para o arquivo de áudio reamostrado.
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
    # Testar com um arquivo de amostra
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    output_base = "data/processed_audio"
    
    if os.path.exists(sample_wav):
        out_path = ensure_resampled_exists(sample_wav, output_base)
        print(f"Arquivo reamostrado salvo em: {out_path}")
    else:
        print(f"Arquivo não encontrado: {sample_wav}")
