"""
Módulo para a geração da pronúncia de referência (fonemas canônicos).

Este script define a classe ReferenceGenerator, que implementa a funcionalidade para:
1. Transcrever arquivos de áudio para texto usando o modelo Whisper da OpenAI.
2. Converter o texto transcrito em fonemas Arpabet usando a biblioteca g2p_en.
"""

import whisper
from g2p_en import G2p
import os
import re
from typing import List
import ssl
import nltk

# Contornar a verificação de SSL para downloads do modelo e do nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Fazer o pré-download dos recursos do NLTK necessários pelo g2p_en
nltk.download('cmudict', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def simplify_phone(p):
    """
    Simplifica uma string de fonema (Arpabet).

    Args:
        p (str): Fonema de entrada a ser simplificado.

    Returns:
        str ou None: O fonema simplificado ou None se for um silêncio.
    """
    if not p:
        return None
    # 1. Remover números (acentos tônicos)
    p = re.sub(r'\d+', '', p)
    # 2. Remover símbolos especiais
    p = p.replace('*', '').replace(')', '').replace('_', '').replace('`', '').replace('(', '')
    # 3. Padronizar para maiúsculas
    p = p.upper()
    # 4. Ignorar silêncios, espaços e marcadores inválidos
    if p in ['SIL', 'SP', 'SPN', ' ', '']:
        return None
    return p

class ReferenceGenerator:
    """
    Classe que gera fonemas de referência usando Whisper e um conversor grafema-fonema (G2P).
    """

    def __init__(self, whisper_model_size: str = "base"):
        """
        Inicializa o gerador de referência com o modelo Whisper especificado e a ferramenta g2p.

        Args:
            whisper_model_size (str, opcional): Tamanho do modelo Whisper ("tiny", "base", etc.).
        """
        print(f"Carregando modelo Whisper ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.g2p = G2p()
        
    def transcribe(self, audio_path: str) -> str:
        """
        Transcreve o áudio utilizando o modelo Whisper.

        Args:
            audio_path (str): Caminho para o arquivo de áudio WAV.

        Returns:
            str: O texto transcrito.
        """
        result = self.whisper_model.transcribe(audio_path)
        return result['text'].strip()
    
    def to_phonemes(self, text: str) -> List[str]:
        """
        Converte texto em fonemas Arpabet simplificados utilizando o g2p_en.

        Args:
            text (str): Texto a ser convertido.

        Returns:
            List[str]: Uma lista de strings, onde cada string é um fonema simplificado.
        """
        raw_phonemes = self.g2p(text)
        simplified = [simplify_phone(p) for p in raw_phonemes]
        return [p for p in simplified if p]

    def generate(self, audio_path: str) -> str:
        """
        Executa o pipeline completo: Áudio -> Texto -> String de Fonemas Simplificada.

        Args:
            audio_path (str): Caminho para o arquivo de áudio WAV.

        Returns:
            str: Uma string com os fonemas simplificados separados por espaços.
        """
        text = self.transcribe(audio_path)
        phonemes = self.to_phonemes(text)
        return " ".join(phonemes)

if __name__ == "__main__":
    import sys
    from src.utils.data_loader import parse_textgrid

    print("Testando ReferenceGenerator com um arquivo de áudio de amostra...")
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    ann_path = "data/l2arctic_release_v5.0/ABA/annotation/arctic_a0003.TextGrid"
    
    if os.path.exists(sample_wav):
        generator = ReferenceGenerator()
        
        print(f"\nProcessando: {sample_wav}")
        text = generator.transcribe(sample_wav)
        print(f"Transcrição Whisper: '{text}'")
        
        canonical_phonemes = generator.generate(sample_wav)
        print(f"Fonemas Canônicos: '{canonical_phonemes}'")
        
        if os.path.exists(ann_path):
            gt_ann = parse_textgrid(ann_path)
            # Usa 'phone' (alvo) para comparação com o ground truth
            gt_phones = [simplify_phone(ann['phone']) for ann in gt_ann]
            gt_phones_str = " ".join([p for p in gt_phones if p])
            print(f"Ground Truth Alvo: '{gt_phones_str}'")
    else:
        print(f"Arquivo não encontrado: {sample_wav}")
