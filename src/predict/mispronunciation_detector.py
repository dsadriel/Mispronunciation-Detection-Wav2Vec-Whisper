"""
Módulo principal do detector de erros de pronúncia.

Este script encapsula a lógica que integra o reconhecimento de fonemas extraídos
diretamente do áudio (modelo acústico) e os fonemas canônicos esperados (a partir
da transcrição Whisper + G2P), alinhando e detectando os desvios de pronúncia.
"""

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import pandas as pd
from src.predict.inference import get_latest_checkpoint, run_inference, simplify_phone, get_ground_truth
from src.predict.reference_generator import ReferenceGenerator
import jiwer

class MispronunciationDetector:
    """
    Classe para a detecção de erros de pronúncia comparando predição acústica com referência esperada.
    """
    
    def __init__(self, model_dir: str = "./wav2vec2-l2arctic-phonemes", whisper_size: str = "base", load_whisper: bool = True):
        """
        Inicializa o detector carregando os modelos acústico e de referência.

        Args:
            model_dir (str, opcional): Caminho para o diretório do modelo Wav2Vec2 ajustado.
            whisper_size (str, opcional): Tamanho do modelo Whisper ("tiny", "base", etc.).
            load_whisper (bool, opcional): Flag para determinar se o modelo Whisper será carregado na memória.
        """
        # Determinar o melhor device disponível (MPS para Mac, CUDA para NVIDIA, ou CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Usando hardware: {self.device}")

        self.model_id = get_latest_checkpoint(model_dir)
        print(f"Carregando Modelo Acústico (Wav2Vec2)...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        
        if load_whisper:
            print(f"Carregando Modelo de Referência (Whisper {whisper_size} + G2P)...")
            self.ref_gen = ReferenceGenerator(whisper_size)
            self.ref_gen.whisper_model.to(self.device)
        else:
            self.ref_gen = None

    def align_and_compare(self, acoustic: str, canonical: str):
        """
        Alinha as strings de fonemas acústicos e canônicos para comparar possíveis erros.

        Utiliza a biblioteca jiwer para realizar um alinhamento dinâmico que detecta
        substituições, deleções e inserções.

        Args:
            acoustic (str): Sequência de fonemas extraída do áudio.
            canonical (str): Sequência de fonemas canônicos esperados.

        Returns:
            list: Lista de dicionários representando o tipo de erro, o fonema esperado e o falado.
        """
        ref_list = canonical.split()
        hyp_list = acoustic.split()
        
        out = jiwer.process_words(canonical, acoustic)
        alignment_chunks = out.alignments[0]
        
        aligned_pairs = []
        for chunk in alignment_chunks:
            ref_sub = ref_list[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_sub = hyp_list[chunk.hyp_start_idx:chunk.hyp_end_idx]
            
            if chunk.type == 'equal':
                for r, h in zip(ref_sub, hyp_sub):
                    aligned_pairs.append({
                        'canonical': r,
                        'acoustic': h,
                        'type': 'correct'
                    })
            elif chunk.type == 'substitute':
                for r, h in zip(ref_sub, hyp_sub):
                    aligned_pairs.append({
                        'canonical': r,
                        'acoustic': h,
                        'type': 'substitution'
                    })
            elif chunk.type == 'delete':
                for r in ref_sub:
                    aligned_pairs.append({
                        'canonical': r,
                        'acoustic': 'sil',
                        'type': 'deletion'
                    })
            elif chunk.type == 'insert':
                for h in hyp_sub:
                    aligned_pairs.append({
                        'canonical': 'sil',
                        'acoustic': h,
                        'type': 'insertion'
                    })
        return aligned_pairs

    def detect(self, audio_path: str):
        """
        Gera a transcrição, fonemas esperados e ouvidos, e reporta erros para o áudio especificado.

        Args:
            audio_path (str): Caminho para o arquivo de áudio WAV.

        Returns:
            tuple: (acoustic_pred, canonical_ref, transcription, aligned_pairs) 
                   Contendo predição acústica, a referência canônica, o texto transcrito e a lista de alinhamentos.
        """
        if self.ref_gen is None:
            raise ValueError("Whisper não foi carregado. Não é possível rodar o pipeline completo detect().")
        print("\n" + "="*50)
        print("         ANÁLISE DE PRONÚNCIA")
        print("="*50)
        
        # 1. Obter Fonemas Acústicos (o que foi efetivamente dito)
        acoustic_pred = run_inference(audio_path, self.processor, self.model, device=self.device)
        
        # 2. Obter Fonemas Canônicos (o que deveria ter sido dito)
        transcription = self.ref_gen.transcribe(audio_path)
        canonical_ref_list = self.ref_gen.to_phonemes(transcription)
        canonical_ref = " ".join([p for p in canonical_ref_list if p.isalnum()])
        
        print(f"Transcrição Whisper: {transcription}")
        print(f"Fonemas Ouvidos:     {acoustic_pred}")
        print(f"Fonemas Esperados:   {canonical_ref}")
        
        # 3. Alinhar e Comparar
        aligned_pairs = self.align_and_compare(acoustic_pred, canonical_ref)
        
        # 4. Mostrar Resultados
        print("\nErros de Pronúncia Detectados:")
        errors = [p for p in aligned_pairs if p['type'] != 'correct']
        if not errors:
            print("  Parabéns! Nossos modelos não detectaram nenhum erro de pronúncia.")
        else:
            for err in errors:
                if err['type'] == 'substitution':
                    print(f"  * Substituição: Esperava-se '{err['canonical']}', mas ouviu-se '{err['acoustic']}'")
                elif err['type'] == 'deletion':
                    print(f"  * Deleção: O fonema '{err['canonical']}' não foi pronunciado.")
                elif err['type'] == 'insertion':
                    print(f"  * Inserção: O fonema '{err['acoustic']}' foi inserido desnecessariamente.")
        
        # 5. Visualização simples do alinhamento usando visualize_alignment do jiwer
        print("\nVisualização do Alinhamento (Comparação):")
        out = jiwer.process_words(canonical_ref, acoustic_pred)
        print(jiwer.visualize_alignment(out))
        print("="*50)
        
        return acoustic_pred, canonical_ref, transcription, aligned_pairs


if __name__ == "__main__":
    detector = MispronunciationDetector()
    
    # Testar em uma amostra
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    if os.path.exists(sample_wav):
        detector.detect(sample_wav)
        
        # Também comparar com Ground Truth se disponível
        ann_path = "data/l2arctic_release_v5.0/ABA/annotation/arctic_a0003.TextGrid"
        if os.path.exists(ann_path):
            gt = get_ground_truth(ann_path)
            print(f"Referência Manual (Ground Truth): {gt}")
    else:
        print("Arquivo de amostra não encontrado.")
