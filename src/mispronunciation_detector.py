import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import pandas as pd
from src.inference import get_latest_checkpoint, run_inference, simplify_phone, get_ground_truth
from src.reference_generator import ReferenceGenerator
import jiwer

class MispronunciationDetector:
    def __init__(self, model_dir: str = "./wav2vec2-l2arctic-phonemes", whisper_size: str = "base", load_whisper: bool = True):
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
        if self.ref_gen is None:
            raise ValueError("Whisper não foi carregado. Não é possível rodar o pipeline completo detect().")
        print("\n" + "="*50)
        print("         ANÁLISE DE PRONÚNCIA")
        print("="*50)
        
        # 1. Get Acoustic Phonemes (what was actually said)
        acoustic_pred = run_inference(audio_path, self.processor, self.model, device=self.device)
        
        # 2. Get Canonical Phonemes (what should have been said)
        transcription = self.ref_gen.transcribe(audio_path)
        canonical_ref_list = self.ref_gen.to_phonemes(transcription)
        canonical_ref = " ".join([p for p in canonical_ref_list if p.isalnum()])
        
        print(f"Transcrição Whisper: {transcription}")
        print(f"Fonemas Ouvidos:     {acoustic_pred}")
        print(f"Fonemas Esperados:   {canonical_ref}")
        
        # 3. Align and Compare
        aligned_pairs = self.align_and_compare(acoustic_pred, canonical_ref)
        
        # 4. Display Results
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
        
        # 5. Simple alignment display using jiwer's visualize_alignment
        print("\nVisualização do Alinhamento (Comparação):")
        out = jiwer.process_words(canonical_ref, acoustic_pred)
        print(jiwer.visualize_alignment(out))
        print("="*50)
        
        return acoustic_pred, canonical_ref, transcription, aligned_pairs


if __name__ == "__main__":
    detector = MispronunciationDetector()
    
    # Test on a sample
    sample_wav = "data/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav"
    if os.path.exists(sample_wav):
        detector.detect(sample_wav)
        
        # Also compare with Ground Truth if available
        ann_path = "data/l2arctic_release_v5.0/ABA/annotation/arctic_a0003.TextGrid"
        if os.path.exists(ann_path):
            gt = get_ground_truth(ann_path)
            print(f"Manual Ground Truth: {gt}")
    else:
        print("Sample file not found.")
