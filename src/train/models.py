"""
Módulo responsável por carregar e inicializar o modelo acústico Wav2Vec 2.0
para a tarefa de treinamento de reconhecimento de fonemas (CTC).
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
import json
import os

def load_wav2vec2_model(vocab_path: str, model_id: str = "facebook/wav2vec2-base"):
    """
    Carrega ou inicializa um modelo Wav2vec 2.0 para CTC (Connectionist Temporal Classification)
    com o vocabulário fornecido.
    
    Args:
        vocab_path (str): Caminho para o arquivo JSON contendo o vocabulário.
        model_id (str): Identificador do modelo base no Hugging Face (padrão: "facebook/wav2vec2-base").
        
    Returns:
        Wav2Vec2ForCTC: O modelo carregado e configurado para a tarefa de CTC.
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    # Modelos padrão Wav2vec 2.0 esperam uma estrutura específica de tokenizador
    # Podemos precisar salvar uma configuração completa de tokenizador
    # Por enquanto, vamos apenas mostrar como inicializar
    
    config = Wav2Vec2Config.from_pretrained(
        model_id,
        vocab_size=len(vocab),
        pad_token_id=vocab.get("[PAD]", 0),
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True
    )
    
    model = Wav2Vec2ForCTC.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
    return model

if __name__ == "__main__":
    vocab_file = "data/vocab.json"
    if os.path.exists(vocab_file):
        # Este é apenas um teste para verificar se o modelo carrega corretamente
        print("Loading model structure (testing)...")
        model = load_wav2vec2_model(vocab_file)
        print("Model loaded successfully.")
        print("Note: In a real fine-tuning, we would use the Trainer API.")
    else:
        print("Vocab file not found.")
