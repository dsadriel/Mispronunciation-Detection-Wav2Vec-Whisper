"""
Módulo responsável por carregar e analisar os dados do dataset L2-ARCTIC.
Oferece funções para processar arquivos de anotação TextGrid e para
construir um DataFrame organizado com as informações dos locutores, 
caminhos dos áudios e caminhos das anotações correspondentes.
"""

import textgrid
import os
import pandas as pd
from typing import List, Dict, Tuple

def parse_textgrid(file_path: str) -> List[Dict]:
    """
    Analisa um arquivo TextGrid e extrai as anotações em nível de fonema.

    Parâmetros:
        file_path (str): Caminho para o arquivo TextGrid a ser analisado.

    Retorna:
        List[Dict]: Uma lista de dicionários contendo os dados extraídos dos intervalos.
                    Cada dicionário contém o tempo de início, fim, fonema alvo, fonema produzido
                    e o tipo de erro (se houver).
    """
    try:
        tg = textgrid.TextGrid.fromFile(file_path)
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return []
        
    phone_tier = next((t for t in tg if t.name == 'phones'), None)
    if not phone_tier: return []
    
    annotations = []
    for interval in phone_tier:
        label = interval.mark.strip()
        if not label: continue
        
        parts = [p.strip() for p in label.split(',')]
        if len(parts) == 3:
            target, produced, error_type = parts
            annotations.append({
                'start': interval.minTime, 'end': interval.maxTime,
                'phone': target, 'produced': produced, 'error_type': error_type
            })
        else:
            annotations.append({
                'start': interval.minTime, 'end': interval.maxTime,
                'phone': label, 'produced': label, 'error_type': None
            })
    return annotations

def get_dataset(data_dir: str) -> pd.DataFrame:
    """
    Retorna um DataFrame limpo contendo informações sobre os locutores, 
    caminhos dos arquivos de áudio (wav) e caminhos de anotação (TextGrid).

    Parâmetros:
        data_dir (str): Diretório base contendo os dados dos locutores.

    Retorna:
        pd.DataFrame: Um DataFrame Pandas com colunas 'speaker', 'file_id', 'wav_path' e 'ann_path'.
    """
    inventory = []
    speakers = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for spk in speakers:
        wav_dir = os.path.join(data_dir, spk, 'wav')
        ann_dir = os.path.join(data_dir, spk, 'annotation')
        if not (os.path.exists(wav_dir) and os.path.exists(ann_dir)): continue
        
        for f in os.listdir(wav_dir):
            if f.endswith('.wav'):
                fid = f.replace('.wav', '')
                ann_path = os.path.join(ann_dir, fid + '.TextGrid')
                if os.path.exists(ann_path):
                    inventory.append({
                        'speaker': spk, 'file_id': fid,
                        'wav_path': os.path.join(wav_dir, f), 'ann_path': ann_path
                    })
    return pd.DataFrame(inventory)
