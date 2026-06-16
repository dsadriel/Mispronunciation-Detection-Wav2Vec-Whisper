import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# Adicionar a raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath('.'))

from src.mispronunciation_detector import MispronunciationDetector
from src.data_loader import parse_textgrid
from src.inference import simplify_phone, run_inference

def evaluate(limit=None):
    print("="*60)
    print("      TF-PLN: AVALIAÇÃO DO DETECTOR DE ERROS DE PRONÚNCIA")
    print("="*60)
    
    # 1. Carregar o test split
    test_csv = "data/test_split.csv"
    if not os.path.exists(test_csv):
        print(f"[ERRO] Arquivo de teste não encontrado em: {test_csv}")
        return
        
    df = pd.read_csv(test_csv)
    if limit:
        print(f"Limitando a avaliação às primeiras {limit} amostras.")
        df = df.head(limit)
    else:
        print(f"Total de amostras para avaliação: {len(df)}")
        
    # 2. Inicializar detector sem carregar o Whisper (otimização de tempo/memória)
    detector = MispronunciationDetector(load_whisper=False)
    
    # Listas globais para métricas
    y_true_all = []
    y_pred_all = []
    
    # Dicionários para métricas por falante (sotaque)
    speaker_results = {}
    
    # Dicionário de contagem de tipos de erro (Substituição, Deleção)
    # Para análise de falsos positivos/negativos
    error_types_stats = {
        'substitution': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
        'deletion': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
    }
    
    # 3. Processar arquivos
    print("\nIniciando inferência e alinhamento no conjunto de teste...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row['wav_path']
        ann_path = row['ann_path']
        speaker = row['speaker']
        
        if not (os.path.exists(wav_path) and os.path.exists(ann_path)):
            continue
            
        try:
            # A. Carregar anotações manuais (Ground Truth)
            annotations = parse_textgrid(ann_path)
            canonical_list = []
            gt_labels = []
            gt_types = []
            
            for ann in annotations:
                c = simplify_phone(ann['phone'])
                p = simplify_phone(ann['produced'])
                if not c:
                    continue
                canonical_list.append(c)
                
                # Identificar se há erro de pronúncia na posição
                is_error = 0
                err_type = None
                if ann['error_type'] in ['s', 'd'] or c != p:
                    is_error = 1
                    err_type = 'deletion' if (p == 'SIL' or not p) else 'substitution'
                
                gt_labels.append(is_error)
                gt_types.append(err_type)
                
            if not canonical_list:
                continue
                
            canonical_str = " ".join(canonical_list)
            
            # B. Rodar inferência acústica (Wav2Vec2)
            acoustic_pred = run_inference(
                wav_path, 
                detector.processor, 
                detector.model, 
                device=detector.device
            )
            
            # C. Alinhamento de predição com o alvo
            aligned_pairs = detector.align_and_compare(acoustic_pred, canonical_str)
            
            # D. Extrair predições correspondentes ao alvo (filtrando inserções)
            pred_labels = []
            pred_types = []
            for pair in aligned_pairs:
                if pair['canonical'] == 'sil':
                    continue  # Pular inserções na hipótese para avaliação direta do alvo
                
                is_pred_error = 0
                pred_type = None
                if pair['type'] in ['substitution', 'deletion']:
                    is_pred_error = 1
                    pred_type = pair['type']
                    
                pred_labels.append(is_pred_error)
                pred_types.append(pred_type)
                
            # Verificar se os tamanhos são correspondentes
            if len(gt_labels) != len(pred_labels):
                # Caso haja discrepância de alinhamento rara, ignoramos a amostra para não corromper
                continue
                
            # E. Acumular dados globais e por speaker
            y_true_all.extend(gt_labels)
            y_pred_all.extend(pred_labels)
            
            if speaker not in speaker_results:
                speaker_results[speaker] = {'y_true': [], 'y_pred': []}
            speaker_results[speaker]['y_true'].extend(gt_labels)
            speaker_results[speaker]['y_pred'].extend(pred_labels)
            
            # Contagem de tipos de erro detalhados
            for gt_l, pred_l, gt_t, pred_t in zip(gt_labels, pred_labels, gt_types, pred_types):
                if gt_l == 1 and pred_l == 1:
                    # True Positive
                    # Se acertou o tipo de erro específico
                    t = gt_t if gt_t in error_types_stats else 'substitution'
                    error_types_stats[t]['true_pos'] += 1
                elif gt_l == 0 and pred_l == 1:
                    # False Positive
                    t = pred_t if pred_t in error_types_stats else 'substitution'
                    error_types_stats[t]['false_pos'] += 1
                elif gt_l == 1 and pred_l == 0:
                    # False Negative
                    t = gt_t if gt_t in error_types_stats else 'substitution'
                    error_types_stats[t]['false_neg'] += 1
                    
        except Exception as e:
            # Ignora erros individuais de leitura de arquivo/processamento
            continue
            
    if not y_true_all:
        print("[ERRO] Nenhuma amostra foi processada com sucesso.")
        return
        
    # 4. Calcular métricas globais
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    
    # 5. Apresentar Relatório
    report_lines = []
    report_lines.append("# Relatório de Avaliação do Detector de Erros de Pronúncia (TF-PLN)")
    report_lines.append(f"\nEste documento apresenta os resultados da avaliação do pipeline de detecção de erros de pronúncia comparado com o ground truth manual do dataset **L2-ARCTIC**.")
    report_lines.append("\n## 📊 Métricas Globais (Nível de Fonema)")
    report_lines.append(f"- **Acurácia (Accuracy)**: {accuracy:.4f} ({accuracy*100:.2f}%)")
    report_lines.append(f"- **Precisão (Precision)**: {precision:.4f} ({precision*100:.2f}%)")
    report_lines.append(f"- **Sensibilidade (Recall)**: {recall:.4f} ({recall*100:.2f}%)")
    report_lines.append(f"- **F1-Score**: {f1:.4f} ({f1*100:.2f}%)")
    
    report_lines.append("\n### Matriz de Confusão:")
    report_lines.append(f"- **Verdadeiros Negativos (TN - Pronúncia Correta Predita Correta)**: {tn}")
    report_lines.append(f"- **Falsos Positivos (FP - Erro Detectado Incorretamente)**: {fp}")
    report_lines.append(f"- **Falsos Negativos (FN - Erro Não Detectado)**: {fn}")
    report_lines.append(f"- **Verdadeiros Positivos (TP - Erro Detectado Corretamente)**: {tp}")
    
    # Métricas por tipo de erro
    report_lines.append("\n## 🔍 Métricas Detalhadas por Tipo de Erro")
    report_lines.append("| Tipo de Erro | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precisão (P) | Revocação (R) | F1-Score |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for err_name, stats in error_types_stats.items():
        tp_e = stats['true_pos']
        fp_e = stats['false_pos']
        fn_e = stats['false_neg']
        
        p_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0
        r_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
        f_e = 2 * p_e * r_e / (p_e + r_e) if (p_e + r_e) > 0 else 0
        
        report_lines.append(f"| {err_name.capitalize()} | {tp_e} | {fp_e} | {fn_e} | {p_e:.4f} | {r_e:.4f} | {f_e:.4f} |")
        
    # Métricas por Falante (Sotaque)
    report_lines.append("\n## 🗣️ Análise Individualizada por Falante (Sotaque L2)")
    report_lines.append("| Falante | Total Fonemas | Erros Reais (GT) | Erros Preditos | Acurácia | Precisão | Revocação | F1-Score |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # Mapeamento de falantes para sotaques no L2-ARCTIC
    # ABA (Arabic), HQTV (Chinese), ERMS (Spanish), SKI (Korean), TLV (Vietnamese), etc.
    # Vamos apenas ordenar por F1-Score decrescente para ver onde o modelo funciona melhor
    speaker_metrics = []
    for spk, data in speaker_results.items():
        y_t = data['y_true']
        y_p = data['y_pred']
        
        spk_acc = accuracy_score(y_t, y_p)
        spk_p, spk_r, spk_f, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
        
        speaker_metrics.append({
            'speaker': spk,
            'total': len(y_t),
            'gt_errors': sum(y_t),
            'pred_errors': sum(y_p),
            'acc': spk_acc,
            'p': spk_p,
            'r': spk_r,
            'f1': spk_f
        })
        
    speaker_metrics = sorted(speaker_metrics, key=lambda x: x['f1'], reverse=True)
    for m in speaker_metrics:
        report_lines.append(
            f"| {m['speaker']} | {m['total']} | {m['gt_errors']} | {m['pred_errors']} | "
            f"{m['acc']:.4f} | {m['p']:.4f} | {m['r']:.4f} | {m['f1']:.4f} |"
        )
        
    # 6. Gravar relatório como arquivo markdown
    output_report_path = "docs/evaluation_results.md"
    with open(output_report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    # 7. Print no terminal
    print("\n" + "="*40)
    print("             RESULTADOS FINAIS")
    print("="*40)
    print(f"Acurácia:   {accuracy*100:.2f}%")
    print(f"Precisão:   {precision*100:.2f}%")
    print(f"Recall:     {recall*100:.2f}%")
    print(f"F1-Score:   {f1*100:.2f}%")
    print(f"\nMatriz de Confusão:")
    print(f"  TN: {tn}  |  FP: {fp}")
    print(f"  FN: {fn}  |  TP: {tp}")
    print("="*40)
    print(f"Relatório completo salvo em: {output_report_path}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do Detector de Erros de Pronúncia (TF-PLN)")
    parser.add_argument("--limit", type=int, default=None, help="Limite de arquivos de áudio para avaliar (opcional)")
    args = parser.parse_args()
    
    evaluate(limit=args.limit)
