import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_evaluation_results(report_path="docs/evaluation_results.md"):
    """
    Parses the evaluation results file to extract TN, FP, FN, and TP.
    """
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expressions to extract TN, FP, FN, TP, robust against Markdown formatting
    tn_match = re.search(r"Verd.*Negativos.*?\bTN\b[^\d]*(\d+)", content, re.IGNORECASE)
    fp_match = re.search(r"Fals.*Positivos.*?\bFP\b[^\d]*(\d+)", content, re.IGNORECASE)
    fn_match = re.search(r"Fals.*Negativos.*?\bFN\b[^\d]*(\d+)", content, re.IGNORECASE)
    tp_match = re.search(r"Verd.*Positivos.*?\bTP\b[^\d]*(\d+)", content, re.IGNORECASE)
    
    if tn_match and fp_match and fn_match and tp_match:
        return {
            'tn': int(tn_match.group(1)),
            'fp': int(fp_match.group(1)),
            'fn': int(fn_match.group(1)),
            'tp': int(tp_match.group(1))
        }
    return None

def plot_confusion_matrix(tn, fp, fn, tp, save_path="docs/confusion_matrix.png", title="Phoneme-Level Confusion Matrix", cmap="viridis"):
    """
    Plots a highly polished 2x2 confusion matrix using Matplotlib.
    """
    # Create the matrix
    matrix = np.array([[tn, fp],
                      [fn, tp]])
    
    # Class labels
    classes = ['Correct\n(Negative)', 'Mispronunciation\n(Positive)']
    
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # Display the heatmap (using specified colormap, default is viridis)
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Formatting ticks
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10, fontweight='semibold')
    ax.set_yticklabels(classes, fontsize=10, fontweight='semibold', rotation=90, va="center")
    
    # Total samples per row (Actual class size)
    row_sums = matrix.sum(axis=1)
    
    # Cell acronyms
    acronyms = [['TN', 'FP'], ['FN', 'TP']]
    
    # Annotate cells with values and percentages
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            # Percentage relative to the row (Actual Class) -> Recall/Spec metrics
            percentage = (val / row_sums[i]) * 100 if row_sums[i] > 0 else 0
            
            label_text = f"{acronyms[i][j]}\n{val}\n({percentage:.1f}%)"
            
            # Automatically calculate text contrast color based on cell background luminance
            color_rgba = im.cmap(im.norm(val))
            luminance = 0.299 * color_rgba[0] + 0.587 * color_rgba[1] + 0.114 * color_rgba[2]
            text_color = "black" if luminance > 0.5 else "white"
            
            # Adding cell labels
            ax.text(j, i, label_text,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight='bold')
            
    # Labels and Title
    ax.set_ylabel('Ground Truth', fontsize=11, fontweight='bold', labelpad=15)
    ax.set_xlabel('Prediction', fontsize=11, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Layout adjustments
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Matriz de confusão salva em: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Gera uma imagem da matriz de confusão para CAPT.")
    parser.add_argument("--tn", type=int, help="Quantidade de Verdadeiros Negativos (True Correct)")
    parser.add_argument("--fp", type=int, help="Quantidade de Falsos Positivos (False Alarm)")
    parser.add_argument("--fn", type=int, help="Quantidade de Falsos Negativos (Missed Error)")
    parser.add_argument("--tp", type=int, help="Quantidade de Verdadeiros Positivos (True Error)")
    parser.add_argument("--output", type=str, default="docs/confusion_matrix.png", help="Caminho para salvar a imagem")
    parser.add_argument("--title", type=str, default="Phoneme-Level Confusion Matrix", help="Título do gráfico")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap padrão do Matplotlib (ex: viridis, Blues, plasma, magma, coolwarm)")
    args = parser.parse_args()

    # If any metric is not provided, try to read from docs/evaluation_results.md
    if None in (args.tn, args.fp, args.fn, args.tp):
        print("Métricas não fornecidas por argumento. Tentando ler de 'docs/evaluation_results.md'...")
        metrics = parse_evaluation_results()
        if metrics:
            print(f"Métricas lidas com sucesso: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
            plot_confusion_matrix(metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'], args.output, args.title, args.cmap)
        else:
            # Default fallback data (L2-ARCTIC typical evaluation values)
            print("Não foi possível ler as métricas do arquivo. Gerando com valores padrão (L2-ARCTIC baseline)...")
            plot_confusion_matrix(18652, 1484, 1586, 1937, args.output, args.title, args.cmap)
    else:
        plot_confusion_matrix(args.tn, args.fp, args.fn, args.tp, args.output, args.title, args.cmap)

if __name__ == "__main__":
    main()
