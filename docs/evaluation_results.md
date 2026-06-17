# Relatório de Avaliação do Detector de Erros de Pronúncia (TF-PLN)

Este documento apresenta os resultados da avaliação do pipeline de detecção de erros de pronúncia comparado com o ground truth manual do dataset **L2-ARCTIC**.

## 📊 Métricas Globais (Nível de Fonema)
- **Acurácia (Accuracy)**: 0.8702 (87.02%)
- **Precisão (Precision)**: 0.5662 (56.62%)
- **Sensibilidade (Recall)**: 0.5498 (54.98%)
- **F1-Score**: 0.5579 (55.79%)

### Matriz de Confusão:
- **Verdadeiros Negativos (TN - Pronúncia Correta Predita Correta)**: 18652
- **Falsos Positivos (FP - Erro Detectado Incorretamente)**: 1484
- **Falsos Negativos (FN - Erro Não Detectado)**: 1586
- **Verdadeiros Positivos (TP - Erro Detectado Corretamente)**: 1937

## 🔍 Métricas Detalhadas por Tipo de Erro
| Tipo de Erro | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precisão (P) | Revocação (R) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Substitution | 1458 | 1098 | 1398 | 0.5704 | 0.5105 | 0.5388 |
| Deletion | 479 | 386 | 188 | 0.5538 | 0.7181 | 0.6253 |

## 🗣️ Análise Individualizada por Falante (Sotaque L2)
| Falante | Total Fonemas | Erros Reais (GT) | Erros Preditos | Acurácia | Precisão | Revocação | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TLV | 914 | 216 | 254 | 0.8840 | 0.7165 | 0.8426 | 0.7745 |
| HQTV | 984 | 270 | 255 | 0.8628 | 0.7647 | 0.7222 | 0.7429 |
| THV | 1006 | 271 | 231 | 0.8628 | 0.7879 | 0.6716 | 0.7251 |
| PNV | 1026 | 170 | 172 | 0.8889 | 0.6628 | 0.6706 | 0.6667 |
| LXC | 985 | 211 | 200 | 0.8406 | 0.6350 | 0.6019 | 0.6180 |
| NCC | 945 | 109 | 138 | 0.8995 | 0.5507 | 0.6972 | 0.6154 |
| BWC | 977 | 174 | 198 | 0.8465 | 0.5606 | 0.6379 | 0.5968 |
| EBVS | 1021 | 197 | 220 | 0.8325 | 0.5591 | 0.6244 | 0.5899 |
| ERMS | 1057 | 225 | 206 | 0.8325 | 0.6165 | 0.5644 | 0.5893 |
| MBMPS | 935 | 116 | 131 | 0.8727 | 0.4885 | 0.5517 | 0.5182 |
| YDCK | 933 | 94 | 94 | 0.9014 | 0.5106 | 0.5106 | 0.5106 |
| ZHAA | 1036 | 110 | 103 | 0.8948 | 0.5049 | 0.4727 | 0.4883 |
| TNI | 924 | 109 | 112 | 0.8755 | 0.4732 | 0.4862 | 0.4796 |
| HJK | 946 | 69 | 90 | 0.9123 | 0.4222 | 0.5507 | 0.4780 |
| YKWK | 1070 | 120 | 106 | 0.8879 | 0.5000 | 0.4417 | 0.4690 |
| RRBI | 1046 | 117 | 132 | 0.8690 | 0.4242 | 0.4786 | 0.4498 |
| YBAA | 1041 | 108 | 110 | 0.8847 | 0.4455 | 0.4537 | 0.4495 |
| SKA | 925 | 124 | 117 | 0.8562 | 0.4615 | 0.4355 | 0.4481 |
| NJS | 968 | 110 | 104 | 0.8698 | 0.4231 | 0.4000 | 0.4112 |
| HKK | 972 | 79 | 118 | 0.8796 | 0.3390 | 0.5063 | 0.4061 |
| ASI | 969 | 169 | 99 | 0.8328 | 0.5354 | 0.3136 | 0.3955 |
| TXHC | 1069 | 119 | 98 | 0.8756 | 0.4286 | 0.3529 | 0.3871 |
| SVBI | 951 | 154 | 68 | 0.8360 | 0.4853 | 0.2143 | 0.2973 |
| ABA | 959 | 82 | 65 | 0.8905 | 0.3231 | 0.2561 | 0.2857 |