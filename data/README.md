# Dados do Projeto (Dataset e Arquivos Auxiliares)

Este diretório contém os artefatos de dados necessários para treinar e avaliar o modelo de detecção de erros de pronúncia.

## Estrutura de Arquivos

*   **`train_split.csv`**: Arquivo CSV contendo a separação dos dados que foram utilizados para treinar (fazer o *fine-tuning*) do nosso modelo acústico (Wav2Vec 2.0). Ele mapeia os caminhos dos áudios para as suas respectivas anotações fonéticas.
*   **`test_split.csv`**: Arquivo CSV contendo a separação dos dados de teste. Utilizado pelos scripts em `src/predict/` para calcular as métricas de avaliação e gerar a matriz de confusão, garantindo que a validação seja feita em falantes/áudios não vistos durante o treino.
*   **`vocab.json`**: Arquivo gerado pelo script `prepare_simple_vocab.py` que define o vocabulário fonético simplificado de 44 tokens adotado pelo modelo acústico, agrupando variantes de fonemas do L2-ARCTIC em classes base.

## ⚠️ Dependência Externa: L2-ARCTIC Corpus

A pasta raiz do dataset **L2-ARCTIC** (geralmente nomeada como `l2arctic_release_v5.0`) **não é versionada** neste repositório devido ao seu tamanho.

Para que os scripts em `src/` funcionem corretamente, é estritamente necessário realizar o download do corpus L2-ARCTIC (disponível [neste link oficial](https://psi.engr.tamu.edu/l2-arctic-corpus/#:~:text=Your%20Name%20(required))) e extraí-lo aqui dentro da pasta `data/`. A estrutura esperada ficará assim:

```text
TF-PLN/
└── data/
    ├── l2arctic_release_v5.0/    <-- (Coloque o dataset extraído aqui)
    │   ├── ABA/
    │   ├── EBVS/
    │   ├── HKK/
    │   └── ... (outros falantes)
    ├── train_split.csv
    ├── test_split.csv
    └── vocab.json
```
