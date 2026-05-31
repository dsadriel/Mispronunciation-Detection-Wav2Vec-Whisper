# Estratégia de Fine-tuning: Wav2vec 2.0 para Reconhecimento de Fonemas

Este documento descreve a abordagem técnica para ajustar o modelo Wav2vec 2.0 para a tarefa de reconhecimento fonético utilizando o dataset L2-ARCTIC.

## 1. Arquitetura: CTC (Connectionist Temporal Classification)
Utilizaremos o `Wav2Vec2ForCTC` da biblioteca Transformers. Esta arquitetura é ideal para reconhecimento de fala pois:
- Não exige alinhamento perfeito entre áudio e labels no treinamento.
- Prevê uma distribuição de probabilidade sobre o vocabulário para cada frame de áudio.
- Utiliza um símbolo especial `[PAD]` e um `blank` para lidar com silêncios e transições.

## 2. Preparação do Vocabulário
O L2-ARCTIC utiliza anotações baseadas em uma mistura de Arpabet e símbolos específicos. Nosso vocabulário incluirá:
- Fonemas padrão do inglês.
- Símbolos de erro (opcionalmente, ou focaremos na produção real do falante).
- Tokens especiais: `[PAD]`, `[UNK]`, `|` (espaço entre palavras).

## 3. Divisão do Dataset
Para garantir a validade acadêmica e evitar vazamento de dados (*data leakage*):
- **Treino (80%):** Utilizado para o ajuste dos pesos do modelo.
- **Teste (20%):** Utilizado exclusivamente para a avaliação final das métricas (PER, Precisão, Recall).
- **Estratégia:** A divisão será feita de forma estratificada por falante, garantindo que o modelo veja uma variedade de sotaques durante o treino e seja testado em todos eles.

## 4. Hiperparâmetros Iniciais
- **Modelo Base:** `facebook/wav2vec2-base` ou `facebook/wav2vec2-large-lv60`.
- **Learning Rate:** 1e-4 com scheduler linear.
- **Optimizer:** AdamW.
- **Loss:** CTCLoss.

## 5. Pipeline de Inferência
Após o fine-tuning, o modelo receberá um áudio e retornará a sequência de fonemas produzida. Esta sequência será comparada com a referência canônica (gerada via Whisper + G2P) para detectar erros.
