# Mispronunciation Detection for L2 English Speakers (TF-PLN)

Este repositório contém o desenvolvimento do Trabalho Final da disciplina de Processamento de Linguagem Natural (UFRGS - 2026). O projeto foca na detecção automática de erros de pronúncia (substituições, deleções e inserções) em falantes não nativos de inglês, utilizando arquiteturas baseadas em Transformers.

## 🎯 Objetivo
Desenvolver um pipeline de detecção de erros de pronúncia que compara a extração fonética acústica de um modelo **Wav2vec 2.0** com a transcrição fonética alvo gerada a partir de modelos de reconhecimento de fala (**Whisper**) e algoritmos de **Grapheme-to-Phoneme (G2P)**.

## 🛠️ Metodologia e Arquitetura
O sistema opera através de uma abordagem de alinhamento de sequências:

1.  **Reconhecimento Fonético Acústico:** Utilização de um modelo Wav2vec 2.0 (fine-tuned para reconhecimento de fonemas) para extrair a sequência de fonemas realmente produzida pelo falante a partir do áudio bruto.
2.  **Geração de Referência (Canonical):** 
    - Transcrição do áudio para texto via **OpenAI Whisper**.
    - Conversão do texto transcrito em uma sequência de fonemas ideal (canonical) via **Phonemizer (G2P)**.
3.  **Alinhamento e Detecção:** 
    - Alinhamento das duas sequências (Acústica vs. Canônica) utilizando algoritmos de distância de edição (Levenshtein) ou Dynamic Time Warping (DTW).
    - Identificação de discrepâncias categorizadas como: **Substituição (S)**, **Deleção (D)** ou **Inserção (I)**.
4.  **Validação:** Comparação dos erros detectados com o *ground truth* do dataset **L2-ARCTIC**, utilizando métricas de precisão, recall, F1-score e taxa de erro fonético (PER).

## 📊 Dataset
Utilizamos o **L2-ARCTIC Corpus**, que contém gravações de falantes não nativos de diversas origens (Árabe, Chinês, Hindi, Coreano, Espanhol, Vietnamita) com anotações manuais detalhadas de erros fonéticos.

## 🚀 Como Executar
(Instruções de instalação e execução serão adicionadas conforme o desenvolvimento)

## 📄 Declaração de Uso de IA
Este projeto utiliza o **Gemini CLI** para:
- Planejamento de arquitetura e roadmap.
- Geração e refatoração de código de processamento de áudio e alinhamento.
- Brainstorming de métricas e análise de resultados.
*Todo o conteúdo foi revisado e validado pelos autores.*
