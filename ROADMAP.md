# Roadmap de Desenvolvimento: TF-PLN - Detecção de Erros de Pronúncia

Este roadmap descreve as fases do projeto para o desenvolvimento, avaliação e redação do Trabalho Final da disciplina de Processamento de Linguagem Natural (UFRGS).

---

## 📅 Datas Críticas e Entregas (Baseado no Enunciado)
- **19/05:** Apresentação do andamento (Concluído)
- **16, 18 e 23/06:** Apresentações Finais (**Em andamento**) - *Apresentação de exatamente 15 minutos*
- **23/06:** Entrega Final (Artigo SBC + Código no GitHub) - *Daqui a 7 dias*

---

## 📋 Requisitos Obrigatórios do Trabalho (UFRGS)
- [x] **Conexão com Linguagem Natural**: Análise fonológica baseada em NLP.
- [x] **Dataset Anotado (Ground Truth)**: Utilização do corpus **L2-ARCTIC**.
- [x] **Uso de Transformers**: Integração de **Wav2Vec 2.0** (Acústico) e **Whisper** (G2P).
- [ ] **Formato do Artigo**: 8 a 10 páginas (excluindo referências) no padrão da **SBC**.
- [ ] **Apresentação Final**: Apresentação oral de exatamente **15 minutos**.
- [x] **Declaração de IA**: Registro transparente do uso do Antigravity CLI e assistentes em [README.md](file:///Users/adsouza/GitHub/TF-PLN/README.md) e [AGENTS.md](file:///Users/adsouza/GitHub/TF-PLN/AGENTS.md).

---

## 🏗️ Fases do Projeto e Status Atual

### Fase 1: Preparação de Dados e Exploração (Concluída)
- [x] Mapear a estrutura do dataset L2-ARCTIC ([dataset_exploration.md](file:///Users/adsouza/GitHub/TF-PLN/docs/dataset_exploration.md)).
- [x] Implementar parser para arquivos `.TextGrid` para extrair anotações manuais (`src/data_loader.py`).
- [x] Criar scripts de pré-processamento de áudio (resampling para 16kHz).
- [x] Separar dataset em treino/teste estratificado por falante (`src/prepare_fine_tuning.py`).

### Fase 2: Pipeline de Reconhecimento e Pivô de Vocabulário (Concluída)
- [x] Configurar modelo **Wav2vec 2.0** (`Wav2Vec2ForCTC`) para reconhecimento de fonemas.
- [x] Resolver instabilidade do `CTCLoss` no chip Apple Silicon (MPS/CPU fallback).
- [x] Implementar pivô estratégico de redução de vocabulário de **142 para 44 tokens** (`src/prepare_simple_vocab.py`).
- [x] Treinar o modelo acústico por 25 épocas, atingindo **Phone Error Rate (PER) de 17.6% (Acurácia de 82.4%)** e loss de validação de **0.3046** ([technical_challenges.md](file:///Users/adsouza/GitHub/TF-PLN/docs/technical_challenges.md)).
- [x] Integrar **OpenAI Whisper** + **g2p_en** para geração da sequência fonética de referência (canonical) (`src/reference_generator.py`).

### Fase 3: Alinhamento e Detecção de Erros (Concluída)
- [x] Mapear fonemas do Wav2vec 2.0 para o vocabulário simplificado do L2-ARCTIC.
- [x] Implementar visualização básica de alinhamento com a biblioteca `jiwer`.
- [x] Refinar a lógica de identificação estruturada de **Substituições (S)**, **Deleções (D)** e **Inserções (I)** de forma programática em `src/mispronunciation_detector.py` (método `align_and_compare`).

### Fase 4: Avaliação e Análise de Resultados (Pendente)
- [ ] **Pendente**: Criar script de avaliação em massa (ex: `src/evaluate_detector.py`) para processar o [test_split.csv](file:///Users/adsouza/GitHub/TF-PLN/data/test_split.csv).
- [ ] **Pendente**: Calcular métricas de classificação de erro de pronúncia (**Precisão, Recall e F1-Score**) de forma global e individual por falante/sotaque.
- [ ] **Pendente**: Gerar matriz de confusão fonética para identificar as substituições mais comuns.
- [ ] **Pendente**: Realizar análise qualitativa com foco nos falantes de maior taxa de desvio (ex: THV, HQTV).

### Fase 5: Documentação, Artigo SBC e Apresentação (Pendente)
- [ ] **Pendente**: Redigir o artigo científico (8 a 10 páginas) conforme estrutura recomendada no enunciado (Resumo, Introdução, Referencial, Trabalhos Relacionados, Metodologia, Experimentos/Resultados, Conclusão).
- [ ] **Pendente**: Consolidar a seção obrigatória de Declaração de Uso de IA Generativa.
- [ ] **Pendente**: Montar os slides de apresentação e ensaiar para cumprir o tempo de **exatamente 15 minutos**.

