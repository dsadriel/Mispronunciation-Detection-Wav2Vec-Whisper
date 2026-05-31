# Roadmap de Desenvolvimento: TF-PLN

Este roadmap descreve as fases do projeto para a detecção de erros de pronúncia.

## 📅 Datas Críticas
- **19/05:** Apresentação do andamento (Concluído).
- **16/06 - 23/06:** Apresentações finais.
- **23/06:** Entrega final (Artigo SBC + Código).

---

## 🏗️ Fases do Projeto

### Fase 1: Preparação de Dados e Exploração (Semana 1)
- [x] Mapear a estrutura do dataset L2-ARCTIC.
- [x] Implementar parser para arquivos `.TextGrid` (extração de anotações de erro).
- [x] Criar scripts de pré-processamento de áudio (resampling para 16kHz).
- [x] Explorar subconjunto de falantes para testes iniciais (ex: ABA, SKI).

### Fase 2: Pipeline de Reconhecimento (Semana 1-2)
- [ ] Configurar modelo **Wav2vec 2.0** para reconhecimento de fonemas.
- [ ] Integrar **OpenAI Whisper** para transcrição robusta.
- [ ] Configurar **G2P (Phonemizer)** para gerar sequências fonéticas canônicas.
- [ ] Validar extração básica em áudios de controle.

### Fase 3: Alinhamento e Detecção de Erros (Semana 2-3)
- [ ] Implementar algoritmo de alinhamento de sequências (Levenshtein/DTW).
- [ ] Desenvolver lógica de identificação de S, D, I (Substituição, Deleção, Inserção).
- [ ] Mapear fonemas do Wav2vec 2.0 para o conjunto do L2-ARCTIC (IPA/Arpabet).

### Fase 4: Avaliação e Análise de Resultados (Semana 3)
- [ ] Executar pipeline em todo o subconjunto selecionado do L2-ARCTIC.
- [ ] Calcular métricas: Precisão, Recall, F1-Score (por tipo de erro e global).
- [ ] Gerar matrizes de confusão fonética.
- [ ] Realizar análise qualitativa dos erros mais comuns e falhas do modelo.

### Fase 5: Documentação e Artigo (Semana 4)
- [ ] Redigir seções de Metodologia e Resultados no formato SBC.
- [ ] Consolidar referências bibliográficas.
- [ ] Preparar slides para a apresentação final.
- [ ] Revisão final do código e documentação.
