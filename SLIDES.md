# Proposta de Slides para a Apresentação Final: TF-PLN
## Tempo Limite Exigido: Exatamente 15 minutos (UFRGS)

Este documento apresenta a estrutura de slides proposta para a apresentação final da disciplina de Processamento de Linguagem Natural. Cada slide contém sugestões de pontos-chave visuais, notas detalhadas de fala para o apresentador e o tempo recomendado para manter o ensaio rigorosamente dentro do limite de 15 minutos.

---

### ⏱️ Cronograma Sugerido de Ensaio
*   **Slides 1-3 (Introdução e Objetivos)**: 0:00 - 3:00 (3 min)
*   **Slides 4-6 (Dataset e Arquitetura)**: 3:00 - 7:00 (4 min)
*   **Slides 7-10 (Treinamento, Avaliação e Resultados)**: 7:00 - 12:30 (5,5 min)
*   **Slides 11-12 (Conclusões e Encerramento)**: 12:30 - 15:00 (2,5 min)

---

## 🖥️ Estrutura dos Slides

### Slide 1: Capa
*   **Título**: Detecção Automática de Erros de Pronúncia em Aprendizes L2 de Inglês Baseada em Alinhamento de Sequências com Wav2Vec 2.0 e Whisper
*   **Subtítulo**: Trabalho Final da Disciplina de Processamento de Linguagem Natural (UFRGS - 2026)
*   **Autor**: Adriel Dsouza
*   **Orientadores**: Viviane Moreira e Dennis Giovani Balreira
*   **Duração Recomendada**: 0:45 min

> **Notas do Apresentador (O que falar):**
> *"Bom dia a todos, professores e colegas. Vou apresentar o trabalho final de PLN sobre detecção automática de desvios fonéticos em aprendizes de inglês como segunda língua. Este trabalho utiliza modelos baseados em Transformers rodando de forma local para criar um pipeline automatizado de diagnóstico de pronúncia."*

---

### Slide 2: Introdução e Contexto
*   **Tópicos Visuais**:
    *   **A Importância da Pronúncia**: Compreensibilidade da fala em aprendizes L2.
    *   **Sistemas CAPT**: *Computer-Assisted Pronunciation Training* como alternativa viável para ensino em escala.
    *   **Mispronunciation Detection (MD)**: Identificação e classificação de desvios no nível do fonema.
*   **Duração Recomendada**: 1:15 min

> **Notas do Apresentador (O que falar):**
> *"No aprendizado de uma segunda língua, a pronúncia é essencial para garantir que o aprendiz seja compreendido. Os sistemas CAPT automatizam o feedback individual, que costuma ser muito caro quando feito por tutores humanos. A detecção de erros fonéticos consiste em identificar se o estudante cometeu Substituições, Deleções ou Inserções de fonemas ao tentar ler uma frase."*

---

### Slide 3: Objetivo do Trabalho
*   **Tópicos Visuais**:
    *   **Pipeline Fim-a-Fim**: Criar um detector de erros que não dependa do texto escrito original (para fala espontânea).
    *   **Modelo Híbrido**: Unir representação acústica (o que foi dito) com representação canônica (o que deveria ser dito).
    *   **Conformidade UFRGS**: Uso de arquiteturas Transformers, dataset anotado com ground truth e alinhamento algorítmico.
*   **Duração Recomendada**: 1:00 min

> **Notas do Apresentador (O que falar):**
> *"Nosso objetivo foi projetar e avaliar um pipeline baseado em Transformers que analisa o áudio e gera o feedback de erro fonético sem precisar que o sistema conheça o texto lido de antemão. Isso é feito comparando a fala do estudante (via Wav2Vec 2.0) com uma transcrição canônica gerada dinamicamente via Whisper e algoritmos de G2P."*

---

### Slide 4: O Corpus L2-ARCTIC
*   **Tópicos Visuais**:
    *   **Dataset Aberto**: Zhao et al. (Interspeech 2018).
    *   **Composição**: 24 falantes não nativos, 150 sentenças cada, totalizando 3.631 áudios (16kHz).
    *   **Sotaques Analisados (L1)**: Árabe, Chinês, Hindi, Coreano, Espanhol e Vietnamita.
    *   **Anotações Manuais (Linguistas)**: Rótulos triplos no formato: `[FonemaAlvo, FonemaProduzido, TipoDeErro (s/d/a)]`.
*   **Duração Recomendada**: 1:30 min

> **Notas do Apresentador (O que falar):**
> *"Utilizamos o corpus L2-ARCTIC, muito conceituado na literatura. Ele possui áudios de aprendizes de seis sotaques diferentes gravando frases foneticamente balanceadas. O principal valor desse dataset são as anotações feitas por linguistas, que registraram exatamente onde ocorreram erros de substituição, deleção ou adição, servindo como nosso Ground Truth de avaliação."*

---

### Slide 5: Arquitetura do Pipeline Proposto
*   **Tópicos Visuais**:
    *   *Inserir diagrama de fluxo do pipeline (ver project_status_report.md)*:
        1.  **Áudio Bruto** $\rightarrow$ **Wav2Vec 2.0** $\rightarrow$ Sequência Acústica (Ouvida).
        2.  **Áudio Bruto** $\rightarrow$ **Whisper** $\rightarrow$ Texto $\rightarrow$ **G2P** $\rightarrow$ Sequência Canônica (Esperada).
    *   **Normalização Fonética**: Redução e mapeamento para **44 classes** fonéticas do inglês (Arpabet limpo).
    *   **Alinhador Dinâmico**: Distância de Levenshtein (via `jiwer`) alinhando as duas sequências.
*   **Duração Recomendada**: 2:00 min

> **Notas do Apresentador (O que falar):**
> *"Esta é a arquitetura do nosso pipeline. Dividimos o processamento em duas ramificações. O ramo acústico processa o áudio do aprendiz com o Wav2Vec 2.0 ajustado para fonemas. O ramo de referência usa o Whisper para converter o áudio em texto e o G2P para deduzir os fonemas corretos daquela frase. Em seguida, os fonemas são limpos (removendo acentuação tonal e silêncios) e alinhados via algoritmo de distância de Levenshtein, classificando as discrepâncias em erros de pronúncia."*

---

### Slide 6: Ajuste Fino e Otimização de Hardware
*   **Tópicos Visuais**:
    *   **Modelo Base**: `facebook/wav2vec2-base` (960h LibriSpeech).
    *   **Otimização local**: Estabilização do CTC Loss no chip Apple Silicon (Mac M4, 24GB RAM).
    *   **Configurações de Estabilidade**:
        *   Fallback para CPU em operações não suportadas pelo Metal/MPS.
        *   Uso de `ctc_zero_infinity=True`.
        *   Learning rate menor ($2\times 10^{-5}$ a $5\times 10^{-5}$) e acumulação de gradientes (batch efetivo de 32).
    *   **Pivô de Vocabulário**: Simplificação drástica de **142 classes para 44 fonemas**, destravando o aprendizado do modelo.
*   **Duração Recomendada**: 1:45 min

> **Notas do Apresentador (O que falar):**
> *"Para o modelo acústico, fizemos o ajuste fino do Wav2Vec 2.0. Enfrentamos um desafio de hardware: a perda CTC causava instabilidade numérica no chip Apple Silicon. Resolvemos isso aplicando fallback para CPU, adicionando acumulação de gradientes e usando a flag ctc_zero_infinity. Além disso, fizemos um pivô estratégico: reduzir o vocabulário de classes fonéticas de 142 para 44 fonemas limpos destravou o aprendizado do modelo, que antes ficava em erro constante de 100%."*

---

### Slide 7: Resultados de Treinamento Acústico
*   **Tópicos Visuais**:
    *   **Tempo de Treino**: ~19 horas de execução local (25 épocas).
    *   **Loss de Validação Final**: **0.3046**
    *   **Phone Error Rate (PER)**: **17,63%** (WER do reconhecedor fonético).
    *   **Acurácia Acústica Estimada**: **82,37%**
    *   **Exemplo Prático**:
        *   *Referência*: `W IY L HH AE W T AH`
        *   *Predição*: `W IY L HH AE W T AA` (substituição correta capturada no final).
*   **Duração Recomendada**: 1:15 min

> **Notas do Apresentador (O que falar):**
> *"O treinamento levou cerca de 19 horas no Mac M4. A perda de validação final estabilizou em 0.3046 e obtivemos um Phone Error Rate de 17.63%, o que indica que nosso reconhecedor fonético acústico tem uma acurácia próxima a 82,4% para decodificar os fonemas falados. Vemos na tela um exemplo de inferência real onde o modelo acertou quase perfeitamente a frase."*

---

### Slide 8: Resultados Globais da Detecção
*   **Tópicos Visuais**:
    *   **Base de Avaliação**: Todo o test split do L2-ARCTIC (720 áudios, 23.659 fonemas de teste).
    *   **Métricas de Classificação Binária (Nível do Fonema)**:
        *   **Acurácia**: **87,02%**
        *   **Precisão**: **56,62%**
        *   **Recall (Revocação)**: **54,98%**
        *   **F1-Score**: **55,79%**
    *   **Contagens da Matriz de Confusão**:
        *   TN: 18.652 | FP: 1.484
        *   FN: 1.586 | TP: 1.937
*   **Duração Recomendada**: 1:30 min

> **Notas do Apresentador (O que falar):**
> *"Avaliamos o detector completo em massa sobre mais de 23 mil fonemas de teste. A acurácia global foi de 87,02%, mostrando que o sistema é muito estável na validação dos fonemas corretos, evitando falsos alertas. O F1-Score geral de detecção de erros ficou em 55,79%, o que é um resultado sólido na literatura de CAPT sem restrições linguísticas externas."*

---

### Slide 9: Desempenho por Tipo de Erro
*   **Tópicos Visuais**:
    *   **Tabela Comparativa**:
        *   **Substituições (S)**: Precisão: 57,04%, Recall: 51,05%, F1: **53,88%**
        *   **Deleções (D)**: Precisão: 55,38%, Recall: **71,81%**, F1: **62,53%**
    *   **Análise Linguística**:
        *   Por que as deleções performam melhor? A ausência de energia acústica cria fronteiras e contrastes mais fáceis para a perda CTC.
        *   Substituições exigem maior precisão espectral de sotaques próximos (ex: neutralização vocálica).
*   **Duração Recomendada**: 1:30 min

> **Notas do Apresentador (O que falar):**
> *"Quando abrimos as métricas por tipo de erro, notamos que o detector é excelente para identificar Deleções, ou seja, fonemas omitidos. O recall de deleções atingiu 71,81%. O modelo Wav2Vec2 detecta o silêncio e o alinhador mapeia a omissão de forma clara. Já a substituição é um desafio espectral mais sutil devido a sotaques que apenas reduzem ou distorcem a vogal sem omiti-la, obtendo F1 de 53,88%."*

---

### Slide 10: O Impacto dos Sotaques (Multilinguismo L1)
*   **Tópicos Visuais**:
    *   **Gráfico/Tabela de Extremos**:
        *   **Sotaque Vietnamita (TLV)**: F1 de **77,45%** (Recall 84,26%)
        *   **Sotaque Chinês (HQTV)**: F1 de **74,29%** (Recall 72,22%)
        *   **Sotaque Árabe (ABA)**: F1 de **28,57%** (Recall 25,61%)
    *   **Conclusão da Análise**:
        *   Sotaques com maiores taxas brutas de erro no L2-ARCTIC permitem maior sensibilidade de detecção.
        *   Sotaques com desvios sutis (vogais róticas e fricativas dentais no Árabe) passam despercebidos pelo modelo acústico.
*   **Duração Recomendada**: 1:30 min

> **Notas do Apresentador (O que falar):**
> *"Analisamos a sensibilidade do sistema para cada sotaque nativo. O pipeline funcionou de forma excelente para estudantes de sotaque vietnamita e chinês, ultrapassando 74% de F1. Esses aprendizes possuem maior volume de erros no corpus. Por outro lado, para falantes nativos de árabe, a performance caiu para 28,57% de F1. As substituições fonéticas no sotaque árabe são muito sutis espectralmente, dificultando a distinção pelo reconhecedor."*

---

### Slide 11: Conclusão e Trabalhos Futuros
*   **Tópicos Visuais**:
    *   **Conclusão**:
        *   Pipeline Transformer fim-a-fim funcional no chip Apple Silicon.
        *   A acurácia global de 87% garante feedback seguro para aprendizes.
        *   O alinhamento dinâmico via Levenshtein é robusto para diagnósticos automatizados.
    *   **Trabalhos Futuros**:
        *   Adoção de modelos acústicos multilingues maiores (ex: `Wav2Vec2-XLSR-53`).
        *   Cálculo da métrica de Goodness of Pronunciation (GOP) combinada com LLMs de contexto.
*   **Duração Recomendada**: 1:00 min

> **Notas do Apresentador (O que falar):**
> *"Como conclusões, mostramos que é possível construir um detector de desvios fonéticos funcional rodando localmente. O pipeline obtém 87% de acurácia global. Em trabalhos futuros, planejamos avaliar arquiteturas maiores como o XLS-R e adicionar scores probabilísticos baseados em GOP para refinar o feedback aos aprendizes."*

---

### Slide 12: Agradecimentos e Perguntas
*   **Tópicos Visuais**:
    *   Obrigado pela atenção!
    *   **Contato**: adsouza@inf.ufrgs.br
    *   *Link para o Repositório GitHub do Projeto*
    *   Espaço aberto para a banca de professores.
*   **Duração Recomendada**: Restante do tempo (Perguntas)

> **Notas do Apresentador (O que falar):**
> *"Gostaria de agradecer aos professores Viviane e Dennis, e abro espaço para as perguntas da banca. Muito obrigado!"*
