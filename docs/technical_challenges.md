# Relatório de Gargalos Técnicos e Lições Aprendidas - Projeto TF-PLN

Este documento registra os principais desafios enfrentados durante o ajuste fino (fine-tuning) do modelo Wav2Vec2 para detecção de fonemas no dataset L2-ARCTIC, bem como as soluções aplicadas e os resultados finais.

## 1. Instabilidade do CTC Loss no Mac (MPS)
**Problema:** O treinamento sofria falhas catastróficas constantes (`nan` loss e `nan` gradients) ao rodar no chip Apple Silicon (M1/M2/M3).
**Causa:** A operação `ctc_loss` não está totalmente implementada para o backend MPS (Metal Performance Shaders). O fallback automático para CPU, combinado com um learning rate padrão, criava instabilidades numéricas.
**Solução:**
- Ativação explícita do fallback: `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`.
- Uso de `ctc_zero_infinity=True` na configuração do modelo para evitar que valores infinitos corrompam os pesos.
- Redução drástica do learning rate (inicialmente para `2e-5`) e aplicação de `gradient_accumulation_steps=2`.

## 2. Incompatibilidade de Versões (Transformers v5)
**Problema:** O código seguia padrões antigos da biblioteca `transformers`, resultando em `AttributeError` e `TypeError`.
**Gargalos Específicos:**
- `as_target_processor()` foi removido. Substituído por chamadas diretas ao `processor(text=...)`.
- `TrainingArguments` mudou os nomes de parâmetros (`evaluation_strategy` -> `eval_strategy`).
- `Trainer` agora exige o parâmetro `processing_class` em vez de `tokenizer` para modelos de áudio.

## 3. Integridade dos Dados e Tokenização
**Problema:** O modelo parava de aprender (loss 0) ou explodia logo no início.
**Causa:** 
- Presença de registros com anotações vazias no dataset L2-ARCTIC (ex: `arctic_a0209` e `arctic_a0272`).
- Divergência entre o delimitador do vocabulário (`|`) e o delimitador enviado pelo script de carga (espaço).
**Solução:**
- Implementação de um filtro no `load_data` para descartar registros sem fonemas produzidos.
- Alinhamento do separador de fonemas para `|` para garantir que o CTC identifique as fronteiras corretamente.

## 4. Estratégia de Pivot: Simplificação de Vocabulário
Após observar que o modelo não conseguia aprender com o vocabulário original (142 tokens), realizamos uma simplificação estratégica:
- **Redução de Classes:** De 142 para **44 tokens**.
- **Consolidação:** Remoção de acentos tônicos (`AA0`, `AA1` -> `AA`) e limpeza de símbolos de anotação.
- **Resultado:** O modelo passou de "silêncio total" para transcrições precisas em poucas épocas.

## 5. Resultados Finais e Performance de Hardware
O treinamento final foi otimizado para o hardware **Mac M4 (10 cores CPU, 24GB RAM)**.

| Métrica | Resultado |
|-----------|-------------|
| **Tempo de Treino Total** | ~19 horas (acumuladas) |
| **Épocas Concluídas** | 25 épocas |
| **Batch Size Efetivo** | 32 (16 por device + 2 grad accumulation) |
| **Loss de Validação Final** | **0.3046** |
| **Word Error Rate (WER)** | **0.1763 (17.6%)** |
| **Acurácia Fonética Estimada** | **82.4%** |

### Observação Qualitativa:
Na 25ª época, o modelo atingiu o "ponto de maturidade", sendo capaz de transcrever sentenças complexas como:
- **Referência:** `W IY L HH AE W T AH W AO CH AW R CH AE N S AH S`
- **Predição:** `W IY L HH AE W T AH W AA CH AW R CH AE N S AH S`

## Lições Aprendidas:
1.  **Simplificação de Domínio:** Em tarefas complexas de PLN, reduzir o espaço de busca (vocabulário) é mais eficaz do que aumentar o tempo de treino.
2.  **Otimização de Hardware Local:** Configurações como `num_proc=8` e `dataloader_num_workers=4` são essenciais para viabilizar treinos de Deep Learning em máquinas locais Pro/Max.
3.  **Persistência em Pesquisa:** A queda inicial de 100% de erro para 17% prova que modelos de fala exigem uma "fase de latência" longa antes de mostrarem resultados visíveis.
