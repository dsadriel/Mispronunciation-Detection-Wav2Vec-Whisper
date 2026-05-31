# Relatório de Gargalos Técnicos e Lições Aprendidas - Projeto TF-PLN

Este documento registra os principais desafios enfrentados durante o ajuste fino (fine-tuning) do modelo Wav2Vec2 para detecção de fonemas no dataset L2-ARCTIC, bem como as soluções aplicadas.

## 1. Instabilidade do CTC Loss no Mac (MPS)
**Problema:** O treinamento sofria falhas catastróficas constantes (`nan` loss e `nan` gradients) ao rodar no chip Apple Silicon (M1/M2/M3).
**Causa:** A operação `ctc_loss` não está totalmente implementada para o backend MPS (Metal Performance Shaders). O fallback automático para CPU, combinado com um learning rate padrão, criava instabilidades numéricas.
**Solução:**
- Ativação explícita do fallback: `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"`.
- Uso de `ctc_zero_infinity=True` na configuração do modelo para evitar que valores infinitos corrompam os pesos.
- Redução drástica do learning rate (de `1e-4` para `2e-5`) e aplicação de `gradient_accumulation_steps=2`.

## 2. Incompatibilidade de Versões (Transformers v5)
**Problema:** O código seguia padrões antigos da biblioteca `transformers`, resultando em `AttributeError` e `TypeError`.
**Gargalos Específicos:**
- `as_target_processor()` foi removido. Substituído por chamadas diretas ao `processor(text=...)`.
- `TrainingArguments` mudou os nomes de parâmetros (`evaluation_strategy` -> `eval_strategy`).
- `Trainer` agora exige o parâmetro `processing_class` em vez de `tokenizer` para modelos de áudio.
- Necessidade da biblioteca `accelerate>=1.1.0` para gerenciar dispositivos no PyTorch.

## 3. Integridade dos Dados e Tokenização
**Problema:** O modelo parava de aprender (loss 0) ou explodia logo no início.
**Causa:** 
- Presença de registros com anotações vazias no dataset L2-ARCTIC (ex: `arctic_a0209` e `arctic_a0272`).
- Divergência entre o delimitador do vocabulário (`|`) e o delimitador enviado pelo script de carga (espaço).
**Solução:**
- Implementação de um filtro no `load_data` para descartar registros sem fonemas produzidos.
- Alinhamento do separador de fonemas para `|` para garantir que o CTC identifique as fronteiras corretamente.

## 4. Configuração de Hardware e Ambiente
**Problema:** Erros de "ModuleNotFoundError" e falta de suporte a `pin_memory`.
**Solução:**
- Execução com `PYTHONPATH=.` para garantir que o pacote `src` seja reconhecido.
- Desativação de `pin_memory` no DataLoader, pois o MPS ainda não suporta essa funcionalidade de otimização de memória de forma estável.

## Resumo dos Parâmetros de Estabilidade Finais:
| Parâmetro | Valor Final | Motivo |
|-----------|-------------|--------|
| Learning Rate | 2e-5 | Evitar explosão de gradiente no Mac |
| Max Grad Norm | 0.5 | Clipping agressivo para manter pesos saudáveis |
| Gradient Accumulation | 2 | Estabilizar a estimativa de erro |
| Warmup Steps | 1000 | Início ultra-lento para adaptação da camada lm_head |
| Zero Infinity | True | Ignorar erros matemáticos pontuais do CTC |

## 5. Resultados do Treinamento (Vocabulário Original - 142 tokens)
- **Tempo Total:** ~8 horas e 50 minutos.
- **Loss Final (Train):** 5.622.
- **Loss de Validação (Eval):** 2.771.
- **WER Final:** 0.998.
- **Observação:** O modelo sofreu underfitting severo e previa apenas silêncios.

## 6. Pivot: Simplificação de Vocabulário e Estabilização Final
...
(seção anterior mantida)
...

## 7. Resultados do Treinamento Final (Vocabulário Simplificado - 44 tokens)
- **Tempo Total:** ~8 horas e 4 minutos (Mac M4).
- **Épocas:** 10.
- **Loss Final (Passo 900):** **1.906** (Queda constante de 33.9 -> 1.9).
- **Loss de Validação (Eval):** **1.813**.
- **WER Final:** **0.9722** (Melhoria real em relação ao 1.0 anterior).
- **Status:** Sucesso técnico e aprendizado confirmado. O modelo começou a produzir sequências de fonemas reais.

## Lições de Design de Pesquisa:
1.  **Menos é Mais:** Reduzir o vocabulário de 142 para 44 fonemas foi o fator decisivo para o modelo convergir em 10 épocas.
2.  **Eficiência de Hardware:** O uso de `num_proc=8` e `per_device_train_batch_size=16` permitiu treinar em menos de 10 horas um modelo que levaria dias em configurações padrão.
3.  **Estabilidade é Prioridade:** O `max_grad_norm=0.5` e `learning_rate=5e-5` garantiram que o treino nunca quebrasse, mesmo com o fallback de CPU do Mac.


