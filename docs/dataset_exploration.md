# Exploração do Dataset L2-ARCTIC

O dataset L2-ARCTIC contém anotações detalhadas de erros de pronúncia em arquivos `.TextGrid`. 

## Estrutura de Arquivos
Cada falante (ex: `ABA`, `SKI`) possui a seguinte estrutura:
- `annotation/`: Arquivos `.TextGrid` com anotações de palavras e fonemas.
- `wav/`: Arquivos de áudio `.wav` originais.
- `transcript/`: Transcrições textuais das frases.
- `textgrid/`: Arquivos `.TextGrid` gerados automaticamente (podem diferir das anotações manuais).

## Formato das Anotações de Erro
As anotações de erro encontram-se no tier **'phones'** dos arquivos `.TextGrid` na pasta `annotation/`.

O formato das etiquetas de intervalo quando ocorre um erro é:
`[FonemaAlvo],[FonemaProduzido],[TipoDeErro]`

### Tipos de Erro:
- **`s` (Substitution):** O falante substituiu o fonema alvo por outro.
    - Ex: `ER0,AH0,s` (Alvo: ER0, Produzido: AH0)
    - Ex: `P, B, s` (Alvo: P, Produzido: B)
- **`d` (Deletion):** O falante omitiu o fonema alvo. Geralmente o fonema produzido é marcado como `sil`.
    - Ex: `R,sil,d` (Alvo: R deletado)
- **`a` (Addition/Insertion):** O falante inseriu um fonema extra. Geralmente o fonema alvo é marcado como `sil`.
    - Ex: `sil,K,a` (Inserção de K)

### Observações sobre os Símbolos:
- São utilizados tanto símbolos **Arpabet** (ex: `AA1`, `ER0`, `DH`) quanto **IPA** (ex: `ə`, `ɪ`, `ɹ`).
- Algumas anotações possuem espaços extras (ex: `P, B, s`) ou vírgulas extras.
- Existem marcações de fonemas "normais" (sem erro) que são apenas o símbolo do fonema (ex: `EH1`, `T`).

## Distribuição de Erros por Falante (Amostra de 50 frases/falante)

Com base na análise inicial, observamos as seguintes taxas de erro (Substituições + Deleções + Adições):

| Falante | Total Fonemas | Erros (S+D+A) | Taxa de Erro |
| :--- | :--- | :--- | :--- |
| **THV** | 1880 | 460 | 24.4% |
| **HQTV** | 1895 | 440 | 23.2% |
| **TLV** | 1970 | 427 | 21.6% |
| **EBVS** | 2048 | 397 | 19.3% |
| **ASI** | 1842 | 319 | 17.3% |
| **ZHAA** | 1910 | 197 | 10.3% |
| **ABA** | 1858 | 155 | 8.3% |

Os falantes **THV**, **HQTV** e **TLV** apresentam as maiores taxas de erro, o que os torna candidatos ideais para testar a sensibilidade do modelo de detecção.

## Desafios para o Parser:

1.  **Normalização de Símbolos:** Será necessário converter IPA para Arpabet (ou vice-versa) para manter a consistência, ou focar apenas no tipo de erro ignorando a representação.
2.  **Limpeza de Strings:** Remover espaços em branco e tratar variações na formatação das etiquetas.
3.  **Mapeamento de Tempo:** Relacionar os intervalos de tempo com os frames do modelo Wav2vec 2.0.
