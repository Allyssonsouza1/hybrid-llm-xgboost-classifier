# hybrid-llm-xgboost-classifier
# Classificação de Subsegmentos Food Service

## Sobre o Projeto
Este projeto aplica técnicas de **Aprendizagem de Máquina Supervisionada** e **NLP** para classificar estabelecimentos do setor alimentício em subsegmentos a partir de dados cadastrais, textuais e fontes externas. Os subsegmentos-alvo são:

- **CASH AND CARRY** (atacarejos)
- **AUTO SERVICO** (supermercados/minimercados)
- **LANCHONETE**
- **CONFEITARIA**
- **PADARIA**

O desenvolvimento seguiu as fases da metodologia **CRISP-DM**, garantindo alinhamento entre as necessidades do negócio e o rigor técnico. O principal desafio reside na alta similaridade semântica entre os nomes das empresas (ex: um grande atacarejo e um supermercado de bairro frequentemente compartilham vocabulário idêntico).

### Resultados do melhor Modelo 
* **Acurácia (Teste Off-Sample):** `0.8468`
* **F1-Score Macro (Cross-Validation):** `0.8715`
* **F1-Score Macro (Teste):** `0.8113`
* **Modelo Final:** XGBoost otimizado via Grid Search com Stratified K-Fold (5 Folds)

> **Nota:** Os dados originais pertencem a um case técnico e não estão publicamente disponíveis.

## Tecnologias e Ferramentas
| Categoria | Tecnologias Utilizadas |
| :--- | :--- |
| **Linguagem** | Python 3.14.2 |
| **Manipulação de Dados** | Pandas, Numpy |
| **Visualização** | Matplotlib, Seaborn |
| **NLP** | Bag of Words, TF-IDF, LLM (GPT-4o-mini) |
| **Modelagem** | Scikit-learn (Decision Tree, Random Forest), XGBoost |
| **Validação** | Stratified K-Fold Cross-Validation, Grid Search |
| **Artefatos** | PyArrow (Arquivos .parquet), Pickle (.pkl) |

## Metodologia

O projeto percorre todas as etapas do CRISP-DM:

1. **Entendimento do Negócio:** Definição do problema de classificação multiclasse em 5 subsegmentos do setor food service.

2. **Entendimento dos Dados:** Análise exploratória de 7 fontes de dados distintas (cadastro empresarial, CNAE, Google Places, iFood, dívida ativa), identificando alta incidência de dados faltantes e desbalanceamento natural das classes.

3. **Preparação dos Dados:**
   - **Tratamento de Chaves (CNPJ):** Correção de notação científica, artefatos float e padronização de zeros à esquerda para garantir integridade nos joins.
   - **Limpeza Textual (NLP):** Padronização de strings, remoção de caracteres especiais e stopwords customizadas.
   - **Feature Engineering (Bag of Words):** Construção de dicionário com termos de domínio para converter texto em vetores numéricos baseados na frequência de palavras-chave por classe.
   - **Feature Engineering (LLM):** Utilização do GPT-4o-mini como extrator semântico para pré-classificar registros, substituindo o Bag of Words por uma interpretação contextual do texto (razão social, nome fantasia, categoria e descrição CNAE).
   - **Tratamento Anti-Leakage:** Imputação do `capital_social` via mediana agrupada pela divisão do CNAE, evitando vazamento de informações.

4. **Modelagem:** Avaliação iterativa de 3 famílias de algoritmos baseados em árvores de decisão, culminando na otimização via Grid Search.

5. **Avaliação:** Validação final com dados nunca vistos (off-sample), confirmando a robustez e capacidade de generalização.

### Detalhamento: Bag of Words vs LLM

A extração de features textuais foi abordada por duas estratégias distintas ao longo do projeto:

**Bag of Words (Abordagem Clássica)**
Construí manualmente um dicionário de termos discriminantes por subsegmento (ex: "atacado", "cash", "carry" para Cash and Carry; "padaria", "panificadora" para Padaria). A partir dele, cada registro recebeu variáveis binárias indicando a presença ou ausência desses termos em campos como `nome_fantasia`, `razao_social` e `desc_subclasse`. Essa abordagem foi adotada porque o tamanho reduzido da base (~1.700 registros) inviabilizava técnicas automatizadas puras como TF-IDF(que foi testado anteriormente), sendo necessário injetar conhecimento de domínio diretamente na engenharia de features.

**LLM como Extrator Semântico (Abordagem Experimental)**
Na etapa de evolução (Arquitetura Híbrida V3), substituí o dicionário manual pelo GPT-4o-mini atuando como pré-classificador. O modelo recebia os campos textuais de cada registro (razão social, nome fantasia, categoria principal e descrição CNAE) e retornava uma classificação semântica. A hipótese era que a LLM conseguiria interpretar contextos ambíguos que a contagem de palavras não captura (ex: diferenciar "Atacadão" como Cash and Carry de "Supermercado Atacadão" como Auto Serviço). Na prática, a ausência de dados em campos-chave limitou a performance, pois o modelo operava "às cegas" em registros incompletos.

## Comparativo de Performance dos Modelos

| Modelo | Features | Acurácia (Treino) | Acurácia (Valid.) | Acurácia (Teste) | F1-Macro (Treino) | F1-Macro (Valid.) | F1-Macro (Teste) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Decision Tree (M1 - Baseline) | Apenas Texto | 0.8013 | 0.8498 | N/A | 0.8184 | 0.8670 | N/A |
| Decision Tree (M2) | Texto + Capital Social | 0.8085 | 0.8466 | N/A | 0.8313 | 0.8649 | N/A |
| Decision Tree (M3) | Texto + Capital + CNAE | 0.8069 | 0.8498 | N/A | 0.8287 | 0.8725 | N/A |
| Random Forest | Apenas Texto | 0.8013 | 0.8498 | N/A | 0.8184 | 0.8670 | N/A |
| XGBoost Baseline | Apenas Texto | 0.8005 | 0.8498 | N/A | 0.8177 | 0.8670 | N/A |
| **XGBoost (Tuned)** | **Apenas Texto** | **N/A** | **N/A** | **0.8468** | **0.8715** | **0.8715** | **0.8113** |
| Híbrida V3 (XGBoost + LLM) | Num. + Semântica LLM | 0.8261 | 0.8211 | N/A | 0.8505 | 0.8644 | N/A |

---

## Conclusão e Modelo Final

O **XGBoost (Tuned)** (otimizado via Grid Search) foi selecionado como modelo de produção pelos seguintes motivos:

- **Estabilidade Comprovada:** O Cross-Validation com 5 folds confirmou F1-Score médio de `0.8715` com margem de erro de apenas 0.0164, descartando overfitting.
- **Ganho Real:** Elevação de 4.58 pontos percentuais em relação ao baseline (de `0.8257` para `0.8715`) através do ajuste de hiperparâmetros (`n_estimators: 200`, `max_depth: 5`, `learning_rate: 0.1`).
- **Generalização:** Performance consistente na base de teste off-sample (Acurácia `0.85`, F1-Score `0.87`), validando a capacidade do modelo em dados nunca vistos.

### Análise por Categoria
- **Lanchonete (Recall 1.00):** Todas as lanchonetes da base de teste foram corretamente identificadas.
- **Auto Serviço (Precision 0.94):** Alta confiança nas predições desta classe.
- **Cash and Carry (Recall 0.62):** Principal desafio, onde atacadistas são confundidos com varejo comum devido à similaridade textual.

### LLM

Apliquei o GPT-4o-mini como extrator semântico (Arquitetura Híbrida V3). A ideia era testar se a IA conseguiria interpretar contextos ambíguos (Cash and Carry vs Auto Serviço). Embora estável, a ausência de dados em campos-chave limitou a performance da LLM, não superando o modelo clássico otimizado.

### Próximos Passos

1. **Enriquecimento Físico/Financeiro:** Inserir dados como área da loja (m2), número de checkouts ou faixa de faturamento presumido.
2. **Ampliação da Base:** Aumentar o volume de dados rotulados para permitir abordagens mais robustas de NLP (ex: TF-IDF, embeddings).

## Estrutura de Diretórios
```text
Driva_case_Allysson_souza/
├── food_places.ipynb              # Notebook principal 
├── dicionario.md                  # Dicionário de dados detalhado
├── modelo_xgb_campeao.pkl         # Modelo XGBoost treinado 

```
