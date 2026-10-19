# Customer Churn Prediction

Projeto de Ciência de Dados e Machine Learning para previsão de cancelamento de clientes no setor de telecomunicações.

## 1. Visão geral

Este projeto simula um ambiente corporativo da empresa fictícia **Conecta Brasil Telecom**, com o objetivo de identificar clientes com maior probabilidade de cancelar seus serviços nos próximos 30 dias.

O projeto foi desenvolvido para fins educacionais e composição de portfólio profissional, utilizando dados sintéticos.

## 2. Problema de negócio

O cancelamento de clientes (*customer churn*) representa um desafio para empresas de telecomunicações.

A identificação antecipada de clientes com maior risco pode apoiar a priorização de ações de retenção e a alocação de recursos comerciais.

O modelo fornece estimativas probabilísticas e não substitui a avaliação das equipes responsáveis.

## 3. Objetivo analítico

Estimar a probabilidade de cancelamento efetivo do serviço nos próximos 30 dias, considerando informações disponíveis na data de referência.

## 4. Tecnologias utilizadas

* Python
* Pandas e NumPy
* Scikit-learn
* PostgreSQL e SQL
* Matplotlib e Seaborn
* Jupyter Notebook
* Pytest
* Joblib
* Git e GitHub

## 5. Dados utilizados

Foi gerado um conjunto sintético contendo **1.000 clientes**, com as seguintes variáveis utilizadas no modelo:


| Variável                           | Descrição                                     |
| ----------------------------------- | ----------------------------------------------- |
| `<span>total_atendimentos</span>`   | Quantidade total de atendimentos                |
| `<span>atendimentos_abertos</span>` | Quantidade de atendimentos ainda abertos        |
| `<span>media_satisfacao</span>`     | Média de satisfação do cliente               |
| `<span>cancelou</span>`             | Variável-alvo: 0 = não cancelou; 1 = cancelou |

A taxa de cancelamento observada no conjunto sintético foi de **13,10%**.

Os dados não representam clientes reais.

## 6. Metodologia

O desenvolvimento contemplou:

1. Definição do problema de negócio e da variável-alvo.
2. Modelagem e consultas SQL em PostgreSQL.
3. Geração e validação de dados sintéticos.
4. Análise exploratória e preparação dos dados.
5. Separação estratificada entre treino, validação e teste.
6. Treinamento de um modelo de Regressão Logística.
7. Seleção de um limiar de classificação (*threshold*) utilizando validação.
8. Avaliação do modelo.
9. Interpretação dos coeficientes.
10. Persistência do modelo com Joblib.
11. Implementação de previsões com validação de entrada.
12. Testes automatizados com Pytest.

## 7. Resultados do modelo

**Modelo:** Regressão Logística

**Threshold selecionado:** 0,20

Resultados obtidos no conjunto de teste:


| Métrica          | Resultado |
| ----------------- | --------- |
| Acurácia         | 76,00%    |
| Precisão         | 28,00%    |
| Recall            | 53,85%    |
| F1-score          | 36,84%    |
| ROC-AUC           | 0,7465    |
| Average Precision | 0,3567    |

O conjunto de teste utilizado na avaliação final continha 200 registros.

O threshold de 0,20 foi escolhido para melhorar o equilíbrio entre precisão e recall no conjunto de validação, considerando a classe de cancelamento.

Esses resultados são experimentais e não demonstram desempenho em produção.

## 8. Previsões e validação

O projeto contém funcionalidades para:

* Carregar o modelo treinado.
* Estimar a probabilidade de cancelamento.
* Classificar clientes utilizando o threshold configurado.
* Validar os dados recebidos.
* Rejeitar entradas inconsistentes.

O modelo salvo está localizado em:

`<span>models/modelo_churn_logistico.joblib</span>`

## 9. Como executar

### Pré-requisitos

* Python compatível com as dependências do projeto.
* Git.
* Ambiente virtual Python recomendado.

### Instalação

Clone o repositório:

```
git clone https://github.com/edusousa78/customer-churn-prediction.git
cd customer-churn-prediction
```

Crie e ative um ambiente virtual no Windows PowerShell:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```
python -m pip install -r requirements.txt
```

### Executar os testes

```
python -m pytest -v
```

### Executar as previsões de demonstração

```
python src/prever_churn_validado.py
```

O script utiliza exemplos fictícios para demonstrar as previsões e a validação de entrada.

## 10. Testes automatizados

Foram implementados testes com Pytest para verificar a validação dos dados e a integração com o modelo.

**Último resultado registrado: 11 testes aprovados.**

## 11. Limitações

* Os dados utilizados são sintéticos.
* O modelo não foi validado com dados reais de uma operadora.
* As variáveis foram simuladas e não representam necessariamente a complexidade operacional de uma empresa.
* A avaliação não comprova retorno financeiro ou eficácia de campanhas de retenção.
* Durante o desenvolvimento didático, houve experimentação anterior com o conjunto de teste; portanto, as métricas não devem ser interpretadas como uma avaliação independente definitiva.
* Uma implantação real exigiria validação temporal, monitoramento, avaliação de custos e controles de governança de dados.

## 12. Finalidade

Este repositório integra um portfólio profissional em Ciência de Dados e demonstra competências em Python, SQL, Machine Learning, validação de dados, testes automatizados e versionamento de código.

## 13. Aviso

A Conecta Brasil Telecom é uma empresa fictícia. Este projeto não utiliza dados pessoais reais e não constitui um sistema de decisão comercial validado para produção.
