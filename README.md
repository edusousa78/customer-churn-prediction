# Customer Churn Prediction — Conecta Brasil Telecom

Projeto de Ciência de Dados, Machine Learning e Engenharia de Software para estimar o risco de cancelamento de clientes no setor de telecomunicações.

**Status:** Projeto educacional funcional, com API publicada, testes automatizados e integração contínua.

> **Aviso:** A Conecta Brasil Telecom é uma empresa fictícia. Todos os dados utilizados são sintéticos. O modelo não foi validado para decisões comerciais reais.

## 1. Visão geral

O projeto simula um problema corporativo de retenção de clientes: identificar clientes com maior probabilidade de cancelar seus serviços nos próximos 30 dias.

A solução contempla preparação de dados, análise exploratória, treinamento e avaliação de um modelo de Regressão Logística, interpretação das previsões, API REST, containerização, testes automatizados e documentação técnica.

### Objetivos

* Estimar a probabilidade de churn por cliente.
* Classificar clientes utilizando um threshold configurável no artefato do modelo.
* Identificar fatores associados às previsões.
* Comparar cenários sintéticos por análise de sensibilidade.
* Disponibilizar previsões por meio de uma API.
* Demonstrar boas práticas de desenvolvimento e versionamento.

## 2. Problema de negócio

O cancelamento de clientes (*customer churn*) pode representar perda de receita e aumento dos custos de aquisição de novos clientes.

Uma estimativa antecipada de risco pode apoiar a priorização de análises e iniciativas de retenção.

O modelo fornece estimativas probabilísticas e não substitui avaliações humanas, análises econômicas ou validação operacional.

## 3. Tecnologias utilizadas


| Categoria               | Tecnologias                 |
| ----------------------- | --------------------------- |
| Linguagem               | Python                      |
| Manipulação de dados  | Pandas, NumPy               |
| Machine Learning        | Scikit-learn                |
| Banco de dados          | PostgreSQL, SQL             |
| Visualização          | Matplotlib, Seaborn         |
| Experimentação        | Jupyter Notebook            |
| API                     | FastAPI, Uvicorn, Pydantic  |
| Persistência do modelo | Joblib                      |
| Testes                  | Pytest                      |
| Qualidade de código    | Ruff                        |
| Containerização       | Docker, Docker Compose      |
| Versionamento e CI      | Git, GitHub, GitHub Actions |
| Hospedagem              | Render                      |

## 4. Dados utilizados

O conjunto de dados de modelagem contém **1.000 registros sintéticos**.

### Variáveis utilizadas no modelo


| Variável                           | Descrição                                     |
| ----------------------------------- | ----------------------------------------------- |
| `<span>total_atendimentos</span>`   | Quantidade total de atendimentos do cliente     |
| `<span>atendimentos_abertos</span>` | Quantidade de atendimentos ainda abertos        |
| `<span>media_satisfacao</span>`     | Média de satisfação, na escala de 1 a 5      |
| `<span>churn_30_dias</span>`        | Variável-alvo: 0 = não cancelou; 1 = cancelou |

A taxa de churn no conjunto sintético é de **13,10%**.

O arquivo utilizado no treinamento está localizado em:

`<span>data/processed/customer_churn_sintetico.csv</span>`

O projeto também contempla exercícios de modelagem e consultas SQL em PostgreSQL.

## 5. Metodologia

O fluxo de desenvolvimento incluiu:

1. Definição do problema e da variável-alvo.
2. Geração e validação de dados sintéticos.
3. Exploração dos dados.
4. Preparação das variáveis.
5. Separação estratificada entre treino, validação e teste.
6. Treinamento de Regressão Logística.
7. Seleção do threshold com base no conjunto de validação.
8. Avaliação com métricas de classificação.
9. Interpretação dos coeficientes e previsões individuais.
10. Persistência e validação do artefato do modelo.
11. Desenvolvimento de API REST.
12. Implementação de testes automatizados.
13. Containerização e publicação.
14. Análise de sensibilidade e testes de consistência.

## 6. Resultados do modelo

**Algoritmo:** Regressão Logística
**Threshold:** `<span>0.20</span>`
**Conjunto de teste:** 200 registros sintéticos


| Métrica          | Resultado |
| ----------------- | --------- |
| Acurácia         | 76,00%    |
| Precisão         | 28,00%    |
| Recall            | 53,85%    |
| F1-score          | 36,84%    |
| ROC-AUC           | 0,7465    |
| Average Precision | 0,3567    |

### Matriz de confusão


| Resultado             | Quantidade |
| --------------------- | ---------- |
| Verdadeiros negativos | 138        |
| Falsos positivos      | 36         |
| Falsos negativos      | 12         |
| Verdadeiros positivos | 14         |

O recall de 53,85% indica que o modelo identificou aproximadamente 54% dos cancelamentos no conjunto de teste.

A precisão de 28% indica que uma parcela relevante dos alertas não corresponde a cancelamentos efetivos.

Esses resultados são experimentais e não representam desempenho comprovado em produção.

## 7. API REST

A API foi desenvolvida com FastAPI e disponibilizada no Render.

**URL pública:**
[https://conecta-brasil-churn-api.onrender.com](https://conecta-brasil-churn-api.onrender.com/)

**Documentação Swagger:**
[https://conecta-brasil-churn-api.onrender.com/docs](https://conecta-brasil-churn-api.onrender.com/docs)

### Verificação de funcionamento

`<span>GET /health</span>`

Resposta esperada:

```
{
  "status": "ok",
  "mensagem": "API funcionando corretamente"
}
```

### Previsão de churn

`<span>POST /prever</span>`

Exemplo de requisição:

```
{
  "total_atendimentos": 7,
  "atendimentos_abertos": 4,
  "media_satisfacao": 1.5
}
```

Exemplo de resposta:

```
{
  "probabilidade_churn": 0.9307,
  "classificacao": 1,
  "alerta": true,
  "threshold": 0.2
}
```

A API valida os dados de entrada e rejeita solicitações inconsistentes.

O ambiente de hospedagem gratuito pode apresentar um tempo de inicialização após períodos de inatividade.

## 8. Interpretabilidade do modelo

O projeto inclui análise dos coeficientes da Regressão Logística e explicações individuais das previsões.

Script:

`<span>src/explicar_previsao_individual.py</span>`

A explicação apresenta:

* Valores das variáveis do cliente.
* Coeficientes do modelo.
* Contribuições individuais para o logit.
* Intercepto e logit total.
* Probabilidade estimada de churn.
* Threshold e classificação final.

As contribuições explicam o cálculo matemático do modelo e não estabelecem relações causais.

## 9. Análise de sensibilidade

Script:

`<span>src/analisar_sensibilidade.py</span>`

O projeto compara quatro cenários sintéticos para um mesmo cliente.


| Cenário                                   | Probabilidade estimada |
| ------------------------------------------ | ---------------------- |
| Cliente original                           | 93,07%                 |
| Satisfação aumentada                     | 72,65%                 |
| Menos atendimentos abertos                 | 51,18%                 |
| Satisfação aumentada e menos pendências | 17,17%                 |

Esses cenários demonstram como o modelo reage a alterações nas variáveis de entrada.

As diferenças representam simulações matemáticas, não efeitos comprovados de intervenções reais.

## 10. Como executar localmente

### Pré-requisitos

* Python 3.13 e dependências compatíveis.
* Git.
* Docker Desktop, opcional para execução containerizada.

### Clonar o repositório

```
git clone https://github.com/edusousa78/customer-churn-prediction.git
cd customer-churn-prediction
```

### Criar ambiente virtual no Windows PowerShell

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalar dependências

```
python -m pip install -r requirements.txt
```

### Executar previsão local

```
python src/prever_churn_validado.py
```

### Executar explicação individual

```
python src/explicar_previsao_individual.py
```

### Executar análise de sensibilidade

```
python src/analisar_sensibilidade.py
```

### Iniciar a API localmente

```
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

A documentação estará disponível em:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 11. Docker

O repositório contém `<span>Dockerfile</span>` e `<span>compose.yaml</span>`.

Para construir a imagem e iniciar a aplicação:

```
docker compose up --build
```

Para encerrar os serviços:

```
docker compose down
```

A API poderá ser acessada na porta configurada pelo Docker Compose.

## 12. Testes automatizados e qualidade

O projeto utiliza Pytest para verificar validações, previsões, integridade de artefatos, API, interpretabilidade, análise de sensibilidade e consistência das previsões.

**Última execução local registrada: 36 testes aprovados.**

### Executar os testes

```
python -m pytest -q
```

### Executar análise estática

```
python -m ruff check .
```

## 13. Integração contínua

O projeto utiliza GitHub Actions para executar verificações automatizadas a partir das alterações enviadas ao repositório.

O workflow está localizado em:

`<span>.github/workflows/tests.yml</span>`

O pipeline inclui verificações de qualidade, testes e validações relacionadas à construção e execução da aplicação containerizada.

As execuções podem ser acompanhadas em:

[https://github.com/edusousa78/customer-churn-prediction/actions](https://github.com/edusousa78/customer-churn-prediction/actions)

## 14. Documentação adicional

* `<span>MODEL_CARD.md</span>`: características, avaliação e limitações do modelo.
* `<span>RELATORIO_EXECUTIVO.md</span>`: resultados e interpretação para o contexto de negócio.
* `<span>notebooks/</span>`: exploração e experimentação.
* `<span>reports/figures/</span>`: gráficos e visualizações.
* `<span>metadata/</span>`: metadados do modelo.

## 15. Limitações e próximos passos

* Os dados são sintéticos e não representam clientes reais.
* O modelo não foi validado em uma operadora de telecomunicações.
* O conjunto de teste foi consultado durante o desenvolvimento educacional; portanto, as métricas não constituem uma avaliação independente definitiva.
* Os cenários de sensibilidade não comprovam efeitos causais.
* A avaliação não demonstra retorno financeiro de campanhas de retenção.
* Uma implantação real exigiria dados representativos, validação temporal, monitoramento, governança e avaliação de custos.

Entre os próximos passos possíveis estão a avaliação com dados públicos reais, a análise de drift e o aperfeiçoamento do monitoramento.

## 16. Finalidade do projeto

Este projeto foi desenvolvido para estudo e composição de portfólio profissional em Ciência de Dados e Machine Learning.

Demonstra competências em Python, SQL, modelagem preditiva, avaliação estatística, testes automatizados, desenvolvimento de APIs, Docker, CI/CD, documentação e comunicação de resultados.

**Repositório:**
[https://github.com/edusousa78/customer-churn-prediction](https://github.com/edusousa78/customer-churn-prediction)

---

**Aviso final:** projeto exclusivamente educacional, desenvolvido com dados sintéticos e empresa fictícia. Não deve ser utilizado como sistema comercial de decisão sem validação adicional.

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
