# Model Card — Customer Churn Prediction

## 1. Identificação do modelo

* **Projeto:** Conecta Brasil Telecom — Customer Churn Prediction
* **Versão:** 1.0 — protótipo educacional
* **Algoritmo:** Regressão Logística (scikit-learn)
* **Arquivo do modelo:** `<span>models/modelo_churn_logistico.joblib</span>`
* **Tipo de problema:** Classificação binária
* **Variável-alvo:** `<span>cancelou</span>`

## 2. Objetivo

Estimar a probabilidade de cancelamento de clientes de uma operadora fictícia de telecomunicações, utilizando indicadores de atendimento e satisfação.

O modelo tem finalidade educacional e demonstra um fluxo de desenvolvimento de Machine Learning, desde a preparação dos dados até a avaliação e disponibilização de previsões.

## 3. Dados utilizados

* **Origem:** Dados sintéticos gerados para o projeto.
* **Arquivo:** `<span>data/processed/customer_churn_sintetico.csv</span>`
* **Quantidade de registros:** 1.000 clientes.
* **Distribuição da variável-alvo:** 869 clientes sem cancelamento e 131 com cancelamento.
* **Taxa de churn:** 13,10%.

Os registros são fictícios e não representam clientes reais.

## 4. Variáveis preditoras


| Variável                           | Descrição                                    |
| ----------------------------------- | ---------------------------------------------- |
| `<span>total_atendimentos</span>`   | Quantidade total de atendimentos do cliente    |
| `<span>atendimentos_abertos</span>` | Quantidade de atendimentos ainda abertos       |
| `<span>media_satisfacao</span>`     | Média de satisfação do cliente, entre 1 e 5 |

## 5. Treinamento

O desenvolvimento utilizou uma divisão estratificada dos dados:

* Treinamento: 600 registros (60%).
* Validação: 200 registros (20%).
* Teste: 200 registros (20%).

A semente aleatória utilizada foi `<span>random_state=42</span>`.

O algoritmo foi treinado utilizando `<span>LogisticRegression(max_iter=1000)</span>`.

## 6. Critério de classificação

O modelo produz uma probabilidade estimada de cancelamento.

Foi selecionado um limiar de decisão (*threshold*) de **0,20**, a partir de uma avaliação da métrica F1 no conjunto de validação.

* Probabilidade menor que 0,20: previsão de não cancelamento.
* Probabilidade maior ou igual a 0,20: previsão de cancelamento.

O limiar foi escolhido para o exercício e não representa uma política comercial validada.

## 7. Resultados da avaliação

Resultados obtidos no conjunto de teste de 200 registros:


| Métrica          | Resultado |
| ----------------- | --------- |
| Acurácia         | 76,00%    |
| Precisão         | 28,00%    |
| Recall            | 53,85%    |
| F1-score          | 36,84%    |
| ROC-AUC           | 0,7465    |
| Average Precision | 0,3567    |

Matriz de confusão:


| <br/>               | Previsto: não cancelou | Previsto: cancelou |
| ------------------- | ----------------------- | ------------------ |
| Real: não cancelou | 138                     | 36                 |
| Real: cancelou      | 12                      | 14                 |

Os resultados refletem exclusivamente os dados sintéticos deste experimento. Além disso, durante o desenvolvimento educacional, o conjunto de teste foi consultado em etapas anteriores; portanto, não deve ser considerado uma avaliação completamente independente de todas as decisões do projeto.

## 8. Interpretação

Os coeficientes do modelo indicaram:

* Associação positiva entre maior quantidade de atendimentos e risco estimado de churn.
* Associação positiva entre atendimentos abertos e risco estimado de churn.
* Associação negativa entre satisfação média e risco estimado de churn.

Essas associações não demonstram causalidade.

## 9. Uso pretendido

O modelo foi desenvolvido para:

* Demonstrar técnicas de classificação supervisionada.
* Exercitar avaliação de modelos com classes desbalanceadas.
* Construir um projeto de portfólio em Ciência de Dados.
* Simular a identificação de clientes que poderiam receber atenção preventiva.

## 10. Limitações e riscos

* O dataset é sintético.
* O modelo não foi validado com dados reais.
* A precisão de 28% indica uma proporção relevante de falsos positivos.
* Existem falsos negativos, incluindo clientes que cancelariam e não seriam identificados.
* Não foram realizados estudos de viés, estabilidade temporal ou impacto operacional.
* O desempenho pode mudar significativamente em uma população real.
* Não deve ser utilizado para decisões comerciais reais sem validação adicional.

## 11. Reprodutibilidade e qualidade

O repositório contém:

* Scripts Python de preparação, treinamento, avaliação e inferência.
* Dataset sintético processado.
* Modelo serializado com Joblib.
* Testes automatizados com Pytest.
* Análise estática com Ruff.
* Pipeline de integração contínua com GitHub Actions.
* Dependências registradas em `<span>requirements.txt</span>`.

O arquivo Joblib deve ser carregado apenas quando sua origem for confiável.

## 12. Próximas melhorias

* Avaliar o modelo com dados independentes e representativos.
* Investigar calibração das probabilidades.
* Avaliar custo de falsos positivos e falsos negativos.
* Implementar monitoramento de desempenho e mudanças nos dados.
* Documentar a versão dos dados e do modelo.
* Comparar novos algoritmos e estratégias de classificação.
