# Projeto-Avaliativo-Módulo-1
Mini Projeto de Tratamento de dados

# Análise de Risco de Crédito e Modelagem Preditiva

Este projeto visa desenvolver e avaliar modelos de Machine Learning para prever a inadimplência de empréstimos (`loan_status`). Através de um pipeline completo de análise de dados, desde o pré-processamento até a avaliação de modelos e um veredito de negócios, buscamos identificar a melhor abordagem para auxiliar na tomada de decisão em concessão de crédito.

## 🚀 Visão Geral do Projeto

O objetivo principal é construir um modelo preditivo robusto para classificar se um cliente irá pagar ou não um empréstimo. A análise focou na minimização de Falsos Negativos (prever que o cliente pagará, mas ele não o faz), que representam o maior custo para a instituição financeira.

## 📊 Etapas Realizadas

1.  **Carregamento e Exploração Inicial dos Dados:**
    *   Análise da estrutura do dataset, tipos de dados e estatísticas descritivas.
    *   Identificação de distribuições de variáveis (ex: `person_age`) e o desbalanceamento da variável alvo (`loan_status`).
    *   Visualização de correlações entre variáveis numéricas.

2.  **Pré-processamento de Dados:**
    *   **Remoção de Duplicatas:** Identificadas e removidas 165 linhas duplicadas para garantir a unicidade das observações.
    *   **Tratamento de Valores Ausentes:** Valores ausentes em `person_emp_length` e `loan_int_rate` foram imputados pela **mediana**, escolhida por sua robustez a outliers e distribuições assimétricas.
    *   **Tratamento de Outliers:** Abordagem customizada, com remoção de valores irrealistas (`person_age` > 100, `person_emp_length` > 60) e *capping* para `person_income` (no percentil 99). Outliers em variáveis como `loan_amnt` foram mantidos por serem considerados informativos.

3.  **Engenharia de Features:**
    *   Criação da nova variável `comprometimento_renda`, calculada como `(loan_amnt / person_income) * 100`, fundamental para a análise de risco de crédito.

4.  **Preparação de Dados para Modelagem:**
    *   **Codificação de Variáveis Categóricas:** Utilizado One-Hot Encoding para converter variáveis como `person_home_ownership` e `loan_intent` em formato numérico.
    *   **Divisão de Dados:** O dataset foi dividido em conjuntos de treino (80%) e teste (20%) de forma **estratificada** para manter a proporção da variável alvo.
    *   **Balanceamento de Classes:** Aplicação de **SMOTE** (Synthetic Minority Over-sampling Technique) **apenas no conjunto de treino** para lidar com o desbalanceamento da classe (`loan_status`), resultando em uma distribuição 50/50 e evitando vazamento de dados.
    *   **Escalonamento de Features:** Utilizado `StandardScaler` nas variáveis contínuas (apenas `fit_transform` no treino e `transform` no teste). Foi ressaltada a importância para modelos baseados em distância (KNN) e a desnecessidade para modelos baseados em árvore (Decision Tree).

5.  **Modelagem e Otimização:**
    *   **KNN (K-Nearest Neighbors):** Otimizado o hiperparâmetro `n_neighbors` (testados [3, 5, 7, 9, 11, 13]), avaliando métricas como `Accuracy`, `Precision`, `Recall` e `F1-score` nos conjuntos de treino e teste. A configuração `n_neighbors=7` ou `13` mostrou o melhor equilíbrio.
    *   **Árvore de Decisão (Decision Tree):** Otimizado o hiperparâmetro `max_depth` (testados [3, 5, 7, 9, 11, None]), usando dados não escalonados. A configuração `max_depth=11` se destacou.
    *   **Diagnóstico de Overfitting:** Análise da diferença entre o desempenho nos conjuntos de treino e teste para identificar e mitigar o overfitting.

6.  **Avaliação e Veredito de Negócios:**
    *   **Modelos Selecionados:** KNN (`n_neighbors=7`) e Árvore de Decisão (`max_depth=11`) foram escolhidos para a avaliação final.
    *   **Relatórios de Classificação e Matrizes de Confusão:** Gerados para ambos os modelos no conjunto de teste.
    *   **Análise de Impacto de Erros:** Detalhado o impacto financeiro de Falsos Positivos (perda de oportunidade) e Falsos Negativos (prejuízo direto), com a conclusão de que FNs são o erro mais custoso.
    *   **Veredito Final:** O **Modelo de Árvore de Decisão (max_depth=11)** foi o **escolhido para produção**. Apesar do KNN ter um `Recall` marginalmente maior para a classe de inadimplência (75% vs 72%), a Árvore de Decisão demonstrou uma **`Precision` significativamente superior (95% vs 63%)** para a classe de inadimplência. Isso significa que, ao prever um calote, a Árvore de Decisão é muito mais confiável, minimizando Falsos Positivos (decisões de negar crédito erradas) e, de forma equilibrada, evitando Falsos Negativos (empréstimos concedidos a quem não pagará).

## 🛠️ Tecnologias Utilizadas

*   Python 3
*   Pandas
*   NumPy
*   Scikit-learn
*   Imbalanced-learn (imblearn)
*   Matplotlib
*   Seaborn
