# Análise e Predição de Execução Orçamentária em Despesas Públicas: Um Estudo de Caso Aplicado à FACAPE (2021–2025)

**Resumo**— A eficiência na execução orçamentária constitui um pilar fundamental para a governança e a sustentabilidade financeira de autarquias públicas de ensino superior. Este trabalho apresenta um estudo de caso aplicado às despesas da Faculdade de Ciências Aplicadas e Sociais de Petrolina (FACAPE) no período compreendido entre 2021 e 2025. Utilizou-se o framework metodológico CRISP-DM (*Cross-Industry Standard Process for Data Mining*) para a análise e a inferência de inconsistências e falhas na liquidação de empenhos contratuais. A partir de um *dataset* consolidado de 1.870 registros históricos (via MCP-Brasil), implementou-se uma abordagem analítica tripartida: classificação supervisionada com validação temporal (*out-of-time*) para a predição de execuções problemáticas, detecção de anomalias via algoritmos não supervisionados e agrupamento estrutural (*clustering*). O algoritmo *Logistic Regression* obteve desempenho preditivo robusto para o cenário atual, alcançando uma AUC-ROC de 0,73. O mapeamento multidimensional identificou 10% de contratos anômalos, com destaque para distorções em folhas de pagamento. Adicionalmente, propõe-se uma arquitetura de monitoramento contínuo fundamentada em *Evidently AI*, *PostgreSQL* e *Grafana* para a mitigação de *data drift*. Os resultados ratificam o potencial da Ciência de Dados na antecipação de riscos fiscais e no refinamento do planejamento orçamentário público.

**Palavras-chave**— Finanças Públicas, CRISP-DM, *Machine Learning*, Execução Orçamentária, Detecção de Anomalias.

---

## I. INTRODUÇÃO

A gestão de recursos no ecossistema da administração pública brasileira demanda estrita observância aos preceitos constitucionais de legalidade, impessoalidade, moralidade, publicidade e eficiência (Art. 37, CF/88). No contexto das autarquias municipais de ensino superior, como a Faculdade de Ciências Aplicadas e Sociais de Petrolina (FACAPE), a convergência entre o planejamento orçamentário e a execução financeira é imperativa para assegurar a perenidade das operações acadêmicas e administrativas.

Historicamente, a análise de despesas públicas tem se pautado em avaliações *ex-post*, caracterizando-se por auditorias reativas após a consolidação dos gastos. Tal paradigma limita a capacidade de intervenção preventiva em contratos que apresentam desvios de execução. Com a ascensão da Ciência de Dados e a padronização de ciclos de vida de mineração de dados como o CRISP-DM, torna-se factível a transformação de registros brutos de portais de transparência em ativos preditivos e prescritivos.

---

## II. COMPREENSÃO E PREPARAÇÃO DOS DADOS

### A. Coleta e Consolidação do Dataset
O *corpus* de dados foi extraído dinamicamente via **MCP-Brasil** conectando-se ao sistema SAGRES do TCE-PE, abrangendo os exercícios de 2021 a 2025. Após o saneamento e unificação, obteve-se um universo de **1.870 registros válidos** de empenho.

### B. Engenharia de Atributos e Lookback Temporal
Aplicou-se a técnica de *lookback* temporal para evitar *data leakage*, computando preditores históricos como a `hist_taxa_execucao` e `hist_anos_ativo`. A variável `valor_empenhado_log` foi identificada como o preditor de maior impacto (96% de importância relativa), confirmando que o porte financeiro é o principal indutor de risco na autarquia.

---

## III. METODOLOGIA

### A. Validação Temporal Out-of-Time
O conjunto de dados foi cindido cronologicamente: registros de 2021 a 2023 para treinamento (**1.051 registros**) e dados de 2024 e 2025 para teste (**819 registros**), refletindo a capacidade de generalização para novos exercícios fiscais.

### B. Algoritmos Avaliados
Foram testadas seis arquiteturas. No cenário atual com dados expandidos, a **Regressão Logística** e o **Random Forest** apresentaram os melhores equilíbrios de separação de classe, com AUC-ROC atingindo **0,73**.

---

## IV. RESULTADOS E DISCUSSÃO

### A. Detecção de Anomalias (Isolation Forest)
O *Isolation Forest* (com $\alpha=0.10$) identificou **187 contratos** (10% da base) como anomalias. O caso mais expressivo envolveu a Folha de Pagamento, com empenhos individuais superiores a R$ 10 milhões, exibindo comportamentos estatísticos discrepantes em relação aos fornecedores comerciais padrão.

### B. Análise por Segmentos de Risco (K-Means)
A clusterização ($k=3$) estratificou os fornecedores em:
*   **Médio Risco (330 fornecedores)**: Ticket médio de R$ 388.754,68, recorrentes e de grande porte.
*   **Baixo Risco (310 fornecedores)**: Ticket médio de R$ 14.831,96, alta conformidade (99.5% de execução).
*   **Alto Risco (13 fornecedores)**: Novos fornecedores ou com baixíssima taxa de execução (2%).

---

## V. CONCLUSÃO
A integração do MCP-Brasil permitiu uma análise mais densa e fiel à realidade do TCE-PE. O modelo demonstrou alta sensibilidade ao valor dos contratos, permitindo à FACAPE priorizar a fiscalização sobre os empenhos de maior vulto financeiro, onde o risco de ineficiência alocativa é máximo.

---

## REFERÊNCIAS
[1] Brasil. Constituição da República Federativa do Brasil de 1988.
[2] P. Chapman et al., *CRISP-DM 1.0*.
[3] F. T. Liu et al., "Isolation Forest", 2008.
