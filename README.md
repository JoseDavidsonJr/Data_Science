# FACAPE — Análise de Execução Orçamentária (2021–2025)


Sistema de inteligência de dados para análise de risco em contratos públicos da FACAPE (Autarquia Educacional do Vale do São Francisco). O projeto utiliza **CRISP-DM**, **MCP-Brasil** para coleta automatizada e **Prometheus/Grafana** para monitoramento de drift de modelo.

## Integrantes

| Nome | E-mail |
|---|---|
| Caio Lassalvia de Barros | lasalviacaio3@gmail.com |
| José Davidson Lopes Pimentel Júnior | juniorpimenteldavidson@gmail.com |
| Vinicius de Carvalho Pereira | vinicius.carvalho03101@gmail.com |

## 🏗️ Arquitetura Técnica

- **Coleta:** Integração dinâmica com o **TCE-PE (SAGRES)** via `mcp-brasil`.
- **Inteligência:** 
  - Classificação de risco (RandomForest/LogisticRegression).
  - Detecção de anomalias financeiras (Isolation Forest).
  - Perfilamento de fornecedores (K-Means).
- **MLOps:**
  - Tracking de experimentos com **MLflow**.
  - Monitoramento de Data Drift com **Evidently AI**.
  - Visualização de métricas em tempo real com **Prometheus & Grafana**.

## 🚀 Como Executar

### Pré-requisitos
- Python 3.11+
- Docker & Docker Compose

### Configuração Rápida (Local)

1. **Instalar dependências e coletar dados:**
   ```bash
   make install
   make fetch
   ```

2. **Treinar modelos e gerar relatórios:**
   ```bash
   make all-models
   ```

3. **Subir ambiente completo (API + Monitoramento):**
   ```bash
   make up
   ```
   - **API:** http://localhost:8000/docs
   - **Grafana (MLOps & BI):** http://localhost:3000 (admin/admin)
   - **Prometheus:** http://localhost:9090
   - **MLflow UI:** `make mlflow-ui` (http://localhost:5000)

## 🧪 Testes e Qualidade

O projeto utiliza `pytest` para garantir a integridade das transformações de dados:
```bash
python -m pytest tests/
```

## 📊 Estrutura de Diretórios

```text
data/           Scripts de carga (SQLite) e ingestão via MCP-Brasil.
src/
  api.py        Serviço FastAPI com métricas Prometheus.
  business_exporter.py  Exportador de métricas de BI para o Grafana.
  features.py   Engenharia de features com lookback temporal (evita leakage).
  monitor.py    Engine de detecção de drift e auditoria em PostgreSQL.
  train.py      Pipeline de classificação e tracking MLflow.
grafana/        Provisionamento automático de dashboards e data sources.
reports/        Insights visuais gerados automaticamente.
tests/          Suíte de testes unitários.
```

## 📈 Metodologia e Resultados
O projeto processa atualmente **1.870 registros** extraídos do SAGRES (TCE-PE). O modelo de classificação utiliza um split temporal (**Treino: 2021-2023 / Teste: 2024-2025**) para simular o uso real. 

- **Inteligência de Negócio**: Dashboard Grafana integrado com Top 10 fornecedores e evolução orçamentária (Empenhado vs Pago).
- **Impacto**: A feature mais relevante é o `valor_empenhado_log`, indicando que o vulto financeiro é o principal indutor de risco na autarquia.

