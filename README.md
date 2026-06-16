# FACAPE — Análise de Execução Orçamentária (2021–2025)

[![FACAPE Data Pipeline](https://github.com/USER/REPO/actions/workflows/pipeline.yml/badge.svg)](https://github.com/USER/REPO/actions/workflows/pipeline.yml)

Sistema de inteligência de dados para análise de risco em contratos públicos da FACAPE (Autarquia Educacional do Vale do São Francisco). O projeto utiliza **CRISP-DM**, **MCP-Brasil** para coleta automatizada e **Prometheus/Grafana** para monitoramento de drift de modelo.

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
   - **Grafana:** http://localhost:3000 (admin/admin)
   - **Prometheus:** http://localhost:9090

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
  features.py   Engenharia de features com lookback temporal (evita leakage).
  monitor.py    Engine de detecção de drift e auditoria em PostgreSQL.
  train.py      Pipeline de classificação e tracking MLflow.
grafana/        Provisionamento automático de dashboards e data sources.
reports/        Insights visuais gerados automaticamente.
tests/          Suíte de testes unitários.
```

## 📈 Metodologia de Risco
O modelo de classificação utiliza um split temporal (**Treino: 2021-2023 / Teste: 2024-2025**) para simular o uso real. A feature mais impactante é o histórico de execução do fornecedor (`hist_taxa_execucao`), provando que o comportamento passado é o melhor preditor de riscos futuros na gestão pública.
