# FACAPE — Análise de Execução Orçamentária (2021–2025)

Projeto de Ciência de Dados (AV2) aplicando CRISP-DM sobre despesas públicas da FACAPE de 2021 a 2025.

## Problema de negócio

> Dado um contrato de empenho, qual a probabilidade de ele ter execução problemática — pagamento abaixo de 70% ou anulação parcial?

Além da predição, o projeto identifica fornecedores por perfil de risco (KMeans) e detecta contratos com padrão financeiro atípico (Isolation Forest).

## Dataset

- **Fonte:** Portal de transparência da FACAPE
- **Período:** 2021–2025 (5 arquivos JSON)
- **Volume:** ~973 registros após limpeza
- **Campos principais:** fornecedor, CPF/CNPJ, valor empenhado/liquidado/pago/anulado/retido

## Modelos

| Modelo | Tipo | Pergunta |
|--------|------|----------|
| RandomForest / GBM | Classificação | Execução será problemática? |
| Isolation Forest | Anomaly Detection | Contrato tem padrão atípico? |
| KMeans | Clustering | Qual o perfil de risco do fornecedor? |

**Validação:** out-of-time — treino em 2021–2023, teste em 2024–2025.

## Como rodar

### Local

```bash
# 1. Instalar dependências
make install

# 2. Treinar todos os modelos (salva em models/)
make all-models

# 3. Subir a API
make api
# → http://localhost:8000/docs
```

### Docker

```bash
make docker-build
make docker-run
# → http://localhost:8000/docs
```

### MLflow UI

```bash
make mlflow-ui
# → http://localhost:5000
```

## Estrutura do projeto

```
data/           Scripts de carga dos JSONs → DataFrame
src/
  features.py   Feature engineering (lookback temporal de fornecedores)
  train.py      Classificação + MLflow (6 experimentos)
  anomaly.py    Isolation Forest + MLflow (3 experimentos)
  cluster.py    KMeans perfilamento de risco + MLflow (3 experimentos)
  api.py        FastAPI — POST /predict
notebooks/      EDA exploratória
models/         Modelos serializados (gerados por make train)
reports/        Gráficos, artigo e slides
```

## Resultado principal

**Melhor classificador:** RandomForest (treino 2021–2023, teste 2024–2025).
A feature mais importante é `hist_taxa_execucao` — o histórico do fornecedor
em anos anteriores é o melhor preditor de risco futuro.

Ver detalhes em `reports/` e no MLflow UI.
