"""
API de predição — FastAPI.

POST /predict  → risco de execução problemática de um contrato
GET  /health   → healthcheck
"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

app = FastAPI(
    title="FACAPE — Previsão de Execução Problemática",
    description=(
        "Dado um novo contrato, retorna a probabilidade de execução problemática "
        "(taxa de pagamento < 70% ou presença de anulação)."
    ),
    version="1.0.0",
)

# Carrega o modelo na inicialização
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class ContractInput(BaseModel):
    valor_empenhado: float = Field(..., gt=0, description="Valor empenhado em R$")
    tipo: int = Field(..., ge=0, le=1, description="0 = Pessoa Física, 1 = Pessoa Jurídica")
    perc_retido: float = Field(0.0, ge=0, description="% retido sobre o empenhado")
    perc_saldo: float = Field(0.0, ge=0, description="% saldo a pagar sobre o empenhado")
    hist_taxa_execucao: float = Field(0.85, ge=0, le=1, description="Taxa de execução histórica do fornecedor")
    hist_anos_ativo: int = Field(0, ge=0, description="Quantos anos o fornecedor apareceu anteriormente")
    hist_perc_anulacao: float = Field(0.0, ge=0, le=1, description="% histórico de contratos com anulação")
    fornecedor_novo: int = Field(1, ge=0, le=1, description="1 = fornecedor sem histórico anterior")


class PredictionOutput(BaseModel):
    execucao_problematica: bool
    probabilidade: float
    interpretacao: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: ContractInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute 'make train' primeiro.")

    features = np.array([[
        np.log1p(data.valor_empenhado),
        data.tipo,
        data.perc_retido,
        data.perc_saldo,
        data.hist_taxa_execucao,
        data.hist_anos_ativo,
        data.hist_perc_anulacao,
        data.fornecedor_novo,
    ]])

    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    if prob >= 0.7:
        interpretacao = "Alto risco: contrato provavelmente terá problemas de execução."
    elif prob >= 0.4:
        interpretacao = "Risco moderado: monitorar execução do contrato."
    else:
        interpretacao = "Baixo risco: execução dentro do padrão histórico."

    return PredictionOutput(
        execucao_problematica=bool(pred),
        probabilidade=round(prob, 4),
        interpretacao=interpretacao,
    )
