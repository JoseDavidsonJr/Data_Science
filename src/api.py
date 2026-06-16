import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app
from src.monitor import PREDICTION_COUNTER, executar_pipeline_de_drift

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FACAPE-API")

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

app = FastAPI(
    title="FACAPE — Previsão de Risco de Contratos",
    description="Serviço de IA para predição de problemas em execuções orçamentárias.",
    version="1.1.0",
)

# Prometheus ASGI app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Singleton para o modelo
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Modelo carregado com sucesso de {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
    else:
        logger.error(f"Arquivo de modelo não encontrado em {MODEL_PATH}")

class ContractInput(BaseModel):
    valor_empenhado: float = Field(..., gt=0)
    tipo: int = Field(..., ge=0, le=1)
    perc_retido: float = Field(0.0, ge=0)
    perc_saldo: float = Field(0.0, ge=0)
    hist_taxa_execucao: float = Field(0.85, ge=0, le=1)
    hist_anos_ativo: int = Field(0, ge=0)
    hist_perc_anulacao: float = Field(0.0, ge=0, le=1)
    fornecedor_novo: int = Field(1, ge=0, le=1)

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
        logger.error("Tentativa de predição sem modelo carregado.")
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    PREDICTION_COUNTER.inc()

    try:
        # Monitoramento de Drift (assíncrono simulado)
        df_current = pd.DataFrame([data.model_dump()])
        executar_pipeline_de_drift(df_current)

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

        pred = bool(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.7:
            interpretacao = "ALTO RISCO: Histórico ou padrão financeiro indicam alta chance de problema."
        elif prob >= 0.4:
            interpretacao = "RISCO MODERADO: Recomenda-se acompanhamento da liquidação."
        else:
            interpretacao = "BAIXO RISCO: Contrato alinhado aos padrões de conformidade."

        return PredictionOutput(
            execucao_problematica=pred,
            probabilidade=round(prob, 4),
            interpretacao=interpretacao,
        )
    except Exception as e:
        logger.error(f"Erro processando predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no processamento.")
