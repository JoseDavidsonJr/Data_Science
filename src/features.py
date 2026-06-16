"""Feature engineering com lookback temporal de fornecedores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_data import load_jsons

# Features usadas pelos modelos de classificação
FEATURE_COLS = [
    "valor_empenhado_log",
    "tipo",
    "perc_retido",
    "hist_taxa_execucao",
    "hist_anos_ativo",
    "hist_perc_anulacao",
    "fornecedor_novo",
]

# Features usadas pelo Isolation Forest
ANOMALY_FEATURE_COLS = [
    "valor_empenhado_log",
    "taxa_execucao",
    "perc_anulado",
    "perc_retido",
    "perc_saldo",
    "tipo",
]

# Features usadas pelo clustering de fornecedores
CLUSTER_FEATURE_COLS = [
    "valor_total_log",
    "anos_ativo",
    "taxa_execucao_media",
    "perc_anulado_medio",
    "perc_retido_medio",
    "tipo",
]


def add_supplier_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada registro, calcula features baseadas no histórico do fornecedor
    em anos ANTERIORES ao registro — evita leakage.
    """
    df = df.sort_values("ano").copy()

    hist_taxa, hist_anos, hist_anulacao, eh_novo = [], [], [], []

    for _, row in df.iterrows():
        passado = df[
            (df["fornecedor"] == row["fornecedor"]) & (df["ano"] < row["ano"])
        ]
        if len(passado) == 0:
            hist_taxa.append(None)
            hist_anos.append(0)
            hist_anulacao.append(None)
            eh_novo.append(1)
        else:
            hist_taxa.append(passado["taxa_execucao"].mean())
            hist_anos.append(int(passado["ano"].nunique()))
            hist_anulacao.append(float((passado["valor_anulado"] > 0).mean()))
            eh_novo.append(0)

    df["hist_taxa_execucao"] = hist_taxa
    df["hist_anos_ativo"] = hist_anos
    df["hist_perc_anulacao"] = hist_anulacao
    df["fornecedor_novo"] = eh_novo

    # Fornecedores sem histórico recebem a média do dataset (impute by mean)
    global_taxa = df["taxa_execucao"].mean()
    global_anulacao = (df["valor_anulado"] > 0).mean()

    # Evita FutureWarnings de downcasting silencioso
    df["hist_taxa_execucao"] = df["hist_taxa_execucao"].fillna(global_taxa).infer_objects()
    df["hist_perc_anulacao"] = df["hist_perc_anulacao"].fillna(global_anulacao).infer_objects()

    return df


def build_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de histórico, log do valor e define o target."""
    df = add_supplier_history(df)
    df["valor_empenhado_log"] = np.log1p(df["valor_empenhado"])

    # Target: execução problemática = taxa < 70% OU houve anulação parcial
    df["target"] = (
        (df["taxa_execucao"] < 0.70) | (df["valor_anulado"] > 0)
    ).astype(int)

    return df


def build_supplier_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por fornecedor para o clustering de risco."""
    perfil = (
        df.groupby("fornecedor")
        .agg(
            valor_total=("valor_pago", "sum"),
            anos_ativo=("ano", "nunique"),
            taxa_execucao_media=("taxa_execucao", "mean"),
            perc_anulado_medio=("perc_anulado", "mean"),
            perc_retido_medio=("perc_retido", "mean"),
            tipo=("tipo", "first"),
        )
        .reset_index()
    )
    perfil["valor_total_log"] = np.log1p(perfil["valor_total"])
    return perfil
