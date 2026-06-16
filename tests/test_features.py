import pytest
import pandas as pd
import numpy as np
from src.features import add_supplier_history, build_classification_features

def test_add_supplier_history():
    # Mock data
    data = [
        {"fornecedor": "A", "ano": 2021, "taxa_execucao": 1.0, "valor_anulado": 0},
        {"fornecedor": "A", "ano": 2022, "taxa_execucao": 0.5, "valor_anulado": 100},
        {"fornecedor": "B", "ano": 2022, "taxa_execucao": 0.8, "valor_anulado": 0},
    ]
    df = pd.DataFrame(data)
    
    df_result = add_supplier_history(df)
    
    # Verifica se fornecedor novo em 2021 foi marcado corretamente
    row_2021 = df_result[(df_result["fornecedor"] == "A") & (df_result["ano"] == 2021)].iloc[0]
    assert row_2021["fornecedor_novo"] == 1
    
    # Verifica histórico do fornecedor A em 2022
    row_2022 = df_result[(df_result["fornecedor"] == "A") & (df_result["ano"] == 2022)].iloc[0]
    assert row_2022["fornecedor_novo"] == 0
    assert row_2022["hist_taxa_execucao"] == 1.0 # Média do ano anterior
    assert row_2022["hist_anos_ativo"] == 1

def test_build_classification_features():
    data = [
        {"fornecedor": "A", "ano": 2021, "taxa_execucao": 0.6, "valor_anulado": 0, "valor_empenhado": 1000, "valor_pago": 600, "valor_retido": 0, "valor_saldo_a_pagar": 400, "perc_retido": 0, "perc_saldo": 0.4, "tipo": 1},
    ]
    df = pd.DataFrame(data)
    df_result = build_classification_features(df)
    
    # Target deve ser 1 pois taxa_execucao < 0.7
    assert df_result["target"].iloc[0] == 1
    assert "valor_empenhado_log" in df_result.columns
    assert df_result["valor_empenhado_log"].iloc[0] == pytest.approx(np.log1p(1000))
