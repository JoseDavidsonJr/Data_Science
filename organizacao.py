import pandas as pd
import json

arquivo_json = r"C:\Facape_despesas\despesas21.json"

with open(arquivo_json, "r", encoding="utf-8") as f:
    dados = json.load(f)

dados_organizados = []

for item in dados:
    linha = {
        "id": item.get("id"),

        # fornecedor
        "fornecedor": item.get("fornecedor", {})
                          .get("pessoa", {})
                          .get("nome"),

        # valores principais
        "valorEmpenhado": item.get("valorEmpenhado"),
        "valorLiquidado": item.get("valorLiquidado"),
        "valorPago": item.get("valorPago"),
        "valorExecutado": item.get("valorExecutado"),

        # valores do bloco "valores"
        "empenho_emissao": item.get("valores", {}).get("EMPENHO_EMISSAO"),
        "empenho_liquidacao": item.get("valores", {}).get("EMPENHO_LIQUIDACAO"),
        "empenho_pagamento": item.get("valores", {}).get("EMPENHO_PAGAMENTO"),
    }

    dados_organizados.append(linha)

df = pd.DataFrame(dados_organizados)

# Mostrar 20 linhas
print(df.head(20))

# (opcional) salvar
df.to_csv("ids_fornecedores_valores.csv", index=False, encoding="utf-8-sig")