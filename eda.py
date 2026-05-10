import pandas as pd
import json
import os
import matplotlib.pyplot as plt

pasta = input(
    "Digite o caminho da pasta dos JSONs: "
).strip().replace('"', '')

dados = []

for arquivo in os.listdir(pasta):

    if arquivo.lower().endswith(".json"):

        caminho = os.path.join(
            pasta,
            arquivo
        )

        ano = arquivo.lower() \
            .replace("despesas", "") \
            .replace(".json", "")

        ano = int("20" + ano)

        with open(
            caminho,
            "r",
            encoding="utf-8"
        ) as f:

            json_data = json.load(f)

        for item in json_data:

            dados.append({

                "ano": ano,

                "fornecedor": (
                    item.get("fornecedor", {})
                        .get("pessoa", {})
                        .get("nome")
                ),

                "valor_empenhado":
                    item.get("valorEmpenhado") or 0,

                "valor_pago":
                    item.get("valorPago") or 0,

                "valor_liquidado":
                    item.get("valorLiquidado") or 0,

                "valor_retido":
                    item.get("valorRetido") or 0
            })

df = pd.DataFrame(dados)

print(df.head())

print(df.info())

print(df.describe())

# ========================================
# NULOS
# ========================================

print("\nNulos:")
print(df.isnull().sum())

# ========================================
# RESUMO ANUAL
# ========================================

resumo = (
    df.groupby("ano")[
        [
            "valor_empenhado",
            "valor_pago",
            "valor_liquidado",
            "valor_retido"
        ]
    ]
    .sum()
    .reset_index()
)

print(resumo)

# ========================================
# GRÁFICO
# ========================================

plt.figure(figsize=(10, 5))

plt.plot(
    resumo["ano"],
    resumo["valor_empenhado"],
    marker="o"
)

plt.title(
    "Valor Empenhado por Ano"
)

plt.xlabel("Ano")

plt.ylabel("Valor")

plt.grid()

plt.show()