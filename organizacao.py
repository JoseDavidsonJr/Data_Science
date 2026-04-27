import pandas as pd
import json
import os
import sqlite3

# =========================
# CONFIGURAÇÃO
# =========================

pasta = r"C:\Facape_despesas"
banco = "facape_despesas.db"

dados_organizados = []

print("Arquivos encontrados:")
print(os.listdir(pasta))

# =========================
# LEITURA DOS JSONS
# =========================

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(".json"):
        print(f"Lendo arquivo: {arquivo}")

        caminho_arquivo = os.path.join(pasta, arquivo)

        # Extrair ano do nome do arquivo
        # Exemplo: despesas21.JSON → 2021
        ano = arquivo.lower().replace("despesas", "").replace(".json", "")
        ano = "20" + ano

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        for item in dados:
            linha = {
                "id_registro": item.get("id"),
                "fornecedor": item.get("fornecedor", {})
                                  .get("pessoa", {})
                                  .get("nome"),
                "cpf_cnpj": item.get("fornecedor", {})
                                .get("pessoa", {})
                                .get("cpfCnpj"),
                "ano": ano,

                # Valores principais
                "valor_empenhado": item.get("valorEmpenhado"),
                "valor_liquidado": item.get("valorLiquidado"),
                "valor_pago": item.get("valorPago"),
                "valor_executado": item.get("valorExecutado"),

                "empenhado": item.get("empenhado"),
                "liquidado": item.get("liquidado"),
                "pago": item.get("pago")
            }

            dados_organizados.append(linha)

# DataFrame principal
df_total = pd.DataFrame(dados_organizados)

print("\n===== AMOSTRA =====")
print(df_total.head())

print("\n===== INFO =====")
print(df_total.info())

# =========================
# RESUMO POR ANO
# =========================

resumo_ano = (
    df_total.groupby("ano")[
        ["valor_empenhado", "valor_liquidado", "valor_pago", "valor_executado"]
    ]
    .sum()
    .reset_index()
)

print("\n===== RESUMO POR ANO =====")
print(resumo_ano)

# =========================
# ANÁLISE DE FORNECEDORES
# =========================

fornecedores_anos = (
    df_total.dropna(subset=["fornecedor"])
    .groupby("fornecedor")["ano"]
    .nunique()
    .reset_index()
    .rename(columns={"ano": "quantidade_de_anos"})
)

fornecedores_repetidos = fornecedores_anos[
    fornecedores_anos["quantidade_de_anos"] > 1
]

ultimo_ano = df_total["ano"].max()

fornecedores_ultimo_ano = set(
    df_total[df_total["ano"] == ultimo_ano]["fornecedor"].dropna()
)

fornecedores_anos_anteriores = set(
    df_total[df_total["ano"] < ultimo_ano]["fornecedor"].dropna()
)

fornecedores_novos = fornecedores_ultimo_ano - fornecedores_anos_anteriores

df_fornecedores_novos = pd.DataFrame({
    "fornecedor_novo": list(fornecedores_novos)
})

# =========================
# SQLITE
# =========================

print("\nConectando ao SQLite...")

conn = sqlite3.connect(banco)

# Tabela principal
df_total.to_sql(
    "despesas",
    conn,
    if_exists="replace",
    index=False
)

# Tabela resumo por ano
resumo_ano.to_sql(
    "resumo_por_ano",
    conn,
    if_exists="replace",
    index=False
)

# Tabela fornecedores
fornecedores_anos.to_sql(
    "fornecedores_participacao",
    conn,
    if_exists="replace",
    index=False
)

# Tabela fornecedores repetidos
fornecedores_repetidos.to_sql(
    "fornecedores_repetidos",
    conn,
    if_exists="replace",
    index=False
)

# Tabela fornecedores novos
df_fornecedores_novos.to_sql(
    "fornecedores_novos",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\nBanco SQLite criado com sucesso!")
print(f"Arquivo gerado: {banco}")
print("\nTabelas criadas:")
print("- despesas")
print("- resumo_por_ano")
print("- fornecedores_participacao")
print("- fornecedores_repetidos")
print("- fornecedores_novos")
