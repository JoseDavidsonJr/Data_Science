import pandas as pd
import json
import os
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# =====================================================
# CONFIGURAÇÃO
# =====================================================

pasta = r"C:\Facape_despesas"
banco = "facape_despesas.db"

dados_organizados = []

print("Arquivos encontrados:")
print(os.listdir(pasta))

# =====================================================
# LEITURA DOS JSONS
# =====================================================

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(".json"):
        print(f"\nLendo arquivo: {arquivo}")

        caminho_arquivo = os.path.join(pasta, arquivo)

        # Exemplo:
        # despesas21.JSON → 2021
        ano = arquivo.lower().replace("despesas", "").replace(".json", "")
        ano = "20" + ano

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        for item in dados:
            linha = {
                "id_registro": item.get("id"),

                # fornecedor
                "fornecedor": item.get("fornecedor", {})
                                  .get("pessoa", {})
                                  .get("nome"),

                "cpf_cnpj": item.get("fornecedor", {})
                                .get("pessoa", {})
                                .get("cpfCnpj"),

                # ano
                "ano": ano,

                # valores principais
                "valor_empenhado": item.get("valorEmpenhado"),
                "valor_liquidado": item.get("valorLiquidado"),
                "valor_pago": item.get("valorPago"),
                "valor_executado": item.get("valorExecutado"),

                # campos formatados
                "empenhado": item.get("empenhado"),
                "liquidado": item.get("liquidado"),
                "pago": item.get("pago")
            }

            dados_organizados.append(linha)

# =====================================================
# DATAFRAME PRINCIPAL
# =====================================================

df_total = pd.DataFrame(dados_organizados)

print("\n==============================")
print("DATAFRAME PRINCIPAL")
print("==============================")

print(df_total.head())
print(df_total.info())

# =====================================================
# RESUMO ANUAL
# =====================================================

resumo_ano = (
    df_total.groupby("ano")[
        [
            "valor_empenhado",
            "valor_liquidado",
            "valor_pago",
            "valor_executado"
        ]
    ]
    .sum()
    .reset_index()
)

print("\n==============================")
print("RESUMO POR ANO")
print("==============================")

print(resumo_ano)

# =====================================================
# ANÁLISE DE FORNECEDORES
# =====================================================

print("\n==============================")
print("ANÁLISE DE FORNECEDORES")
print("==============================")

fornecedores_anos = (
    df_total.dropna(subset=["fornecedor"])
    .groupby("fornecedor")["ano"]
    .nunique()
    .reset_index()
    .rename(columns={"ano": "quantidade_de_anos"})
)

print("\nFornecedores e quantidade de anos:")
print(fornecedores_anos.head(20))

# fornecedores recorrentes
fornecedores_repetidos = fornecedores_anos[
    fornecedores_anos["quantidade_de_anos"] > 1
]

print("\nFornecedores recorrentes:")
print(fornecedores_repetidos.head(20))

# fornecedores novos
ultimo_ano = df_total["ano"].max()

fornecedores_ultimo_ano = set(
    df_total[
        df_total["ano"] == ultimo_ano
    ]["fornecedor"].dropna()
)

fornecedores_anos_anteriores = set(
    df_total[
        df_total["ano"] < ultimo_ano
    ]["fornecedor"].dropna()
)

fornecedores_novos = (
    fornecedores_ultimo_ano - fornecedores_anos_anteriores
)

df_fornecedores_novos = pd.DataFrame({
    "fornecedor_novo": list(fornecedores_novos)
})

print(f"\nFornecedores novos em {ultimo_ano}:")
print(df_fornecedores_novos.head(20))

# =====================================================
# SQLITE
# =====================================================

print("\n==============================")
print("SALVANDO NO SQLITE")
print("==============================")

conn = sqlite3.connect(banco)

df_total.to_sql(
    "despesas",
    conn,
    if_exists="replace",
    index=False
)

resumo_ano.to_sql(
    "resumo_por_ano",
    conn,
    if_exists="replace",
    index=False
)

fornecedores_anos.to_sql(
    "fornecedores_participacao",
    conn,
    if_exists="replace",
    index=False
)

fornecedores_repetidos.to_sql(
    "fornecedores_repetidos",
    conn,
    if_exists="replace",
    index=False
)

df_fornecedores_novos.to_sql(
    "fornecedores_novos",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\nBanco SQLite criado com sucesso!")

# =====================================================
# MODELO 1 — PREVER FORNECEDOR RECORRENTE
# =====================================================

print("\n==============================")
print("MODELO 1 — FORNECEDOR RECORRENTE")
print("==============================")

fornecedores_modelo = (
    df_total.dropna(subset=["fornecedor"])
    .groupby("fornecedor")["ano"]
    .nunique()
    .reset_index()
)

# Se apareceu em mais de 1 ano → recorrente
fornecedores_modelo["recorrente"] = (
    fornecedores_modelo["ano"]
    .apply(lambda x: 1 if x > 1 else 0)
)

# Mostrar quais são os fornecedores recorrentes
fornecedores_recorrentes = fornecedores_modelo[
    fornecedores_modelo["recorrente"] == 1
]

print("\n===== FORNECEDORES RECORRENTES =====")
print(
    fornecedores_recorrentes
    .sort_values(by="ano", ascending=False)
)

# X = quantidade de anos
X = fornecedores_modelo[["ano"]]

# y = recorrente ou não
y = fornecedores_modelo["recorrente"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Criar modelo
modelo_fornecedor = LogisticRegression()

# Treinar
modelo_fornecedor.fit(X_train, y_train)

# Prever
previsoes = modelo_fornecedor.predict(X_test)

# Avaliar
acc = accuracy_score(y_test, previsoes)

print(f"\nAcurácia do modelo: {acc:.2f}")

# Exemplo de previsão
print("\nExemplo de previsão:")

novo_dado = pd.DataFrame({
    "ano": [3]
})

resultado = modelo_fornecedor.predict(novo_dado)[0]

print("Se o fornecedor apareceu em 3 anos:")

if resultado == 1:
    print("→ Fornecedor recorrente")
else:
    print("→ Fornecedor NÃO recorrente")
# =====================================================
# MODELO 2 — PREVISÃO DE DESPESAS FUTURAS
# =====================================================

print("\n==============================")
print("MODELO 2 — PREVISÃO DE DESPESAS FUTURAS")
print("==============================")

# Garantir que ano está como número
resumo_ano["ano"] = resumo_ano["ano"].astype(int)

print("\nResumo anual:")
print(resumo_ano[["ano", "valor_pago"]])

# X = ano
X = resumo_ano[["ano"]]

# y = total pago
y = resumo_ano["valor_pago"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Criar modelo de regressão
modelo_despesa = LinearRegression()

# Treinar
modelo_despesa.fit(X_train, y_train)

# Fazer previsões no teste
previsoes = modelo_despesa.predict(X_test)

# Avaliar erro médio
mae = mean_absolute_error(y_test, previsoes)

print(f"\nMAE (erro médio absoluto): {mae:.2f}")

# =========================
# PREVISÃO DE 2026
# =========================

previsao_2026 = pd.DataFrame({
    "ano": [2026]
})

valor_2026 = modelo_despesa.predict(previsao_2026)[0]

print("\nPrevisão para 2026:")
print(f"→ Valor estimado de despesas: R$ {valor_2026:,.2f}")

# =========================
# PREVISÃO DE 2027
# =========================

previsao_2027 = pd.DataFrame({
    "ano": [2027]
})

valor_2027 = modelo_despesa.predict(previsao_2027)[0]

print("\nPrevisão para 2027:")
print(f"→ Valor estimado de despesas: R$ {valor_2027:,.2f}")

# =========================
# TENDÊNCIA
# =========================

coeficiente = modelo_despesa.coef_[0]

print("\nAnálise de tendência:")

if coeficiente > 0:
    print("→ As despesas estão em CRESCIMENTO")
elif coeficiente < 0:
    print("→ As despesas estão em QUEDA")
else:
    print("→ As despesas estão ESTÁVEIS")
