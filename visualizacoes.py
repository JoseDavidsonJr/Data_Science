import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIGURAÇÃO
# =====================================================

pasta = os.path.dirname(os.path.abspath(__file__))

dados_organizados = []

# =====================================================
# LEITURA DOS JSONS (igual ao main.py)
# =====================================================

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(".json"):
        caminho_arquivo = os.path.join(pasta, arquivo)

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
                "valor_empenhado": item.get("valorEmpenhado"),
                "valor_liquidado": item.get("valorLiquidado"),
                "valor_pago": item.get("valorPago"),
                "valor_executado": item.get("valorExecutado"),
            }
            dados_organizados.append(linha)

df_total = pd.DataFrame(dados_organizados)
df_total["ano"] = df_total["ano"].astype(int)

resumo_ano = (
    df_total.groupby("ano")[
        ["valor_empenhado", "valor_liquidado", "valor_pago"]
    ]
    .sum()
    .reset_index()
)

# =====================================================
# CONFIGURAÇÃO VISUAL GERAL
# =====================================================

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# =====================================================
# GRÁFICO 1 — Evolução anual dos pagamentos (linha)
# =====================================================
# O que mostra: como o total pago cresceu de 2021 a 2025

plt.figure()

sns.lineplot(
    data=resumo_ano,
    x="ano",
    y="valor_pago",
    marker="o",
    linewidth=2.5,
    color="steelblue"
)

# Adiciona o valor em cima de cada ponto
for _, row in resumo_ano.iterrows():
    plt.text(
        row["ano"],
        row["valor_pago"] + 300_000,
        f'R$ {row["valor_pago"]/1_000_000:.1f}M',
        ha="center",
        fontsize=10
    )

plt.title("Evolução dos Pagamentos por Ano")
plt.xlabel("Ano")
plt.ylabel("Total Pago (R$)")
plt.tight_layout()
plt.savefig("grafico_01_evolucao_anual.png", dpi=150)
plt.show()

print("Gráfico 1 salvo: grafico_01_evolucao_anual.png")

# =====================================================
# GRÁFICO 2 — Empenhado vs Pago por ano (barras agrupadas)
# =====================================================
# O que mostra: a diferença entre o que foi prometido e o que saiu de fato

resumo_melted = resumo_ano.melt(
    id_vars="ano",
    value_vars=["valor_empenhado", "valor_pago"],
    var_name="tipo",
    value_name="valor"
)

# Deixa os nomes mais legíveis na legenda
resumo_melted["tipo"] = resumo_melted["tipo"].replace({
    "valor_empenhado": "Empenhado",
    "valor_pago": "Pago"
})

plt.figure()

sns.barplot(
    data=resumo_melted,
    x="ano",
    y="valor",
    hue="tipo",
    palette=["steelblue", "seagreen"]
)

plt.title("Empenhado vs Pago por Ano")
plt.xlabel("Ano")
plt.ylabel("Valor (R$)")
plt.legend(title="")
plt.tight_layout()
plt.savefig("grafico_02_empenhado_vs_pago.png", dpi=150)
plt.show()

print("Gráfico 2 salvo: grafico_02_empenhado_vs_pago.png")

# =====================================================
# GRÁFICO 3 — Top 10 fornecedores (barras horizontal)
# =====================================================
# O que mostra: quem recebeu mais dinheiro no total dos 5 anos

top10 = (
    df_total.groupby("fornecedor")["valor_pago"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure()

sns.barplot(
    data=top10,
    x="valor_pago",
    y="fornecedor",
    palette="Blues_r"
)

# Adiciona o valor no final de cada barra
for i, row in top10.iterrows():
    plt.text(
        row["valor_pago"] + 100_000,
        i,
        f'R$ {row["valor_pago"]/1_000_000:.1f}M',
        va="center",
        fontsize=9
    )

plt.title("Top 10 Fornecedores por Valor Total Pago (2021–2025)")
plt.xlabel("Total Pago (R$)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("grafico_03_top10_fornecedores.png", dpi=150)
plt.show()

print("Gráfico 3 salvo: grafico_03_top10_fornecedores.png")

# =====================================================
# GRÁFICO 4 — Distribuição dos valores pagos (histograma)
# =====================================================
# O que mostra: a maioria dos contratos é pequena,
# poucos contratos concentram valores gigantes

# Remove zeros pra não distorcer o gráfico
df_sem_zero = df_total[df_total["valor_pago"] > 0]

plt.figure()

sns.histplot(
    df_sem_zero["valor_pago"],
    bins=40,
    color="steelblue",
    kde=True  # linha de densidade por cima
)

plt.title("Distribuição dos Valores Pagos por Contrato")
plt.xlabel("Valor Pago (R$)")
plt.ylabel("Quantidade de Contratos")
plt.tight_layout()
plt.savefig("grafico_04_distribuicao_valores.png", dpi=150)
plt.show()

print("Gráfico 4 salvo: grafico_04_distribuicao_valores.png")

# =====================================================
# GRÁFICO 5 — Fornecedores novos vs recorrentes por ano (barras empilhadas)
# =====================================================
# O que mostra: quantos fornecedores eram novos a cada ano

contagem_por_ano = df_total.groupby("ano")["fornecedor"].nunique().reset_index()
contagem_por_ano.columns = ["ano", "total_fornecedores"]

fornecedores_por_ano = (
    df_total.dropna(subset=["fornecedor"])
    .groupby("fornecedor")["ano"]
    .apply(list)
    .reset_index()
)

novos_por_ano = {}
vistos = set()

for ano in sorted(df_total["ano"].unique()):
    fornecedores_ano = set(
        df_total[df_total["ano"] == ano]["fornecedor"].dropna()
    )
    novos = fornecedores_ano - vistos
    novos_por_ano[ano] = len(novos)
    vistos.update(fornecedores_ano)

df_novos = pd.DataFrame({
    "ano": list(novos_por_ano.keys()),
    "fornecedores_novos": list(novos_por_ano.values())
})

df_novos = df_novos.merge(contagem_por_ano, on="ano")
df_novos["fornecedores_recorrentes"] = (
    df_novos["total_fornecedores"] - df_novos["fornecedores_novos"]
)

plt.figure()

plt.bar(df_novos["ano"], df_novos["fornecedores_recorrentes"],
        label="Recorrentes", color="steelblue")
plt.bar(df_novos["ano"], df_novos["fornecedores_novos"],
        bottom=df_novos["fornecedores_recorrentes"],
        label="Novos", color="coral")

plt.title("Fornecedores Novos vs Recorrentes por Ano")
plt.xlabel("Ano")
plt.ylabel("Quantidade de Fornecedores")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_05_novos_vs_recorrentes.png", dpi=150)
plt.show()

print("Gráfico 5 salvo: grafico_05_novos_vs_recorrentes.png")

# =====================================================
# GRÁFICO 6 — Taxa de execução média por ano
# =====================================================
# O que mostra: em média, qual % do que foi empenhado foi pago de fato

df_total["taxa_execucao"] = (
    df_total["valor_pago"] / df_total["valor_empenhado"]
).replace([float("inf"), -float("inf")], None)

taxa_por_ano = (
    df_total.groupby("ano")["taxa_execucao"]
    .mean()
    .reset_index()
)

plt.figure()

sns.barplot(
    data=taxa_por_ano,
    x="ano",
    y="taxa_execucao",
    palette="Greens"
)

# Linha de referência em 100%
plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, label="100%")

# Valor em cima de cada barra
for i, row in taxa_por_ano.iterrows():
    plt.text(
        i,
        row["taxa_execucao"] + 0.01,
        f'{row["taxa_execucao"]*100:.1f}%',
        ha="center",
        fontsize=10
    )

plt.title("Taxa de Execução Média por Ano\n(quanto do empenho virou pagamento)")
plt.xlabel("Ano")
plt.ylabel("Taxa Média")
plt.ylim(0, 1.2)
plt.legend()
plt.tight_layout()
plt.savefig("grafico_06_taxa_execucao.png", dpi=150)
plt.show()

print("Gráfico 6 salvo: grafico_06_taxa_execucao.png")

# =====================================================
# FIM
# =====================================================

print("\nTodos os gráficos gerados e salvos na pasta atual.")
