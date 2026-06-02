import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# LEITURA DOS JSONS
# =====================================================

pasta = os.path.dirname(os.path.abspath(__file__))

dados_organizados = []

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(".json"):
        caminho_arquivo = os.path.join(pasta, arquivo)

        ano = arquivo.lower().replace("despesas", "").replace(".json", "")
        ano = "20" + ano

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        for item in dados:
            linha = {
                "id_registro":    item.get("id"),
                "fornecedor":     item.get("fornecedor", {}).get("pessoa", {}).get("nome"),
                "cpf_cnpj":       item.get("fornecedor", {}).get("pessoa", {}).get("cpfCnpj"),
                "ano":            int("20" + arquivo.lower().replace("despesas","").replace(".json","")),
                "valor_empenhado": item.get("valorEmpenhado"),
                "valor_liquidado": item.get("valorLiquidado"),
                "valor_pago":      item.get("valorPago"),
            }
            dados_organizados.append(linha)

df = pd.DataFrame(dados_organizados)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Remove linhas sem fornecedor ou com empenhado zerado
df = df.dropna(subset=["fornecedor", "cpf_cnpj"])
df = df[df["valor_empenhado"] > 0]

# Tipo do fornecedor: CPF (11 dígitos) = PF, CNPJ (14) = PJ
def tipo_fornecedor(cpf_cnpj):
    apenas_numeros = str(cpf_cnpj).replace(".", "").replace("-", "").replace("/", "").strip()
    return 0 if len(apenas_numeros) == 11 else 1  # 0 = PF, 1 = PJ

df["tipo"] = df["cpf_cnpj"].apply(tipo_fornecedor)

# Taxa de execução: quanto do empenho virou pagamento
df["taxa_execucao"] = df["valor_pago"] / df["valor_empenhado"]

# Saldo pendente
df["saldo_pendente"] = df["valor_liquidado"] - df["valor_pago"]

print("=" * 50)
print("DADOS CARREGADOS")
print("=" * 50)
print(f"Total de registros: {len(df)}")
print(f"Pessoas físicas (PF): {(df['tipo'] == 0).sum()}")
print(f"Pessoas jurídicas (PJ): {(df['tipo'] == 1).sum()}")
print()

# =====================================================
# MODELO 1 — CLUSTERING DE FORNECEDORES (KMeans)
# =====================================================
# Agrupa fornecedores por perfil usando:
# - valor total recebido nos 5 anos
# - quantidade de anos que apareceu
# - taxa de execução média
# - tipo (PF ou PJ)

print("=" * 50)
print("MODELO 1 — CLUSTERING DE FORNECEDORES")
print("=" * 50)

# Agrega por fornecedor
perfil = (
    df.groupby("fornecedor")
    .agg(
        valor_total=("valor_pago", "sum"),
        quantidade_anos=("ano", "nunique"),
        taxa_execucao_media=("taxa_execucao", "mean"),
        tipo=("tipo", "first"),
    )
    .reset_index()
)

# Features do clustering
X_cluster = perfil[["valor_total", "quantidade_anos", "taxa_execucao_media", "tipo"]]

# Normaliza os dados — necessário pro KMeans funcionar bem
# (sem isso, valor_total domina tudo por ter escala muito maior)
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Treina com 3 grupos
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
perfil["grupo"] = kmeans.fit_predict(X_cluster_scaled)

# Renomeia os grupos com base no valor total médio de cada um
medias = perfil.groupby("grupo")["valor_total"].mean().sort_values()
mapa_grupos = {
    medias.index[0]: "Pequeno",
    medias.index[1]: "Médio",
    medias.index[2]: "Grande",
}
perfil["grupo_nome"] = perfil["grupo"].map(mapa_grupos)

print("\nResumo dos grupos encontrados:")
print(
    perfil.groupby("grupo_nome")
    .agg(
        quantidade_fornecedores=("fornecedor", "count"),
        valor_total_medio=("valor_total", "mean"),
        anos_medio=("quantidade_anos", "mean"),
        taxa_execucao_media=("taxa_execucao_media", "mean"),
    )
    .round(2)
)

print("\nExemplos de cada grupo:")
for grupo in ["Pequeno", "Médio", "Grande"]:
    exemplos = perfil[perfil["grupo_nome"] == grupo]["fornecedor"].head(3).tolist()
    print(f"  {grupo}: {exemplos}")

# Gráfico do clustering
plt.figure(figsize=(10, 6))
cores = {"Pequeno": "steelblue", "Médio": "seagreen", "Grande": "coral"}
for grupo, cor in cores.items():
    subset = perfil[perfil["grupo_nome"] == grupo]
    plt.scatter(
        subset["quantidade_anos"],
        subset["valor_total"],
        label=grupo,
        color=cor,
        alpha=0.7,
        s=80
    )

plt.title("Clustering de Fornecedores por Perfil")
plt.xlabel("Quantidade de Anos que Apareceu")
plt.ylabel("Valor Total Recebido (R$)")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafico_07_clustering.png", dpi=150)
plt.show()
print("\nGráfico salvo: grafico_07_clustering.png")

# Gráfico 07b — Tamanho de cada grupo (quantos fornecedores)
plt.figure(figsize=(7, 4))
contagem_grupos = perfil["grupo_nome"].value_counts().reindex(["Pequeno", "Médio", "Grande"])
cores_bar = ["steelblue", "seagreen", "coral"]
sns.barplot(x=contagem_grupos.index, y=contagem_grupos.values, palette=cores_bar)
for i, v in enumerate(contagem_grupos.values):
    plt.text(i, v + 1, str(v), ha="center", fontsize=11)
plt.title("Quantidade de Fornecedores por Grupo")
plt.xlabel("Grupo")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig("grafico_07b_tamanho_grupos.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_07b_tamanho_grupos.png")

# Gráfico 07c — Boxplot do valor total por grupo
# Mostra a dispersão dos valores dentro de cada grupo
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=perfil,
    x="grupo_nome",
    y="valor_total",
    order=["Pequeno", "Médio", "Grande"],
    palette={"Pequeno": "steelblue", "Médio": "seagreen", "Grande": "coral"}
)
plt.title("Distribuição do Valor Total por Grupo de Fornecedor")
plt.xlabel("Grupo")
plt.ylabel("Valor Total Recebido (R$)")
plt.tight_layout()
plt.savefig("grafico_07c_boxplot_grupos.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_07c_boxplot_grupos.png")

# =====================================================
# MODELO 2 — CLASSIFICAÇÃO
# Pergunta: esse contrato vai ser pago integralmente?
# =====================================================
# Target: 1 = taxa >= 90% (pago quase tudo)
#         0 = taxa < 90%  (ficou saldo pendente)

print()
print("=" * 50)
print("MODELO 2 — CLASSIFICAÇÃO (pago integralmente?)")
print("=" * 50)

df["pago_integralmente"] = (df["taxa_execucao"] >= 0.9).astype(int)

print(f"\nContratos pagos integralmente (>=90%): {df['pago_integralmente'].sum()}")
print(f"Contratos com saldo pendente (<90%):   {(df['pago_integralmente'] == 0).sum()}")

# Features e target
X_clf = df[["valor_empenhado", "tipo", "ano"]]
y_clf = df["pago_integralmente"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

modelo_clf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_clf.fit(X_train, y_train)

previsoes_clf = modelo_clf.predict(X_test)
acc = accuracy_score(y_test, previsoes_clf)

print(f"\nAcurácia do modelo: {acc:.2%}")
print("\nRelatório detalhado:")
print(classification_report(y_test, previsoes_clf, target_names=["Saldo pendente", "Pago integralmente"]))

# Importância das features
importancias = pd.DataFrame({
    "feature": ["Valor Empenhado", "Tipo (PF/PJ)", "Ano"],
    "importancia": modelo_clf.feature_importances_
}).sort_values("importancia", ascending=False)

print("O que mais influencia na previsão:")
print(importancias.to_string(index=False))

# Gráfico de importância
plt.figure(figsize=(8, 4))
sns.barplot(data=importancias, x="importancia", y="feature", palette="Blues_r")
plt.title("O que mais influencia se o contrato será pago integralmente")
plt.xlabel("Importância")
plt.ylabel("")
plt.tight_layout()
plt.savefig("grafico_08_importancia_classificacao.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_08_importancia_classificacao.png")

# Gráfico 08b — Matriz de confusão
# Mostra: dos contratos que o modelo previu como "pago", quantos realmente foram?
# E dos que previu como "pendente", quantos realmente foram pendentes?
cm = confusion_matrix(y_test, previsoes_clf)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Previsto: Pendente", "Previsto: Pago"],
    yticklabels=["Real: Pendente", "Real: Pago"]
)
plt.title("Matriz de Confusão — Classificação")
plt.tight_layout()
plt.savefig("grafico_08b_matriz_confusao.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_08b_matriz_confusao.png")

# Gráfico 08c — % pago integralmente por tipo (PF vs PJ)
taxa_por_tipo = (
    df.groupby("tipo")["pago_integralmente"]
    .mean()
    .reset_index()
)
taxa_por_tipo["tipo_nome"] = taxa_por_tipo["tipo"].map({0: "Pessoa Física (PF)", 1: "Pessoa Jurídica (PJ)"})
taxa_por_tipo["percentual"] = taxa_por_tipo["pago_integralmente"] * 100

plt.figure(figsize=(7, 4))
sns.barplot(data=taxa_por_tipo, x="tipo_nome", y="percentual", palette=["steelblue", "seagreen"])
for i, row in taxa_por_tipo.iterrows():
    plt.text(i, row["percentual"] + 0.5, f'{row["percentual"]:.1f}%', ha="center", fontsize=11)
plt.title("% de Contratos Pagos Integralmente — PF vs PJ")
plt.xlabel("")
plt.ylabel("% Pago Integralmente")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("grafico_08c_pago_por_tipo.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_08c_pago_por_tipo.png")

# Gráfico 08d — % pago integralmente por ano
taxa_por_ano = (
    df.groupby("ano")["pago_integralmente"]
    .mean()
    .reset_index()
)
taxa_por_ano["percentual"] = taxa_por_ano["pago_integralmente"] * 100

plt.figure(figsize=(8, 4))
sns.barplot(data=taxa_por_ano, x="ano", y="percentual", color="steelblue")
for i, row in taxa_por_ano.iterrows():
    plt.text(i, row["percentual"] + 0.5, f'{row["percentual"]:.1f}%', ha="center", fontsize=10)
plt.title("% de Contratos Pagos Integralmente por Ano")
plt.xlabel("Ano")
plt.ylabel("% Pago Integralmente")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("grafico_08d_pago_por_ano.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_08d_pago_por_ano.png")

# Exemplo de previsão
print("\nExemplo de previsão:")
novo = pd.DataFrame({
    "valor_empenhado": [500_000],
    "tipo": [1],       # 1 = PJ (empresa)
    "ano": [2025]
})
resultado = modelo_clf.predict(novo)[0]
probabilidade = modelo_clf.predict_proba(novo)[0][resultado]
print(f"  Empenho de R$500.000 para uma empresa em 2025:")
print(f"  → {'Pago integralmente' if resultado == 1 else 'Vai ter saldo pendente'} ({probabilidade:.0%} de confiança)")

# =====================================================
# MODELO 3 — REGRESSÃO
# Pergunta: dado um empenho de R$X, quanto vai ser pago?
# =====================================================

print()
print("=" * 50)
print("MODELO 3 — REGRESSÃO (quanto vai ser pago?)")
print("=" * 50)

X_reg = df[["valor_empenhado", "tipo", "ano"]]
y_reg = df["valor_pago"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

modelo_reg = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_reg.fit(X_train, y_train)

previsoes_reg = modelo_reg.predict(X_test)
mae = mean_absolute_error(y_test, previsoes_reg)

print(f"\nErro médio absoluto (MAE): R$ {mae:,.2f}")
print("(em média, o modelo erra esse valor pra cima ou pra baixo)")

# Gráfico real vs previsto
plt.figure(figsize=(8, 6))
plt.scatter(y_test, previsoes_reg, alpha=0.4, color="steelblue", s=40)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red", linewidth=1.5, linestyle="--", label="Perfeito"
)
plt.title("Regressão — Valor Real vs Valor Previsto")
plt.xlabel("Valor Real Pago (R$)")
plt.ylabel("Valor Previsto pelo Modelo (R$)")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_09_regressao_real_vs_previsto.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_09_regressao_real_vs_previsto.png")

# Gráfico 09b — Distribuição dos erros
# Mostra se o modelo erra mais pra cima ou pra baixo
erros = y_test.values - previsoes_reg
plt.figure(figsize=(8, 4))
sns.histplot(erros, bins=40, color="steelblue", kde=True)
plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Erro zero")
plt.title("Distribuição dos Erros da Regressão\n(negativo = modelo superestimou, positivo = subestimou)")
plt.xlabel("Erro (R$)")
plt.ylabel("Quantidade")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_09b_distribuicao_erros.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_09b_distribuicao_erros.png")

# Gráfico 09c — Erro médio por faixa de valor empenhado
# Mostra se o modelo é melhor em contratos pequenos ou grandes
df_teste = X_test.copy()
df_teste["valor_pago_real"] = y_test.values
df_teste["valor_pago_previsto"] = previsoes_reg
df_teste["erro_absoluto"] = abs(df_teste["valor_pago_real"] - df_teste["valor_pago_previsto"])

# Divide em faixas de valor empenhado
df_teste["faixa_empenhado"] = pd.qcut(
    df_teste["valor_empenhado"],
    q=4,
    labels=["Pequeno\n(Q1)", "Médio-baixo\n(Q2)", "Médio-alto\n(Q3)", "Grande\n(Q4)"]
)

erro_por_faixa = df_teste.groupby("faixa_empenhado")["erro_absoluto"].mean().reset_index()

plt.figure(figsize=(8, 4))
sns.barplot(data=erro_por_faixa, x="faixa_empenhado", y="erro_absoluto", color="coral")
plt.title("Erro Médio da Regressão por Faixa de Valor Empenhado")
plt.xlabel("Faixa de Valor Empenhado")
plt.ylabel("Erro Médio Absoluto (R$)")
plt.tight_layout()
plt.savefig("grafico_09c_erro_por_faixa.png", dpi=150)
plt.show()
print("Gráfico salvo: grafico_09c_erro_por_faixa.png")

# Exemplo de previsão
print("\nExemplo de previsão:")
novo_reg = pd.DataFrame({
    "valor_empenhado": [500_000],
    "tipo": [1],
    "ano": [2025]
})
estimativa = modelo_reg.predict(novo_reg)[0]
print(f"  Empenho de R$500.000 para uma empresa em 2025:")
print(f"  → Estimativa de pagamento: R$ {estimativa:,.2f}")

# =====================================================
# FIM
# =====================================================

print()
print("=" * 50)
print("TODOS OS MODELOS CONCLUÍDOS")
print("Gráficos salvos: 07, 07b, 07c, 08, 08b, 08c, 08d, 09, 09b, 09c")
print("=" * 50)
