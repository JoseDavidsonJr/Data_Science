import pandas as pd
import json
import os

pasta = r"C:\Facape_despesas"

dados_organizados = []

print("Arquivos encontrados:")
print(os.listdir(pasta))

for arquivo in os.listdir(pasta):
    if arquivo.lower().endswith(".json"):
        print(f"Lendo arquivo: {arquivo}")

        caminho_arquivo = os.path.join(pasta, arquivo)

        ano = arquivo.lower().replace("despesas", "").replace(".json", "")
        ano = "20" + ano

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        for item in dados:
            linha = {
                "id": item.get("id"),
                "fornecedor": item.get("fornecedor", {})
                                  .get("pessoa", {})
                                  .get("nome"),
                "ano": ano,
            }

            #  adiciona TODOS os outros campos automaticamente
            for chave, valor in item.items():
                if chave not in ["fornecedor", "valores"]:
                    linha[chave] = valor

            # inclui também os valores internos
            for chave, valor in item.get("valores", {}).items():
                linha[chave] = valor

            dados_organizados.append(linha)

df_total = pd.DataFrame(dados_organizados)

# =========================
# VISUALIZAÇÃO
# =========================
print("\n===== AMOSTRA (20 linhas) =====")
print(df_total.head(20))

print("\n===== INFO =====")
print(df_total.info())

# =========================
#  SOMA AUTOMÁTICA
# =========================

# seleciona automaticamente colunas numéricas
colunas_numericas = df_total.select_dtypes(include="number").columns

# agrupa por ano e soma tudo
resumo_ano = df_total.groupby("ano")[colunas_numericas].sum().reset_index()

print("\n===== TOTAL POR ANO =====")
print(resumo_ano)

# =========================
#  SALVAR
# =========================

df_total.to_csv("dados_completos.csv", index=False, encoding="utf-8-sig")
resumo_ano.to_csv("resumo_por_ano.csv", index=False, encoding="utf-8-sig")

print("\nArquivos salvos com sucesso!")