"""
Perfilamento de risco de fornecedores — KMeans.

Pergunta de negócio: quais fornecedores têm perfil de alto risco
(baixa execução, histórico de anulações, alta retenção)?

3 experimentos MLflow (k=2,3,4) com inertia registrada.
Modelo final (k=3) salvo em models/cluster_model.pkl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.load_data import load_jsons
from src.features import CLUSTER_FEATURE_COLS, build_supplier_profile

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def run():
    df = load_jsons()
    perfil = build_supplier_profile(df)
    X = perfil[CLUSTER_FEATURE_COLS]

    mlflow.set_experiment("facape-supplier-clustering")

    print("Experimentos KMeans:\n")
    for k in [2, 3, 4]:
        with mlflow.start_run(run_name=f"kmeans_k{k}"):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", KMeans(n_clusters=k, random_state=42, n_init=10)),
            ])
            pipe.fit(X)
            inertia = pipe.named_steps["model"].inertia_

            mlflow.log_param("k", k)
            mlflow.log_metric("inertia", round(inertia, 2))
            mlflow.sklearn.log_model(pipe, "model")
            print(f"  k={k}: inertia={inertia:.2f}")

    # Modelo final: k=3
    pipe_final = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=3, random_state=42, n_init=10)),
    ])
    pipe_final.fit(X)
    perfil["cluster"] = pipe_final.predict(X)

    # Rotula clusters por taxa de execução média (menor taxa = maior risco)
    medias = perfil.groupby("cluster")["taxa_execucao_media"].mean().sort_values()
    mapa_risco = {
        medias.index[0]: "Alto Risco",
        medias.index[1]: "Médio Risco",
        medias.index[2]: "Baixo Risco",
    }
    perfil["perfil_risco"] = perfil["cluster"].map(mapa_risco)

    print("\nResumo por perfil de risco:")
    resumo = (
        perfil.groupby("perfil_risco")
        .agg(
            fornecedores=("fornecedor", "count"),
            valor_medio=("valor_total", "mean"),
            taxa_execucao=("taxa_execucao_media", "mean"),
            perc_anulado=("perc_anulado_medio", "mean"),
            anos_ativo=("anos_ativo", "mean"),
        )
        .round(3)
    )
    print(resumo)

    joblib.dump(pipe_final, MODELS_DIR / "cluster_model.pkl")

    # Gráficos
    ORDEM = ["Baixo Risco", "Médio Risco", "Alto Risco"]
    CORES = {"Baixo Risco": "seagreen", "Médio Risco": "steelblue", "Alto Risco": "coral"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for nome, cor in CORES.items():
        sub = perfil[perfil["perfil_risco"] == nome]
        axes[0].scatter(
            sub["anos_ativo"], sub["valor_total_log"],
            label=nome, color=cor, alpha=0.7, s=80
        )
    axes[0].set_xlabel("Anos Ativo (2021–2025)")
    axes[0].set_ylabel("Log(Valor Total Recebido)")
    axes[0].set_title("Perfil de Risco por Recorrência e Porte")
    axes[0].legend()

    contagem = perfil["perfil_risco"].value_counts().reindex(ORDEM)
    sns.barplot(
        x=contagem.index, y=contagem.values,
        palette=[CORES[c] for c in ORDEM], ax=axes[1]
    )
    for i, v in enumerate(contagem.values):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=11)
    axes[1].set_title("Quantidade de Fornecedores por Perfil")
    axes[1].set_ylabel("Quantidade")

    plt.suptitle("Perfilamento de Risco de Fornecedores FACAPE (KMeans, k=3)")
    plt.tight_layout()
    out = REPORTS_DIR / "grafico_clusters.png"
    plt.savefig(out, dpi=150)
    print(f"\nGráfico salvo: {out}")


if __name__ == "__main__":
    run()
