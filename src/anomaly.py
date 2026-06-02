"""
Detecção de anomalias em contratos públicos — Isolation Forest.

Pergunta de negócio: quais contratos fogem do padrão financeiro histórico
da instituição? (alto valor anulado, retenção atípica, saldo não executado)

3 experimentos MLflow com diferentes níveis de contaminação.
Modelo final salvo em models/anomaly_model.pkl.
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
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.load_data import load_jsons
from src.features import ANOMALY_FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def run():
    df = load_jsons()
    df["valor_empenhado_log"] = np.log1p(df["valor_empenhado"])
    X = df[ANOMALY_FEATURE_COLS]

    mlflow.set_experiment("facape-anomaly-detection")

    print("Experimentos Isolation Forest:\n")
    for contamination in [0.05, 0.10, 0.15]:
        with mlflow.start_run(run_name=f"isoforest_cont_{int(contamination * 100)}pct"):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", IsolationForest(
                    contamination=contamination, random_state=42, n_estimators=100
                )),
            ])
            pipe.fit(X)
            labels = pipe.predict(X)
            n_anomalias = int((labels == -1).sum())

            mlflow.log_param("contamination", contamination)
            mlflow.log_metric("n_anomalias", n_anomalias)
            mlflow.log_metric("perc_anomalias", round(n_anomalias / len(df), 4))
            mlflow.sklearn.log_model(pipe, "model")

            print(f"  contamination={contamination:.2f} → {n_anomalias} anomalias ({n_anomalias / len(df):.1%})")

    # Modelo final: contaminação 10%
    pipe_final = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(contamination=0.10, random_state=42, n_estimators=100)),
    ])
    pipe_final.fit(X)
    df["anomalia"] = (pipe_final.predict(X) == -1).astype(int)

    joblib.dump(pipe_final, MODELS_DIR / "anomaly_model.pkl")

    # Top contratos anômalos
    anomalos = df[df["anomalia"] == 1].sort_values("valor_empenhado", ascending=False)
    print(f"\nTop 10 contratos anômalos (maior valor empenhado):")
    cols = ["fornecedor", "ano", "valor_empenhado", "taxa_execucao", "perc_anulado", "perc_retido"]
    print(anomalos[cols].head(10).to_string(index=False))

    # Gráfico
    normais = df[df["anomalia"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        normais["valor_empenhado_log"], normais["taxa_execucao"],
        alpha=0.5, color="steelblue", s=50, label="Normal"
    )
    axes[0].scatter(
        anomalos["valor_empenhado_log"], anomalos["taxa_execucao"],
        alpha=0.9, color="red", s=80, marker="x", linewidths=2, label="Anômalo"
    )
    axes[0].set_xlabel("Log(Valor Empenhado)")
    axes[0].set_ylabel("Taxa de Execução")
    axes[0].set_title("Anomalias: Valor Empenhado vs Taxa de Execução")
    axes[0].legend()

    axes[1].scatter(
        normais["perc_anulado"], normais["perc_retido"],
        alpha=0.5, color="steelblue", s=50, label="Normal"
    )
    axes[1].scatter(
        anomalos["perc_anulado"], anomalos["perc_retido"],
        alpha=0.9, color="red", s=80, marker="x", linewidths=2, label="Anômalo"
    )
    axes[1].set_xlabel("% Anulado / Empenhado")
    axes[1].set_ylabel("% Retido / Empenhado")
    axes[1].set_title("Anomalias: Anulação vs Retenção")
    axes[1].legend()

    plt.suptitle("Detecção de Anomalias — Contratos FACAPE (Isolation Forest, contaminação=10%)")
    plt.tight_layout()
    out = REPORTS_DIR / "grafico_anomalias.png"
    plt.savefig(out, dpi=150)
    print(f"\nGráfico salvo: {out}")


if __name__ == "__main__":
    run()
