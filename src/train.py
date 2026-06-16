"""
Treino do classificador de execução problemática.

Validação out-of-time:
  - Treino: 2021–2023
  - Teste:  2024–2025

MLflow: 6 experimentos comparados (RF, GBM, Logística × hiperparâmetros).
Melhor modelo salvo em models/model.pkl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.load_data import load_jsons
from src.features import FEATURE_COLS, build_classification_features

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    {
        "name": "rf_baseline",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": "None"},
    },
    {
        "name": "rf_shallow",
        "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5},
    },
    {
        "name": "rf_deep_200",
        "model": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 15},
    },
    {
        "name": "gbm_baseline",
        "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "params": {"model_type": "GradientBoosting", "n_estimators": 100, "learning_rate": 0.1},
    },
    {
        "name": "gbm_conservative",
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        "params": {"model_type": "GradientBoosting", "n_estimators": 200, "learning_rate": 0.05},
    },
    {
        "name": "logistic_baseline",
        "model": LogisticRegression(max_iter=500, random_state=42),
        "params": {"model_type": "LogisticRegression", "max_iter": 500},
    },
]


def temporal_split(df: pd.DataFrame):
    train = df[df["ano"] <= 2023].copy()
    test = df[df["ano"] >= 2024].copy()
    return train, test


def evaluate(pipe, X_test, y_test) -> dict:
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def run():
    print("Carregando e engenhando features...")
    df = load_jsons()
    df = build_classification_features(df)

    train, test = temporal_split(df)
    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    print(f"\nTreino (2021–2023): {len(train)} registros | {y_train.mean():.1%} problemáticos")
    print(f"Teste  (2024–2025): {len(test)} registros  | {y_test.mean():.1%} problemáticos\n")

    # 1. Configuração da infraestrutura de telemetria
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("facape-execucao-problematica")

    best_f1, best_run_id, best_pipe = -1.0, None, None

    for exp in EXPERIMENTS:
        with mlflow.start_run(run_name=exp["name"]):
            
            # 2. Injeção condicional de processamento matemático
            if exp["params"]["model_type"] == "LogisticRegression":
                steps = [("scaler", StandardScaler()), ("model", exp["model"])]
            else:
                steps = [("model", exp["model"])]
            
            pipe = Pipeline(steps)
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_params(exp["params"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model")

            print(
                f"[{exp['name']:22s}] F1={metrics['f1']:.4f} | "
                f"AUC={metrics['roc_auc']:.4f} | "
                f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_run_id = mlflow.active_run().info.run_id
                best_pipe = pipe

    joblib.dump(best_pipe, MODELS_DIR / "model.pkl")
    print(f"\nMelhor pipeline salvo → models/model.pkl (run_id={best_run_id}, F1={best_f1})")

    # Feature importance
    try:
        importances = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": best_pipe.named_steps["model"].feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nImportância das features:")
        print(importances.to_string(index=False))
    except AttributeError:
        pass

    # Relatório detalhado no conjunto de teste
    preds = best_pipe.predict(X_test)
    print("\nRelatório de classificação (conjunto de teste 2024–2025):")
    print(classification_report(y_test, preds, target_names=["Execução OK", "Problemático"]))


if __name__ == "__main__":
    run()