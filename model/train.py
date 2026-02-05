import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True,
                   help="Caminho do CSV. Ex: s3://bucket/data/creditcard.csv ou caminho local.")
    p.add_argument("--output_dir", type=str, default="outputs",
                   help="Diretório onde salvar modelo e configs.")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Threshold para classificar fraude.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

# 1) Ler dados (local, S3 ou SageMaker channel)
input_path = args.input

if os.path.isdir(input_path):
    # SageMaker training channel: procura o CSV dentro do diretório
    files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    if not files:
        raise ValueError(f"Nenhum CSV encontrado em {input_path}")
    input_path = os.path.join(input_path, files[0])

df = pd.read_csv(input_path)

if "Class" not in df.columns:
    raise ValueError("CSV precisa ter a coluna 'Class' (0/1).")

X = df.drop(columns=["Class"])
y = df["Class"]

    # 2) Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # 3) Modelo baseline (scaler + logistic)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_train)

    # 4) Predições e métricas
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = float(roc_auc_score(y_test, y_proba))

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    alerts = int(y_pred.sum())
    frauds_in_test = int(y_test.sum())

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "fraud_rate": float(y.mean()),
            "test_frauds": frauds_in_test
        },
        "model": {
            "type": "LogisticRegression(class_weight=balanced) + StandardScaler",
            "threshold": float(args.threshold),
            "roc_auc": roc_auc,
            "precision_1": float(p),
            "recall_1": float(r),
            "f1_1": float(f1),
            "alerts": alerts
        },
        "confusion_matrix": cm,
        "classification_report": report
    }

    # 5) Salvar artefatos
    model_path = os.path.join(args.output_dir, "model.joblib")
    config_path = os.path.join(args.output_dir, "config.json")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    joblib.dump(pipe, model_path)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(args.threshold)}, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Treino finalizado")
    print("Model:", model_path)
    print("Config:", config_path)
    print("Metrics:", metrics_path)
    print("\nResumo:")
    print("ROC-AUC:", roc_auc)
    print("Precision(1):", float(p), "Recall(1):", float(r), "Alerts:", alerts)


if __name__ == "__main__":
    main()
