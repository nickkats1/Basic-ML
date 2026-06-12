"""Evaluate the persisted best classifier on a holdout set.

Writes ``metrics.json`` and a human-readable ``model_card.md`` alongside the
model artifact, and logs the evaluation run to MLflow.
"""

import json
from pathlib import Path
from typing import Any

import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from helpers.config import Config, get_config, load_object
from helpers.logger import logger


def evaluate(X_test: Any, y_test: Any, config: Config | None = None) -> dict[str, float]:
    """Score the persisted pipeline on the holdout set and emit reports.

    Args:
        X_test: Holdout features (raw; the pipeline preprocesses them).
        y_test: Holdout target labels.
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        A dict of classification metrics.
    """
    config = config or get_config()
    pipeline = load_object(config.model_path)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
    }
    logger.info("evaluation metrics: %s", metrics)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)

    _write_reports(metrics, pipeline, config)
    return metrics


def _write_reports(metrics: dict[str, float], pipeline: Any, config: Config) -> None:
    """Persist metrics.json and model_card.md next to the artifact."""
    out_dir = Path(config.model_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    estimator = pipeline.named_steps.get("model", pipeline)
    card = f"""# Model Card: {config.experiment_name}

## Task
Binary classification — target: `{config.target}`.

## Features
{", ".join(config.features)}

## Estimator
{type(estimator).__name__}

## Holdout metrics
- ROC-AUC: {metrics["roc_auc"]:.4f}
- Accuracy: {metrics["accuracy"]:.4f}
- F1: {metrics["f1"]:.4f}
- Test rows: {metrics["n_test"]}

## Notes
All preprocessing (standardization, one-hot encoding) is embedded in the
persisted scikit-learn pipeline. Metrics computed on a held-out split.
"""
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")
    logger.info("wrote metrics.json and model_card.md to %s", out_dir)
