"""End-to-end training pipeline for the airplane regression project.

Runs once: ingest -> clean -> split -> train/tune (MLflow) -> evaluate ->
sample prediction. Data is fetched and split a single time and threaded
through each stage.
"""

from helpers.config import get_config
from helpers.logger import logger
from src.data.data_ingestion import fetch_raw_data
from src.data.data_transformation import build_preprocessor, split
from src.data.feature_engineering import clean
from src.models.model_evaluation import evaluate
from src.models.model_trainer import train
from src.models.predict import predict


def main() -> None:
    """Execute the full training and evaluation pipeline."""
    config = get_config()

    # Ingest + clean (once).
    data = clean(fetch_raw_data(config), config)

    # Single aligned split + preprocessor fit on the training features only.
    X_train, X_test, y_train, y_test = split(data, config)
    preprocessor = build_preprocessor(X_train, config)

    # Train, tune, log to MLflow, and persist the best pipeline.
    summary = train(X_train, X_test, y_train, y_test, preprocessor, config)
    logger.info("best model: %s", summary["best_model"])

    # Evaluate the persisted pipeline on the real holdout set.
    metrics = evaluate(X_test, y_test, config)
    logger.info("holdout metrics: %s", metrics)

    # Sample prediction using the first holdout row's raw features.
    prediction = predict(X_test.iloc[0].to_dict(), config)
    logger.info("sample prediction: %s (actual=%s)", prediction, y_test.iloc[0])


if __name__ == "__main__":
    main()
