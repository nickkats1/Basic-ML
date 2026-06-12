"""Flask serving app.

Routes:
    GET  /         -> input form (numeric vs categorical fields from the pipeline)
    POST /predict  -> render the prediction
    GET  /health   -> liveness/readiness probe (also reports model availability)
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from helpers.config import get_config
from helpers.logger import logger
from src.models.predict import feature_schema, predict

app = Flask(__name__)
config = get_config()
TARGET = config.target
TITLE = config.app_title
NUMERIC, CATEGORICAL = feature_schema(config)


@app.route("/")
def home():
    """Render the prediction form."""
    return render_template(
        "index.html",
        numeric=NUMERIC,
        categorical=CATEGORICAL,
        target=TARGET,
        title=TITLE,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    """Parse submitted features and render the prediction."""
    try:
        features: dict[str, object] = {}
        for name in NUMERIC:
            features[name] = float(request.form[name])
        for name in CATEGORICAL:
            features[name] = request.form[name]
    except (KeyError, ValueError) as exc:
        return (
            render_template("results.html", error=f"Invalid input: {exc}", title=TITLE),
            400,
        )

    prediction = predict(features, config)
    logger.info("served prediction: %s", prediction)
    return render_template(
        "results.html", prediction=prediction, target=TARGET, title=TITLE
    )


@app.route("/health")
def health():
    """Report service health and whether the model artifact exists."""
    return jsonify(status="ok", model_loaded=Path(config.model_path).exists())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
