# Convenience targets. Most operate on a single PROJECT (default: airplane);
# the *-all targets iterate over every project.
PROJECT ?= airplane
PROJECTS := airplane creditrisk hmda houseprice utilization

.PHONY: help install train serve test lint docker-build clean test-all lint-all

help:
	@echo "Targets (PROJECT=$(PROJECT)):"
	@echo "  install       install runtime + dev deps for PROJECT"
	@echo "  train         run the training pipeline for PROJECT"
	@echo "  serve         run the Flask app for PROJECT (gunicorn)"
	@echo "  test          run pytest for PROJECT"
	@echo "  lint          run ruff for PROJECT"
	@echo "  docker-build  build the Docker image for PROJECT"
	@echo "  test-all      run pytest for every project"
	@echo "  lint-all      run ruff for every project"
	@echo "  clean         remove caches and local mlflow output"

install:
	cd $(PROJECT) && pip install -r requirements-dev.txt

train:
	cd $(PROJECT) && python main.py

serve:
	cd $(PROJECT) && gunicorn app:app -b 0.0.0.0:8080

test:
	cd $(PROJECT) && pytest

lint:
	cd $(PROJECT) && ruff check .

docker-build:
	docker build -t $(PROJECT) ./$(PROJECT)

test-all:
	@for p in $(PROJECTS); do echo "== $$p =="; (cd $$p && pytest) || exit 1; done

lint-all:
	@ruff check $(PROJECTS)

clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d \( -name mlruns -o -name mlartifacts -o -name .pytest_cache \
		-o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "mlflow.db" -not -path "*/venv/*" -delete 2>/dev/null || true
	@echo "cleaned"
