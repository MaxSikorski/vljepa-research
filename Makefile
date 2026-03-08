.PHONY: setup lint test train-ijepa train-vjepa train-vljepa eval serve clean

# Environment
setup:
	./scripts/setup_env.sh

# Code quality
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Training (local smoke tests)
train-ijepa-tiny:
	python -m src.ijepa.train --config configs/ijepa/pretrain_tiny.yaml

train-vjepa-tiny:
	python -m src.vjepa.train --config configs/vjepa/pretrain_tiny.yaml

train-vljepa-tiny:
	python -m src.vljepa.train --config configs/vljepa/pretrain_tiny.yaml

# Training (cloud scale)
train-ijepa:
	python -m src.ijepa.train --config configs/ijepa/pretrain_vitb16.yaml

train-vljepa:
	python -m src.vljepa.train --config configs/vljepa/pretrain.yaml

# Evaluation
eval-ijepa:
	python -m src.ijepa.eval --config configs/ijepa/pretrain_vitb16.yaml

eval-vljepa:
	python -m src.vljepa.eval --config configs/vljepa/pretrain.yaml

# Deployment
serve:
	uvicorn deployments.api.server:app --host 0.0.0.0 --port 8000 --reload

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .mypy_cache

# Downloads
download-checkpoints:
	./scripts/download_checkpoints.sh

download-datasets:
	./scripts/download_datasets.sh
