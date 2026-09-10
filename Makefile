.PHONY: install test prepare-data validate-data baseline train evaluate api ui security-check smoke-prepare

PYTHON ?= python
CONFIG ?= configs/qlora.yaml
SOURCE ?= tests/fixtures/nyu
OUTPUT ?= data

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

security-check:
	$(PYTHON) -m room_vlm.security_check

prepare-data:
	$(PYTHON) scripts/prepare_nyu.py --source $(SOURCE) --config configs/base.yaml --output $(OUTPUT)

smoke-prepare:
	$(PYTHON) scripts/prepare_nyu.py --source tests/fixtures/nyu --config configs/base.yaml --output .tmp/room-vlm-test

validate-data:
	$(PYTHON) scripts/validate_dataset.py --config configs/base.yaml

baseline:
	$(PYTHON) scripts/run_baseline.py --config configs/base.yaml

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --split test --checkpoint $(CHECKPOINT)

api:
	$(PYTHON) -m uvicorn room_vlm.api.app:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) web/gradio_app.py
