PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: setup test lint typecheck data part1 part2 serve clean

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest --cov=src/odl --cov-report=term-missing

lint:
	$(PY) -m ruff check src experiments tests
	$(PY) -m ruff format --check src experiments tests

typecheck:
	$(PY) -m mypy --strict src/odl

data:
	bash scripts/download_obd.sh sample

data-full:
	bash scripts/download_obd.sh full

part1:
	$(PY) experiments/e01_ope_validation.py
	$(PY) experiments/e02_real_bandit.py
	$(PY) experiments/e02b_cross_policy_validation.py

part2:
	$(PY) experiments/e03_distribution_shift.py
	$(PY) experiments/e04_cql_comparison.py

serve:
	$(PY) -m uvicorn odl.service.app:app --host 127.0.0.1 --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
