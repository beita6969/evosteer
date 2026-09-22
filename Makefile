.PHONY: sync test lint format-check typecheck wheels check

sync:
	uv sync

test:
	PYTHONFAULTHANDLER=1 uv run pytest -q

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

wheels:
	uv build --wheel
	uv build --wheel packages/private-evaluation
	uv run python scripts/check_model_wheel.py dist/skillev-0.1.0-py3-none-any.whl

check: format-check lint typecheck test wheels
