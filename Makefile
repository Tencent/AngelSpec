.PHONY: install install-dev test lint format docs docs-serve clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

docs:
	$(MAKE) -C docs html

docs-serve:
	$(MAKE) -C docs serve

clean:
	rm -rf docs/_build build dist *.egg-info
