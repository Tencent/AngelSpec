# Contributing to AngelSpec

Thank you for your interest in contributing to AngelSpec! This guide will help you get started.

## Development Setup

1. Clone the repository and create the conda environment:

```bash
git clone https://github.com/Tencent/angelspec.git
cd angelspec
./tools/build_conda.sh
micromamba activate angelspec
```

Or install into your current environment:

```bash
./tools/build_conda.sh current
```

2. Install the dev dependencies:

```bash
pip install -e ".[dev]"
```

3. (Optional) Install Flash Attention for full feature support:

```bash
pip install -e ".[fa]"
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks enforce this automatically:

```bash
pre-commit install
```

Key style rules:
- Line length: 100 characters
- Python 3.11+
- Absolute imports only (no relative imports)
- Imports sorted with `isort` (via Ruff)

Run the linter manually:

```bash
ruff check .
ruff format --check .
```

## Running Tests

Run the full test suite:

```bash
./tools/run_all_tests.sh
```

Or run individual tests with pytest:

```bash
pytest tests/test_eagle3_loss.py -v
```

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and ensure all tests pass.
3. Run `ruff check .` and `ruff format .` before committing.
4. Write a clear PR description explaining what changed and why.
5. Link any related issues.

## Reporting Issues

When filing a bug report, please include:
- Python version and OS
- GPU type and driver version
- Steps to reproduce the issue
- Full error traceback
