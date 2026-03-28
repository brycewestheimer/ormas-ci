# Contributing to ORMAS-CI

Thank you for your interest in contributing to ORMAS-CI.

## Development Setup

```bash
git clone https://github.com/brycewestheimer/ormas-ci.git
cd ormas-ci
pip install -e ".[dev]"
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

This runs ruff linting and formatting automatically on each commit.

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=ormas_ci --cov-report=term-missing
```

## Code Style

- **Linter/formatter:** [Ruff](https://docs.astral.sh/ruff/) (configuration in `pyproject.toml`)
- **Docstrings:** Google style, required on all public functions and classes
- **Type annotations:** Required on all function signatures
- **Line length:** 100 characters

Run the linter:

```bash
ruff check src/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Ensure all tests pass and linting is clean
3. Add tests for any new functionality
4. Update documentation if the public API changes
5. Open a pull request with a clear description of the changes
