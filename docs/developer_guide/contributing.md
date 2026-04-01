# Contributing

Thank you for your interest in contributing to ORMAS-CI. This guide
expands on the brief [CONTRIBUTING.md](https://github.com/brycewestheimer/ormas-ci/blob/main/CONTRIBUTING.md)
in the repository root.

## Development Setup

```bash
git clone https://github.com/brycewestheimer/ormas-ci.git
cd ormas-ci
pip install -e ".[dev]"
```

This installs the package in editable mode with test and lint tooling
(pytest, ruff, pyright, build).

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This runs ruff linting and formatting automatically on each commit.

### Optional: QDK Integration

If working on the QDK integration, install the `qdk` extra:

```bash
pip install -e ".[dev,qdk]"
```

## Branch and PR Workflow

1. Fork the repository and clone your fork
2. Create a feature branch from `main`
3. Make your changes (see [Code Style](code_style.md) and
   [Testing](testing.md))
4. Ensure all tests pass: `pytest`
5. Ensure linting is clean: `ruff check pyscf/ tests/`
6. Ensure type checking passes: `pyright`
7. Open a pull request against `main` with a clear description

### PR Expectations

- Add tests for any new functionality
- Update documentation if the public API changes
- Keep commits focused -- one logical change per commit
- CI must pass (lint, type check, tests on Python 3.11/3.12/3.13,
  docs build)

## What to Work On

- Check the [issue tracker](https://github.com/brycewestheimer/ormas-ci/issues)
  for open issues
- See the [Development Roadmap](../design/future.md) for planned work
- Small improvements (documentation, test coverage, error messages) are
  always welcome

## Repository Structure

```
ormas-ci/
    pyscf/ormas_ci/     Source code (PySCF namespace package)
    ormas_ci/           Backward-compat shim
    tests/              Test suite
    examples/           Example scripts (01-09)
    benchmarks/         Performance benchmarks
    docs/               Sphinx documentation
```

See [Architecture Overview](architecture_overview.md) for a walkthrough
of the source code modules.
