# Code Style

## Linting and Formatting

ORMAS-CI uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting, configured in `pyproject.toml`.

```bash
ruff check pyscf/ tests/        # Lint
ruff format pyscf/ tests/       # Format
```

### Ruff Configuration

| Setting | Value |
|---------|-------|
| Line length | 100 characters |
| Target Python | 3.11 |
| Rules | `E` (pycodestyle errors), `F` (pyflakes), `W` (pycodestyle warnings), `I` (isort), `N` (pep8-naming), `UP` (pyupgrade) |

## Type Checking

Type annotations are required on all function signatures. The project
uses [pyright](https://microsoft.github.io/pyright/) in basic mode:

```bash
pyright
```

Configuration in `pyproject.toml`:

```toml
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "basic"
include = ["pyscf/ormas_ci"]
```

## Docstrings

Use Google-style docstrings on all public functions and classes:

```python
def count_determinants(config: ORMASConfig) -> int:
    """Count the number of determinants in the ORMAS-restricted space.

    Enumerates valid electron distributions across subspaces and counts
    the resulting determinant combinations without storing them.

    Args:
        config: ORMAS configuration specifying subspaces and constraints.

    Returns:
        Total number of allowed determinants.

    Raises:
        ValueError: If the configuration is invalid.
    """
```

Include `Args`, `Returns`, and `Raises` sections where applicable.
The Sphinx build uses `napoleon` to parse these into the API reference.

## Naming Conventions

The codebase uses domain-specific abbreviations consistently:

| Name | Meaning |
|------|---------|
| `ncas` | Number of active orbitals |
| `nelecas` | Active electrons as `(n_alpha, n_beta)` |
| `n_alpha`, `n_beta` | Alpha/beta electron counts |
| `n_det` | Number of determinants |
| `h1e` | One-electron integrals (2D array) |
| `h2e` | Two-electron integrals (4D or compressed) |
| `eri` | Electron repulsion integrals (same as `h2e`) |
| `ecore` | Core energy (frozen electrons + nuclear repulsion) |
| `ci` | CI coefficient vector |
| `rdm1`, `rdm2` | 1- and 2-particle reduced density matrices |
| `norb` | Number of orbitals (alias for `ncas` in PySCF interface) |
| `nelec` | Number of electrons (alias for `nelecas` in PySCF interface) |

Variable names follow Python conventions: `snake_case` for functions
and variables, `PascalCase` for classes.

## Import Ordering

Ruff's `I` rule enforces isort-compatible import ordering:

1. Standard library
2. Third-party packages (numpy, scipy, pyscf)
3. Local imports (`pyscf.ormas_ci.*`)

Each group is separated by a blank line.
