# Installation

## Quick Install

```bash
pip install pyscf-ormas-ci
```

This installs ORMAS-CI and its required dependencies (NumPy, SciPy, PySCF).

## From Source

```bash
git clone https://github.com/brycewestheimer/ormas-ci.git
cd ormas-ci
pip install -e .
```

## Optional Extras

ORMAS-CI provides several optional dependency groups:

| Extra | Install Command | What It Adds |
|-------|----------------|--------------|
| `dev` | `pip install -e ".[dev]"` | pytest, pytest-cov, ruff, build, pyright |
| `qdk` | `pip install -e ".[qdk]"` | qdk-chemistry for quantum resource estimation |
| `notebooks` | `pip install -e ".[notebooks]"` | Jupyter and matplotlib |
| `docs` | `pip install -e ".[docs]"` | Sphinx, RTD theme, MyST parser |

To install multiple extras:

```bash
pip install -e ".[dev,qdk]"
```

## Requirements

- **Python** >= 3.11
- **NumPy** >= 1.24
- **SciPy** >= 1.10
- **PySCF** >= 2.4

## Verifying the Installation

```python
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace
print("ORMAS-CI installed successfully")
```

For a more thorough check, run the quick-start example from
[Getting Started](../guides/getting_started.md).

## Common Issues

**PySCF build errors on macOS:** PySCF requires a C compiler for its
extension modules. On macOS, ensure Xcode command line tools are
installed (`xcode-select --install`). If using conda, `conda install pyscf`
often avoids build issues.

**NumPy version conflicts:** If you have an older NumPy pinned by another
package, pip may fail to resolve. Use `pip install --upgrade numpy>=1.24`
before installing ORMAS-CI, or use a fresh virtual environment.

**ImportError for pyscf namespace:** ORMAS-CI uses PySCF's namespace
package mechanism (`pyscf.ormas_ci`). If you see import errors, ensure
PySCF and ORMAS-CI are installed in the same Python environment and
that there is no stale `pyscf/ormas_ci` directory on your `PYTHONPATH`.
