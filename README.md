# ORMAS-CI

Occupation-Restricted Multiple Active Space Configuration Interaction solver as a PySCF fcisolver plugin.

[![CI](https://github.com/brycewestheimer/ormas-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/brycewestheimer/ormas-ci/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PySCF Plugin](https://img.shields.io/badge/PySCF-fcisolver%20plugin-green.svg)](https://pyscf.org)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/brycewestheimer/ormas-ci/graph/badge.svg)](https://codecov.io/gh/brycewestheimer/ormas-ci)

## Overview

ORMAS-CI partitions the active orbital space into subspaces with per-subspace electron occupation bounds, then builds and diagonalizes the CI Hamiltonian over only the allowed determinants. It plugs into PySCF as a custom `fcisolver` for CASCI and CASSCF, so existing PySCF workflows need only one extra line to switch from full CI to a restricted expansion.

The package supports both the classic three-space RASCI partitioning and the general ORMAS-CI scheme with an arbitrary number of subspaces. A tested bridge to Microsoft's QDK/Chemistry library lets QDK-driven SCF feed into ORMAS-CI through PySCF's CASCI interface for quantum resource estimation.

## Installation

```bash
pip install pyscf-ormas-ci
```

From source:

```bash
git clone https://github.com/brycewestheimer/ormas-ci.git
cd ormas-ci
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"        # pytest, ruff, build
pip install -e ".[notebooks]"  # jupyter, matplotlib
pip install -e ".[qdk]"        # qdk-chemistry
```

Requires Python >= 3.11, NumPy >= 1.24, SciPy >= 1.10, PySCF >= 2.4.

## Quick Start

Run an ORMAS-CI calculation on water, partitioning the active space into sigma and pi subspaces, then compare against full CASCI:

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace
from pyscf.ormas_ci.determinants import count_determinants, casci_determinant_count

mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='cc-pvtz',
)
mf = scf.RHF(mol).run()

ncas = 5
nelecas = (3, 3)

config = ORMASConfig(
    subspaces=[
        Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=5),
        Subspace("pi", [3, 4], min_electrons=1, max_electrons=4),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

# One line to swap the solver
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
e_ormas, ci = mc.kernel()

# Full CASCI reference
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
e_casci = mc_ref.kernel()[0]

print(f"CASCI energy:    {e_casci:.10f} Hartree")
print(f"ORMAS-CI energy: {e_ormas:.10f} Hartree")
print(f"Difference:      {(e_ormas - e_casci) * 1000:.4f} mHartree")

n_ormas = count_determinants(config)
n_casci = casci_determinant_count(ncas, nelecas)
print(f"Reduction to {n_ormas / n_casci:.1%} of full CASCI space")
```

The ORMAS-CI energy will be at or slightly above the CASCI energy. If the gap is larger than a few mHartree, relax the occupation constraints.

## How It Works

For small determinant spaces (n_det <= 200), the solver builds the CI Hamiltonian explicitly and diagonalizes with dense `numpy.linalg.eigh`. For larger spaces, it uses a pure-Python Davidson-Liu eigensolver with an einsum-based sigma vector that precomputes dense excitation matrices and ERI intermediates for fast H @ c computation via NumPy. For very large spaces (> 300 unique strings per spin channel), it falls back to PySCF's `pyscf.fci.selected_ci` C-level sigma with ARPACK Lanczos. RDMs, spin diagnostics, and the diagonal preconditioner delegate to PySCF's C-level `selected_ci` routines across all paths. See [`docs/design/pyscf_differences.md`](docs/design/pyscf_differences.md) for details on numerical accuracy.

## Documentation

Full docs are in the [`docs/`](docs/) directory, covering [theory](docs/theory/), [architecture](docs/architecture/), [PySCF integration](docs/design/pyscf_integration.md), [QDK integration](docs/design/qdk_integration.md), [API reference](docs/api/public_api.md), and [usage guides](docs/guides/).

## License

MIT
