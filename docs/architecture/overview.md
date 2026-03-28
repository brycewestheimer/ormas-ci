# Module Overview

## Package Structure

```
src/ormas_ci/
    __init__.py          Public API exports
    utils.py             Bit manipulation primitives
    subspaces.py         Subspace data model and validation
    determinants.py      Restricted determinant enumeration
    slater_condon.py     Hamiltonian matrix element evaluation
    hamiltonian.py       CI matrix construction
    solver.py            Eigenvalue solver wrapper
    rdm.py               Reduced density matrix construction
    fcisolver.py         PySCF integration layer
```

## Dependency Graph

Modules depend only on modules above them in this list:

```
utils.py            <- no internal dependencies
subspaces.py        <- no internal dependencies
determinants.py     <- utils, subspaces
slater_condon.py    <- utils
hamiltonian.py      <- slater_condon
solver.py           <- no internal dependencies (numpy/scipy only)
rdm.py              <- utils
fcisolver.py        <- all of the above
```

This ordering allows each module to be implemented and tested independently
before integration. The only module that touches all others is `fcisolver.py`,
which serves as the composition layer.

## Module Responsibilities

**utils.py:** Low-level bit manipulation for occupation string representation.
Functions like popcount, bit-to-index conversion, string generation, and
fermionic phase computation. These are pure functions with no state and no
dependencies on other project modules.

**subspaces.py:** Data model only. Defines Subspace, ORMASConfig, and
RASConfig as dataclasses with validation logic. No computation beyond
constraint checking. No dependencies on other project modules.

**determinants.py:** The core algorithmic module. Enumerates all Slater
determinants satisfying ORMAS occupation constraints. The algorithm is
a three-stage process: enumerate valid per-subspace occupation
distributions, generate per-subspace occupation strings, and combine
across subspaces. See [determinant_enumeration.md](determinant_enumeration.md).

**slater_condon.py:** Computes Hamiltonian matrix elements between pairs
of Slater determinants using the Slater-Condon rules. Handles diagonal,
single excitation, and double excitation cases with proper fermionic
phase factors. See [phase_factors.md](phase_factors.md).

**hamiltonian.py:** Builds the explicit CI Hamiltonian matrix by looping
over determinant pairs and calling slater_condon.matrix_element(). Uses
vectorized excitation precomputation to skip pairs with >2 excitations.
Used as the fallback solver path for small determinant spaces (n_det <=
200); larger spaces use the matrix-free path in fcisolver.py via
PySCF's `pyscf.fci.selected_ci` C-level routines.

**solver.py:** Thin wrapper around numpy.linalg.eigh (dense) and
scipy.sparse.linalg.eigsh (sparse Lanczos). Used by the explicit
Hamiltonian path. The matrix-free path uses eigsh with a
LinearOperator wrapping PySCF's C-level sigma vector instead.

**rdm.py:** Constructs reduced density matrices from CI eigenvectors.
Implements both one-particle (1-RDM) and two-particle (2-RDM) density
matrices (make_rdm1, make_rdm12).

**fcisolver.py:** The user-facing integration layer. Implements the
PySCF fcisolver interface (kernel, make_rdm1, make_rdm12, spin_square)
by composing all other modules. Handles PySCF-specific format
conversions (compressed ERIs, nelecas normalization). This is the only
module that external code interacts with directly.
