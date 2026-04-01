# Module Overview

## Package Structure

```
pyscf/ormas_ci/
    __init__.py          Public API exports
    utils.py             Bit manipulation primitives
    subspaces.py         Subspace data model and validation
    determinants.py      Restricted determinant enumeration
    slater_condon.py     Hamiltonian matrix element evaluation
    hamiltonian.py       CI matrix construction
    solver.py            Eigenvalue solver wrapper (dense/sparse)
    davidson.py          Davidson-Liu iterative eigensolver
    sigma.py             Einsum-based sigma vector engine
    rdm.py               Reduced density matrix construction
    spinflip.py          Spin-flip determinant generation and validation
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
davidson.py         <- no internal dependencies (numpy only)
sigma.py            <- utils
rdm.py              <- slater_condon, utils
spinflip.py         <- utils, subspaces, determinants
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
Used for the explicit solver path when n_det <= direct_ci_threshold (200).

**solver.py:** Thin wrapper around numpy.linalg.eigh (dense) and
scipy.sparse.linalg.eigsh (sparse Lanczos). Used by the explicit
Hamiltonian path and as fallback for very large iterative spaces.

**davidson.py:** Pure-Python/NumPy Davidson-Liu iterative eigensolver
with subspace expansion, restart, multi-root support, and diagonal
preconditioning. Used for the medium-space iterative path.

**sigma.py:** Pure-Python einsum-based sigma vector engine. Precomputes
dense single-excitation operator matrices and ERI-contracted
intermediates once per solve, then provides fast H @ c via NumPy einsum
contractions. Used with the Davidson solver for spaces with <= 300
unique strings per spin channel.

**rdm.py:** Constructs reduced density matrices from CI eigenvectors.
Implements both one-particle (1-RDM) and two-particle (2-RDM) density
matrices (make_rdm1, make_rdm12). Depends on slater_condon for
excitation analysis and utils for bit manipulation.

**spinflip.py:** Spin-flip determinant generation and validation.
Generates determinant lists for SF-ORMAS calculations in the target
M_s sector from a high-spin reference. Provides
`generate_sf_determinants()`, `count_sf_determinants()`, and
`validate_reference_consistency()`.

**fcisolver.py:** The user-facing integration layer. Implements the
PySCF fcisolver interface (kernel, make_rdm1, make_rdm12, spin_square)
by composing all other modules. Handles PySCF-specific format
conversions (compressed ERIs, nelecas normalization) and selects
among three solver paths based on the determinant space size:
explicit Hamiltonian + dense eigh, Davidson + SigmaEinsum, or
PySCF selected_ci + ARPACK. This is the only module that external
code interacts with directly.
