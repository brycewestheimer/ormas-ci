# Differences from Native PySCF CASCI/CASSCF

This document describes how ORMAS-CI differs from PySCF's native FCI
solver (`pyscf.fci.direct_spin1`) in numerical behavior, solver
implementation, and feature support.

## Numerical Accuracy

### Energy

When run as a full CAS (single ORMAS subspace covering all orbitals
with no restriction), ORMAS-CI reproduces PySCF CASCI energies. The
level of agreement depends on the system size:

| Regime | Agreement | Mechanism |
|--------|-----------|-----------|
| n_det <= 200 | Exact (0 Ha) | Dense `numpy.linalg.eigh` on explicit Hamiltonian |
| n_det > 200 | ~1e-10 Ha | Iterative `scipy.sparse.linalg.eigsh` via PySCF's C-level sigma |

For the iterative path, the convergence tolerance is controlled by
`ORMASFCISolver.conv_tol` (default 1e-10).

### CI Vector

For small systems, the CI eigenvectors match PySCF to machine
precision (~1e-15). For larger systems solved iteratively, the CI
vectors agree to the eigsh convergence tolerance.

Notably, PySCF's own Davidson solver also has a finite convergence
residual. At CAS(8,8) with 4,900 determinants, PySCF's CI vector
has a Hamiltonian residual norm of ~3e-5, while our dense
diagonalization path achieves ~3e-14. This means that for the
explicit Hamiltonian path, ORMAS-CI can be **more precise** than
PySCF's iterative solver.

### Reduced Density Matrices

RDM accuracy directly reflects CI vector accuracy. For small systems,
RDMs match PySCF to ~1e-15. For larger systems, the observed ~1e-6
RDM differences arise from CI vector differences between solvers, not
from the RDM construction itself. Our RDM routines (which delegate to
PySCF's `selected_ci.make_rdm1/make_rdm2` C code) produce results
identical to PySCF's RDM code given the same CI vector (verified to
~1e-14).

### Spin

`spin_square()` delegates to `pyscf.fci.selected_ci.spin_square` and
matches PySCF to the same precision as the CI vector.

## Solver Implementation

### PySCF Native FCI

PySCF's `direct_spin1` uses a Davidson iterative eigensolver with a
C-level sigma vector (H * c) that operates directly on the full CAS
string space. It never builds the Hamiltonian matrix explicitly. The
CI vector is stored as a 2D array indexed by (alpha_string_index,
beta_string_index).

### ORMAS-CI

ORMAS-CI uses two solver paths, selected automatically by determinant
count:

**Small spaces (n_det <= 200):** Builds the explicit Hamiltonian matrix
via Slater-Condon rules and diagonalizes with `numpy.linalg.eigh`. This
gives exact eigenvalues/eigenvectors but scales as O(n_det^2) for
construction and O(n_det^3) for diagonalization.

**Large spaces (n_det > 200):** Uses PySCF's
`pyscf.fci.selected_ci.contract_2e` C routine for matrix-free sigma
vector computation, wrapped in `scipy.sparse.linalg.eigsh` (ARPACK
Lanczos). The CI vector is internally mapped between a 1D ORMAS
representation and PySCF's 2D `selected_ci` format on each sigma call.
(Note: we use PySCF's `selected_ci` module as a computational backend
for arbitrary determinant string sets, not as a Selected CI method.)

The threshold is controlled by `ORMASFCISolver.direct_ci_threshold`
(default 200).

### Performance Comparison

At CAS(8,8) with 4,900 determinants, ORMAS-CI is approximately 15x
slower than PySCF's native FCI. The overhead comes from the
Python-level 1D-to-2D CI vector marshalling on each sigma call. For
smaller systems (n_det <= 400), ORMAS-CI is at parity or faster due
to PySCF CASCI framework overhead dominating both solvers.

The real value of ORMAS-CI is the determinant space reduction: an
ORMAS-restricted CAS(8,8) might have 1,000 determinants instead of
4,900, making the restricted solve faster than PySCF's full CAS.

## CI Vector Representation

PySCF stores the CI vector as a 2D numpy array of shape
`(n_alpha_strings, n_beta_strings)`, where the string indices follow
`pyscf.fci.cistring` ordering.

ORMAS-CI stores the CI vector as a 1D numpy array of length `n_det`,
with paired `alpha_strings[i]` / `beta_strings[i]` arrays defining
which determinant each coefficient corresponds to. This representation
naturally handles the ORMAS restriction: only allowed determinants
appear in the arrays.

Conversion between the two formats is handled internally by
`_ci_1d_to_2d()` and `_ci_2d_to_1d()`.

## Multi-Root Behavior

Both PySCF and ORMAS-CI support multi-root calculations via
`fcisolver.nroots`. Differences:

- PySCF uses Davidson with deflation, which handles near-degenerate
  roots robustly.
- ORMAS-CI uses `eigsh(k=nroots)`, which can occasionally miss or
  swap near-degenerate roots if the convergence tolerance is too loose.
  For systems with near-degeneracies, set `conv_tol=1e-12` or tighter.

## Feature Support

### Fully Supported

- `kernel()` -- energy and CI vector
- `make_rdm1()`, `make_rdm1s()` -- 1-particle density matrices
- `make_rdm12()`, `make_rdm12s()` -- 1- and 2-particle density matrices
- `spin_square()` -- S^2 expectation value and multiplicity
- `contract_2e()` -- H * c product (for CASSCF orbital optimization)
- `contract_1e()` -- 1-electron H * c product
- `absorb_h1e()` -- integral preprocessing for CASSCF
- `make_hdiag()` -- diagonal preconditioner
- `pspace()` -- P-space Hamiltonian subblock
- `get_init_guess()` -- initial CI guess vectors
- `large_ci()` -- significant CI coefficients
- `transform_ci_for_orbital_rotation()` -- CI transformation under U
- CASSCF orbital optimization via `mcscf.CASSCF`
- `pyscf.fci.addons.fix_spin_()` spin penalty

### Not Supported

- **State-averaged CASCI/CASSCF** -- requires state-averaged RDM
  construction, which is not yet implemented.
- **Analytical nuclear gradients** -- requires response equation terms
  that depend on the solver internals.
- **Point group symmetry** -- PySCF's symmetry-adapted FCI
  (`direct_spin0_symm`) restricts determinants by spatial symmetry.
  ORMAS-CI does not use spatial symmetry, though subspace constraints
  can achieve a similar effect for cases where the symmetry aligns
  with the orbital partitioning.
- **Frozen core within the active space** -- PySCF supports
  `frozen` parameter on some FCI solvers; ORMAS-CI does not.

## When to Use Which

Use **PySCF native FCI** when:
- You want the fastest possible full CAS calculation
- You need state-averaged CASSCF
- You need analytical gradients

Use **ORMAS-CI** when:
- You want to reduce the determinant space via occupation restrictions
- You need quantum resource estimates for a restricted CI space
- You want RASCI or ORMAS partitioning within PySCF's CASCI/CASSCF
