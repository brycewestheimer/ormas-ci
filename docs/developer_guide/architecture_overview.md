# Architecture Overview for Contributors

This is a narrative walkthrough of the ORMAS-CI codebase for new
contributors. For the formal module dependency graph and detailed
algorithm descriptions, see the [Architecture](../architecture/overview.md)
section.

## Where to Start Reading

Start with these three files in order:

1. **`fcisolver.py`** -- The entry point. This is where PySCF calls
   into ORMAS-CI. Read `ORMASFCISolver.kernel()` to see how the
   solver orchestrates the entire calculation pipeline.

2. **`subspaces.py`** -- The data model. Read `Subspace`,
   `ORMASConfig`, and `RASConfig` to understand how users configure
   the calculation.

3. **`determinants.py`** -- The core algorithm. Read
   `build_determinant_list()` to see how allowed determinants are
   enumerated from the ORMAS configuration.

## The Three Solver Paths

The key architectural decision in `fcisolver.py` is the solver path
selection in `kernel()`:

**Path 1: Dense (n_det <= 200)**
- `hamiltonian.py` builds the explicit H matrix via Slater-Condon rules
- `solver.py` diagonalizes with `numpy.linalg.eigh`
- `slater_condon.py` computes individual matrix elements

**Path 2: Davidson + Einsum (200 < n_det, <= 300 strings/channel)**
- `sigma.py` precomputes dense excitation operator matrices
- `davidson.py` runs the Davidson-Liu iterative eigensolver
- Each Davidson iteration calls the sigma engine for H * c

**Path 3: PySCF SCI Fallback (> 300 strings/channel)**
- CI vector is mapped between ORMAS 1D format and PySCF's 2D
  `selected_ci` format via `_build_sci_mapping()`
- PySCF's C-level `selected_ci.contract_2e()` computes the sigma vector
- `scipy.sparse.linalg.eigsh` (ARPACK Lanczos) finds eigenvalues

## Key Data Structures

**Determinants** are pairs of integers (alpha_string, beta_string).
Bit k set means orbital k is occupied. See
[Bit String Representation](../architecture/bitstrings.md).

**CI vector** is a 1D numpy array of length n_det. Element i is the
coefficient of the i-th determinant. The mapping between coefficients
and determinants is maintained by parallel `alpha_strings[i]` and
`beta_strings[i]` arrays.

**Configuration** is a tree of dataclasses: `ORMASConfig` contains a
list of `Subspace` objects. No inheritance, no abstract interfaces.

## Module Dependency Order

Modules depend only on modules listed above them:

```
utils.py            <- standalone bit manipulation
subspaces.py        <- standalone data model
determinants.py     <- uses utils, subspaces
slater_condon.py    <- uses utils
hamiltonian.py      <- uses slater_condon
solver.py           <- standalone (numpy/scipy only)
davidson.py         <- standalone (numpy only)
sigma.py            <- uses utils
rdm.py              <- uses slater_condon, utils
spinflip.py         <- uses utils, subspaces, determinants
fcisolver.py        <- composes all of the above
```

Only `fcisolver.py` touches all other modules. Each module can be
understood and tested independently. See
[Module Overview](../architecture/overview.md) for detailed
responsibilities.

## How PySCF Calls ORMAS-CI

PySCF's CASCI does the following:

1. Transforms integrals from the AO to the active MO basis
2. Calls `fcisolver.kernel(h1e, eri, norb, nelec)` -- this is where
   ORMAS-CI takes over
3. Uses the returned CI vector and energy
4. Calls `make_rdm1()` / `make_rdm12()` for density matrices
5. For CASSCF: calls `contract_2e()` repeatedly during orbital
   optimization

The full interface contract is documented in
[PySCF Integration](../design/pyscf_integration.md).

## The Pipeline in Detail

For a step-by-step trace of data flowing through the system (from
molecule to energy), see
[Data Flow](../architecture/data_flow.md).
