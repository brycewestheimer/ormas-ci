# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-28

### Changed
- **Namespace package migration:** Renamed from `ormas-ci` / `ormas_ci` to
  `pyscf-ormas-ci` / `pyscf.ormas_ci`. The package now lives under the
  `pyscf` namespace package (`pyscf/ormas_ci/`) instead of `src/ormas_ci/`.
  All imports change from `from ormas_ci import ...` to
  `from pyscf.ormas_ci import ...`.
- Installation command changed from `pip install ormas-ci` to
  `pip install pyscf-ormas-ci`
- A backward-compatibility shim at `ormas_ci/__init__.py` re-exports the
  public API so that existing `from ormas_ci import ...` code continues to
  work with a deprecation warning

## [0.2.0] - 2026-03-27

### Added
- Matrix-free solver path using PySCF's `pyscf.fci.selected_ci` C-level
  routines for sigma vectors, RDMs, `make_hdiag`, and `spin_square`
  (note: uses the `selected_ci` module as a computational backend for
  arbitrary determinant sets, not as a Selected CI method)
- Iterative eigensolver (`scipy.sparse.linalg.eigsh`) with
  `LinearOperator` for large determinant spaces (n_det > 200)
- 1D/2D CI vector mapping layer for PySCF `selected_ci` interoperability
- Vectorized excitation precomputation in Hamiltonian construction via
  16-bit popcount lookup table, skipping >50% of determinant pairs
- Vectorized `make_hdiag` with precomputed Coulomb/exchange arrays
- Integration test suite (`tests/test_sci_integration.py`) comparing
  explicit Hamiltonian and matrix-free paths
- `docs/design/pyscf_differences.md` documenting numerical accuracy and
  behavioral differences from PySCF native FCI
- Performance benchmarks comparing ORMAS-CI against PySCF native FCI
- Dynamic code coverage reporting via CI
- Pre-commit hooks for automated linting and formatting
- Type checking with pyright in CI
- CONTRIBUTORS.md and CONTRIBUTING.md

### Changed
- `popcount()` uses `int.bit_count()` instead of `bin(n).count("1")`
- `bits_to_indices()` cached with `@functools.cache`; return type
  changed from `list[int]` to `tuple[int, ...]`
- `kernel()` automatically selects matrix-free path for n_det > 200
  and explicit Hamiltonian for smaller spaces
- `contract_2e()`, `make_rdm*()`, `spin_square()`, and `make_hdiag()`
  delegate to PySCF's C-level `selected_ci` routines when available
- Default `conv_tol` changed from 1e-10 to 1e-12 for better handling
  of near-degenerate multi-root cases
- Updated `docs/design/performance.md` with current benchmark numbers
- Updated LICENSE copyright year to 2025-2026
- Added `__all__` exports to all submodules

### Fixed
- `nroots >= n_det` no longer crashes with IndexError; falls back to
  full diagonalization and returns all available roots

## [0.1.0] - 2025-12-01

### Added
- ORMAS-CI solver implementing occupation-restricted multiple active space CI
- PySCF CASCI/CASSCF integration as a drop-in `fcisolver` plugin
- RASCI convenience class (`RASConfig`) for classic 3-space partitioning
- General ORMAS-CI with arbitrary subspace count and per-subspace occupation bounds
- Determinant enumeration with subspace constraint pruning
- Hamiltonian construction via Slater-Condon rules (dense and sparse)
- 1-RDM and 2-RDM construction matching PySCF index conventions
- Spin-square expectation value (`spin_square`) for state characterization
- CI vector transformation under orbital rotation
- Bridge to Microsoft QDK/Chemistry for quantum resource estimation
- Comprehensive test suite (144 tests, 95%+ coverage)
- MkDocs documentation with theory, architecture, and integration guides
- Jupyter notebook tutorials
