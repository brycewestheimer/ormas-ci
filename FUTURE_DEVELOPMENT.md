# Future Development Roadmap

Comprehensive plan for evolving ORMAS-CI from proof-of-concept to
high-performance production solver, with end-to-end quantum computing
integration.

## Current State (v0.2.0)

- ORMAS-CI solver with three solver paths selected automatically:
  1. **Small spaces (n_det <= 200):** Explicit Hamiltonian + dense
     `numpy.linalg.eigh`
  2. **Medium spaces (n_det > 200, <= ~300 unique strings/channel):**
     Pure-Python Davidson eigensolver with einsum-based sigma vector
     using precomputed dense excitation matrices (new in v0.2.0)
  3. **Large spaces (> 300 unique strings/channel):** PySCF's
     `selected_ci` C-level sigma + ARPACK `eigsh` fallback
- PySCF's `pyscf.fci.selected_ci` C-level routines used for RDMs,
  `make_hdiag`, and `spin_square` across all paths
- Pure-Python explicit Hamiltonian path retained as fallback for
  small determinant spaces (n_det <= 200)
- 175 tests, pyright-clean, ruff-clean
- PySCF CASCI/CASSCF integration as drop-in `fcisolver` plugin
- QDK/Chemistry integration through PySCF's CASCI interface
- Performance: approaching parity with PySCF native FCI at CAS(8,8)
  for the Davidson+einsum path (improved from ~85x via SCI
  integration, vectorized popcount/caching, excitation precomputation,
  and the Davidson+einsum sigma refactor)
- Phase 1 Python optimizations: `int.bit_count()`, `functools.cache`
  on `bits_to_indices`, vectorized `make_hdiag`, excitation
  classification precomputation
- Phase 2 `pyscf.fci.selected_ci` backend integration: matrix-free
  sigma via `contract_2e`, C-level RDMs, iterative `eigsh` solver,
  C-level `make_hdiag` and `spin_square`
- Phase 2.3 Davidson eigensolver + einsum sigma: pure-Python
  Davidson-Liu solver with dense excitation matrix precomputation
  and NumPy einsum contractions, eliminating PySCF's per-sigma-call
  ERI preprocessing overhead
- Initial quantum benchmarks: JW qubit Hamiltonian construction,
  classical qubit solve, Trotter decomposition, subspace factorization
  (see `benchmarks/bench_qdk_quantum.py`)

### Known Approximations in Current Quantum Benchmarks

Most approximations from the initial implementation have been resolved.
Remaining open items:

- **Excitation operator proxy** (`bench_vs_pyscf.py`): Off-diagonal
  Hamiltonian matrix element count is labeled as a "classical proxy"
  for quantum operator count. Real Pauli term counts are available
  in `bench_qdk_quantum.py` via QDK's JW mapping.
- **No per-subspace Pauli term counts**: Subspace factorization reports
  qubit savings but does not construct per-subspace qubit Hamiltonians.

Resolved items:
- Gate counts: now from recursive QASM3 circuit parsing, not formulas
- QPE: real IQPE simulation via `qdk_full_state_simulator`
- VQE: real shot-based energy via `qdk_base_simulator`
- State prep: method-specific circuits via `SparseIsometryGF2XStatePreparation`
- Model systems: N2 (sigma/pi), FeO (metal/ligand), butadiene (spatial)
- Notebook: uses real QDK-computed values throughout
- Open-shell: uses `ModelOrbitals` bridge (not proxy orbitals)

---

## Phase 0: Replace Approximations with Real Quantum Metrics — Complete

**Goal:** Replace all proxy metrics and analytical estimates with
genuine QDK-computed values, and benchmark on chemically motivated
model systems where RAS/ORMAS partitioning is natural.

**Status:** All 8 tasks implemented. Benchmarks use real QDK circuit
construction, real IQPE/VQE simulation, chemically motivated model
systems (N2, FeO, butadiene, ethylene, ozone, formaldehyde), and
method-specific state preparation circuits.

### Task 0.1: Real Circuit Construction from QDK — Implemented

Recursive QASM3 gate parser with custom-gate expansion in
`benchmarks/bench_qdk_quantum.py:parse_gate_counts()`.

### Task 0.2: Real IQPE Simulation — Implemented

`_run_iqpe()` runs real `IterativePhaseEstimation` with
`qdk_full_state_simulator` at 6-bit and 10-bit precision.

### Task 0.3: Real VQE-Style Energy Estimation — Implemented

`_run_energy_estimator()` runs `qdk_base_simulator` with
`group_commuting()` Pauli grouping at 1K and 10K shots.

### Task 0.4: Replace Proxy Metrics with Real Pauli Term Counts — Implemented

All benchmark systems use `len(qubit_ham.pauli_strings)` from QDK JW
mapping. Classical proxy in `bench_vs_pyscf.py` is labeled as such.

### Task 0.5: Chemically Motivated Model Systems — Implemented

**RAS systems**: Ethylene (cc-pVDZ, cc-pVTZ), Ozone (cc-pVDZ),
Formaldehyde (cc-pVDZ).

**ORMAS systems**: N2 sigma/pi (2.3x, 2.4 mHa), FeO metal/ligand
(3.5x, 0.4 mHa), Butadiene spatial (2.0x).

### Task 0.6: Comprehensive Comparison Pipeline — Implemented

`run_system_benchmark()` runs CASCI/RASCI/ORMAS with method-specific
state prep, real IQPE, real VQE, real gate counts per system.

### Task 0.7: Update Documentation and Notebook — Implemented

All docs, README, performance tables, and notebooks updated to reflect
current benchmark systems and real QDK metrics.

### Task 0.8: Examples Directory — Implemented

Six runnable example scripts in `examples/`:

**PySCF-only examples:**
- `examples/01_basic_ormas.py` -- H2O CAS(6,6) with 3-subspace partitioning
- `examples/02_rasci.py` -- Formaldehyde n/pi/pi* RAS partitioning
- `examples/03_casscf_optimization.py` -- ORMAS-CI in PySCF CASSCF
- `examples/04_convergence_study.py` -- N2 sigma/pi constraint relaxation

**QDK pipeline examples:**
- `examples/05_qdk_pipeline.py` -- QDK SCF -> ORMAS-CI -> qubit Hamiltonian
- `examples/06_qdk_quantum_simulation.py` -- Real IQPE + VQE with
  method-specific state prep

---

## Phase 1: Python-Level Performance (5-20x Speedup) — Implemented

**Goal:** Maximize performance within pure Python using vectorization,
caching, and modern Python features. No compiled extensions.
Target: CAS(8,8) from 15s to 1-3s.

### Task 1.1: Replace `popcount()` with `int.bit_count()` — Implemented

Replaced `bin(n).count("1")` with `n.bit_count()` in
`src/ormas_ci/utils.py:popcount()`.

### Task 1.2: Cache `bits_to_indices()` Results — Implemented

Added `@functools.cache` to `bits_to_indices()` in
`src/ormas_ci/utils.py`. Return type changed to `tuple[int, ...]`
for hashability. Cache hit rate approaches 100% for large CI spaces.

### Task 1.3: Vectorize Diagonal Hamiltonian Elements — Implemented

Added `_make_hdiag_vectorized()` in `fcisolver.py` that precomputes
`jdiag[p,q] = h2e[p,p,q,q]` and `kdiag[p,q] = h2e[p,q,q,p]` once,
then uses numpy fancy indexing for each determinant.

### Task 1.4: Precompute Excitation Classification — Implemented

Added vectorized popcount via 16-bit lookup table in
`hamiltonian.py:_vectorized_popcount()`. Precomputes which (i,j)
pairs have ≤2 excitations before the main loop, skipping >50% of
pairs that are guaranteed zero by Slater-Condon rules.

### Task 1.5: Vectorize Single-Excitation Matrix Elements

The diagonal elements `<det_i|H|det_i>` in `_compute_diagonal()`
loop over occupied orbital pairs with individual H1E/H2E lookups.
These can be vectorized using numpy array indexing and `einsum`.

**Work:**
- In `slater_condon.py:_compute_diagonal()`, replace:
  ```python
  for p in occ_a:
      energy += h1e[p, p]
  for p, q in pairs:
      energy += h2e[p,p,q,q] - h2e[p,q,q,p]
  ```
  with:
  ```python
  occ_a = np.array(bits_to_indices(alpha))
  energy = np.sum(h1e[occ_a, occ_a])
  energy += np.sum(h2e[occ_a[:,None], occ_a[:,None], occ_a[None,:], occ_a[None,:]])
  energy -= np.sum(h2e[occ_a[:,None], occ_a[None,:], occ_a[None,:], occ_a[:,None]])
  # ... plus alpha-beta cross terms
  ```
- Add a batch version `compute_all_diagonals()` that processes
  all n_det diagonal elements at once using advanced indexing
- Wire the batch function into `hamiltonian.py` to compute all
  diagonals before the main loop

**Expected speedup:** 5-10x on diagonal computation (typically 10-15%
of total Hamiltonian construction time).

**Note:** PySCF provides `pyscf.fci.direct_spin1.make_hdiag()` which
computes diagonal Hamiltonian elements using optimized C code. However,
it operates on the full CAS determinant space, not the ORMAS-restricted
space. A custom implementation for the restricted space is needed, but
the PySCF function can serve as a reference for correctness.

**Files:** `src/ormas_ci/slater_condon.py`, `src/ormas_ci/hamiltonian.py`

### Task 1.4: Precompute Excitation Classification

The Hamiltonian double loop calls `popcount(alpha_i ^ alpha_j)` and
`popcount(beta_i ^ beta_j)` for every pair to classify the excitation
level. Most pairs (>50% for typical CI) have >2 excitations and are
immediately skipped. This classification can be vectorized.

**Work:**
- Before the main Hamiltonian loop, precompute excitation levels:
  ```python
  alpha_xor = alpha_strings[:, None] ^ alpha_strings[None, :]
  beta_xor = beta_strings[:, None] ^ beta_strings[None, :]
  # Use vectorized bit_count if available (numpy >=2.0),
  # otherwise use a lookup table for 16-bit chunks
  n_diff_a = vectorized_popcount(alpha_xor) // 2
  n_diff_b = vectorized_popcount(beta_xor) // 2
  n_excitations = n_diff_a + n_diff_b
  ```
- Build index arrays for each excitation class:
  ```python
  diag_pairs = np.where(n_excitations == 0)
  single_pairs = np.where(n_excitations == 1)
  double_pairs = np.where(n_excitations == 2)
  ```
- Process each class in a batch loop instead of the full O(n_det^2)
  iteration. Pairs with >2 excitations (zero matrix element) are
  never visited.

**Memory cost:** O(n_det^2) for the classification arrays. For
n_det=5000, this is ~200MB (uint8). For larger spaces, use a
block-based approach that processes chunks of 1000x1000 pairs.

**Expected speedup:** 3-5x from skipping >50% of pairs and removing
per-pair Python function call overhead.

**Files:** `src/ormas_ci/hamiltonian.py`

### Task 1.5: Vectorize Single-Excitation Matrix Elements

Single excitations are the most common non-zero off-diagonal elements
and involve an O(n_elec) loop over spectator orbitals per pair. These
spectator contributions can be partially vectorized.

**Work:**
- For all single-alpha excitation pairs (from Task 1.4's index
  arrays), extract the hole and particle orbitals in batch using
  vectorized XOR/popcount
- Group single excitations by (hole, particle) pair — many
  determinant pairs share the same hole/particle but differ in
  spectator electrons
- For each (hole, particle) group, vectorize the spectator
  contribution using numpy array indexing into H2E
- Apply the same approach for single-beta excitations

**Expected speedup:** 2-5x on single-excitation evaluation
(typically 30-40% of total Hamiltonian construction time).

**Files:** `src/ormas_ci/slater_condon.py`, `src/ormas_ci/hamiltonian.py`

### Task 1.6: Optimize RDM Construction

The RDM double loop in `_make_rdm12s_impl()` iterates over all
n_det^2 pairs (not just upper triangle) with per-pair coefficient
threshold checks and spectator orbital enumeration.

**Work:**
- Apply the same excitation classification precomputation from
  Task 1.4 to the RDM loop
- Vectorize the diagonal RDM contributions: for all determinants
  with significant coefficients, compute the 1-RDM diagonal
  in batch using `np.add.at()` or direct indexing
- For single-excitation RDM contributions, group by (hole, particle)
  and accumulate spectator contributions in batch
- Use the coefficient sparsity: pre-filter determinant pairs where
  `|c_i * c_j| < threshold` using numpy broadcasting before entering
  the excitation-specific loops

**Expected speedup:** 3-5x on RDM construction. The RDM is typically
as expensive as Hamiltonian construction for CASSCF workflows.

**Files:** `src/ormas_ci/rdm.py`

---

## Phase 2: PySCF `selected_ci` Backend Integration (20-100x Speedup) — Implemented

**Goal:** Replace the explicit Hamiltonian construction with PySCF's
C-level routines for matrix-free sigma vector computation. This is the
single largest performance gain available, because it replaces the
O(n_det^2) Python Hamiltonian construction with O(n_det * n_links)
C-level operations.

**Status:** All tasks implemented including Task 2.3 (Davidson +
einsum sigma). The solver uses a pure-Python Davidson eigensolver
with einsum-based sigma vector for medium spaces, falling back to
PySCF's `selected_ci` + ARPACK eigsh for very large spaces. RDMs,
`make_hdiag`, and `spin_square` delegate to PySCF's C-level routines
via a 1D↔2D CI vector mapping layer.

**Key discovery:** PySCF's `pyscf.fci.selected_ci` module provides
C-level routines for computing the sigma vector (H * c) over
*arbitrary* determinant sets — not just the full CAS space. Despite
the module name, this is not a Selected CI method (CIPSI, ASCI); it
is a generalized computational backend that accepts custom determinant
string lists. This is exactly what ORMAS-CI needs: we pass our
ORMAS-restricted strings and get efficient C-level sigma vectors,
RDMs, and diagnostics in return.

### Task 2.1: Generate Link Tables for ORMAS Determinant Space

PySCF's selected CI uses "link tables" (excitation path tables) that
encode which determinant pairs are connected by single excitations,
along with the target determinant index and phase factor. These tables
enable O(n_det * n_connections) sigma vector computation instead of
O(n_det^2) explicit Hamiltonian construction.

**Work:**
- After `build_determinant_list()` produces the ORMAS alpha/beta
  strings, call `pyscf.fci.selected_ci.cre_des_linkstr()` to
  generate the link tables:
  ```python
  from pyscf.fci import selected_ci
  link_a = selected_ci.cre_des_linkstr(alpha_strings, norb, nelec_a)
  link_b = selected_ci.cre_des_linkstr(beta_strings, norb, nelec_b)
  ```
- The returned link table has shape `(n_strings, n_links, 4)` where
  each entry is `[create_orbital, destroy_orbital, target_address, parity]`
- Cache the link tables on the `ORMASFCISolver` object alongside the
  determinant strings — they are reused for every sigma vector
  computation, RDM construction, and CASSCF orbital optimization step

**Note:** `cre_des_linkstr` calls into PySCF's C library
(`SCIcre_des_linkstr`) for performance. The ORMAS determinant strings
must be passed as a numpy int64 array, which `build_determinant_list()`
already produces.

**Files:** `src/ormas_ci/fcisolver.py`, `src/ormas_ci/determinants.py`

### Task 2.2: Implement Matrix-Free Sigma Vector via `selected_ci.contract_2e()`

Replace the explicit Hamiltonian matrix with PySCF's C-level sigma
vector computation for arbitrary determinant sets.

**Work:**
- Replace the Hamiltonian construction + matrix multiply in `kernel()`
  with a call to `pyscf.fci.selected_ci.contract_2e()`:
  ```python
  from pyscf.fci import selected_ci

  def _sigma(c):
      """Compute H @ c without building H explicitly."""
      return selected_ci.contract_2e(
          eri, c, norb, nelec, link_index=(link_a, link_b)
      )
  ```
- This C function (`SCIcontract_2e_aaaa`, `SCIcontract_2e_bbaa`)
  computes the sigma vector by iterating over excitation links,
  not over all determinant pairs. Complexity drops from O(n_det^2)
  to O(n_det * n_links) where n_links ~ n_orbitals^2.
- The 1-electron contribution can be similarly handled via
  `selected_ci.contract_1e()` or by absorbing h1e into eri
  using `pyscf.fci.direct_spin1.absorb_h1e()`

**Memory savings:** From O(n_det^2) for the explicit Hamiltonian
matrix to O(n_det * n_links) for the link tables. For CAS(8,8)
with 4900 determinants: from ~192 MB to ~5 MB.

**Files:** `src/ormas_ci/fcisolver.py`, `src/ormas_ci/hamiltonian.py`

### Task 2.3: Davidson Eigensolver + Einsum Sigma Vector — Implemented

Replaced `scipy.sparse.linalg.eigsh` (ARPACK Lanczos) with a
pure-Python Davidson-Liu eigensolver and a pure-Python einsum-based
sigma vector that eliminates PySCF's per-sigma-call ERI preprocessing
overhead.

**What was implemented (differs from original plan):**

Instead of using PySCF's `davidson1()`, a standalone Davidson-Liu
solver was implemented in `src/ormas_ci/davidson.py` (~170 lines).
This avoids interface issues with PySCF's Davidson and provides
better debuggability.

Instead of using PySCF's `selected_ci.contract_2e()` for the sigma
vector, a pure-Python einsum-based sigma was implemented in
`src/ormas_ci/sigma.py` (~150 lines). This precomputes dense
single-excitation operator matrices E[p,q,a',a] for each spin channel
and ERI-contracted intermediates once per solve, then computes each
sigma call via a few NumPy einsum contractions.

**Key insight:** The bottleneck in the PySCF selected_ci path was NOT
the C kernel (<1ms) but the Python-level ERI preprocessing inside
`contract_2e()` (~150ms per call). The einsum approach precomputes
all intermediates once, reducing per-sigma cost to ~10-15ms.

**Derivation:** Uses the identity a+_p a+_r a_s a_q = E_pq E_rs -
delta(q,r) E_ps, absorbing the delta correction into effective
one-electron integrals: h1e_eff = h1e - 0.5 * einsum('prrs->ps', eri).

**Implementation:**
- `src/ormas_ci/davidson.py`: Davidson-Liu solver with subspace
  expansion, restart, multi-root support, diagonal preconditioning
- `src/ormas_ci/sigma.py`: `SigmaEinsum` class with precomputed
  excitation matrices and einsum-based sigma (aa, bb, ab terms)
- `src/ormas_ci/fcisolver.py`: `_solve_iterative()` dispatches to
  Davidson+einsum for spaces with <= 300 unique strings/channel,
  falls back to PySCF SCI + ARPACK eigsh for larger spaces

**Performance:** ~20 Davidson iterations at ~10-15ms/sigma vs ~41
ARPACK iterations at ~150ms/sigma. Combined ~20x improvement on
the iterative solve for CAS(8,8).

**Memory:** Dense excitation matrices scale as O(norb^2 * n_str^2)
per channel. CAS(8,8) with 70 unique strings: ~2.5 MB. Feasible
up to ~300 unique strings (~50 MB), beyond which the SCI fallback
is used.

**Files:** `src/ormas_ci/davidson.py` (new), `src/ormas_ci/sigma.py`
(new), `src/ormas_ci/fcisolver.py`

### Task 2.4: Implement ORMAS-Aware Diagonal Preconditioner

The Davidson solver needs diagonal Hamiltonian elements for
preconditioning. The current `make_hdiag()` in `fcisolver.py`
loops over determinants in Python calling `_compute_diagonal()`.
This should use vectorized computation.

**Work:**
- Implement `make_hdiag()` that computes diagonal elements for the
  ORMAS determinant space using batch numpy operations:
  ```python
  def make_hdiag(h1e, eri, norb, nelec, alpha_strings, beta_strings):
      n_det = len(alpha_strings)
      hdiag = np.zeros(n_det)
      # Vectorize over all determinants using advanced indexing
      for i in range(n_det):
          occ_a = bits_to_indices(int(alpha_strings[i]))
          occ_b = bits_to_indices(int(beta_strings[i]))
          # ... numpy sum over h1e and h2e diagonal contributions
  ```
- Alternatively, adapt PySCF's `direct_spin1.make_hdiag()` which uses
  the C function `FCImake_hdiag_uhf`. This function operates on the
  full CAS space, so the result would need to be indexed into the
  ORMAS subset. If the ORMAS space is a large fraction of the full
  CAS (e.g., >50%), this approach is efficient.
- For ORMAS spaces that are a small fraction of full CAS, a custom
  implementation is more efficient.

**Files:** `src/ormas_ci/fcisolver.py`

### Task 2.5: C-Level RDM Construction via PySCF

Replace the Python RDM loops with PySCF's C-level RDM routines
adapted for the ORMAS determinant space.

**Work:**
- PySCF's `pyscf.fci.rdm` module provides C functions for RDM
  construction: `FCImake_rdm1a`, `FCImake_rdm1b`, `FCIrdm12_drv`
- These operate on the full CAS determinant space with link tables
- For the ORMAS restricted space, use the selected CI link tables
  (from Task 2.1) with `selected_ci`'s RDM routines, or adapt
  the full-CAS RDM routines to work with ORMAS-specific link tables
- If PySCF's selected_ci module does not expose RDM construction
  directly, implement a wrapper that:
  1. Uses the ORMAS CI vector (in the restricted basis)
  2. Expands it to the full CAS basis (zero-padding excluded
     determinants)
  3. Calls PySCF's standard RDM routines
  4. This is only efficient if the ORMAS space is a large fraction
     of full CAS; otherwise, a custom C implementation is needed
     (see Phase 3)

**Files:** `src/ormas_ci/rdm.py`, `src/ormas_ci/fcisolver.py`

### Task 2.6: Preserve Explicit Hamiltonian Path as Fallback

The explicit Hamiltonian matrix is still valuable for:
- Small CI spaces (n_det < 200) where construction is cheap
- Debugging and correctness verification
- The `contract_2e()` method which uses the cached H matrix
- Benchmarking against the matrix-free path

**Work:**
- Keep the current `build_ci_hamiltonian()` and `solve_ci()` as
  the fallback path
- Add a `use_direct_ci` flag to `ORMASFCISolver.__init__()`:
  ```python
  def __init__(self, config, use_direct_ci=True):
      self.use_direct_ci = use_direct_ci
  ```
- In `kernel()`, dispatch based on the flag and determinant count:
  ```python
  if self.use_direct_ci and n_det > direct_ci_threshold:
      # Matrix-free Davidson path (Tasks 2.2-2.3)
  else:
      # Explicit Hamiltonian path (current code)
  ```
- Default `direct_ci_threshold` to ~200 determinants (below this,
  explicit construction is competitive with Davidson setup overhead)

**Files:** `src/ormas_ci/fcisolver.py`

### Task 2.7: Integration Tests for Matrix-Free Path — Implemented

`tests/test_sci_integration.py` compares explicit Hamiltonian and
SCI matrix-free paths on N2/STO-3G CAS(6,6) (400 determinants):
energy, RDM1/RDM1s/RDM12, spin_square, make_hdiag, contract_2e,
multi-root eigenvalue validity, nroots >= n_det fallback, and
ORMAS-restricted variational bound.

---

## Phase 3: Compiled Extensions (100-500x Speedup)

**Goal:** Implement performance-critical inner loops in C/C++ with
pybind11 bindings. Target: competitive with PySCF's native FCI
for equivalent-size determinant spaces.

**Prerequisite:** Phase 2 must be complete first. Phase 3 optimizes
the remaining bottlenecks that PySCF's selected_ci infrastructure
does not fully address (e.g., ORMAS-specific link table generation,
custom RDM construction for restricted spaces).

### Task 3.1: C Extension for ORMAS-Specific Link Table Generation

If PySCF's `selected_ci.cre_des_linkstr()` is a bottleneck for large
ORMAS spaces, implement a custom C version that exploits ORMAS
subspace structure.

**Work:**
- ORMAS constraints mean that excitations within a subspace are always
  allowed, while cross-subspace excitations may be forbidden. A custom
  link table generator can skip forbidden cross-subspace excitations
  during construction, reducing the link table size.
- Implement in C with pybind11:
  ```cpp
  // ormas_linkstr.cpp
  void ormas_cre_des_linkstr(
      const int64_t* strings, int n_strings, int norb, int nelec,
      const int* subspace_bounds, int n_subspaces,
      const int* min_occ, const int* max_occ,
      int* link_table, int* n_links_per_string
  );
  ```
- Build with `setuptools` C extension or `pybind11` in `pyproject.toml`

**Expected speedup:** 10-50x on link table generation (one-time cost
per solver invocation, but significant for CASSCF where kernel() is
called many times during orbital optimization).

**Files:** `src/ormas_ci/_ext/ormas_linkstr.cpp` (new),
`pyproject.toml` (build config)

### Task 3.2: C Extension for ORMAS-Aware Sigma Vector

If PySCF's `selected_ci.contract_2e()` does not fully exploit ORMAS
subspace structure, implement a custom sigma vector routine.

**Work:**
- ORMAS subspace constraints create block structure in the
  Hamiltonian: excitations within a subspace are dense, while
  cross-subspace excitations are sparse. A custom sigma routine
  can exploit this block structure for better cache locality.
- Implement the sigma vector computation in C, organized by
  subspace blocks:
  ```cpp
  void ormas_contract_2e(
      const double* eri, const double* ci,
      int norb, int nelec_a, int nelec_b,
      const int64_t* alpha_strings, const int64_t* beta_strings,
      int n_det_a, int n_det_b,
      const int* link_a, const int* link_b,
      double* sigma
  );
  ```
- Bind with pybind11 and integrate into the solver as an alternative
  to `selected_ci.contract_2e()`

**Expected speedup:** 2-5x beyond PySCF's generic selected_ci sigma
vector, from exploiting ORMAS block structure.

**Files:** `src/ormas_ci/_ext/ormas_contract.cpp` (new)

### Task 3.3: C Extension for Bit Manipulation Utilities

Replace the remaining Python bit manipulation utilities with C
implementations for the few cases where they are still called in
tight loops (e.g., determinant enumeration, RDM construction).

**Work:**
- Implement in C:
  - `popcount_array(int64_t* arr, int n)` — batch popcount
  - `bits_to_indices_array(int64_t* arr, int n, int** out)` — batch
    orbital extraction
  - `compute_phase_array(int64_t* strings, int* p, int* q, int n)` —
    batch phase computation
- These are useful for any remaining Python loops that were not
  replaced by PySCF's selected_ci routines

**Expected speedup:** Marginal if Phase 2 is complete (most hot loops
already use PySCF's C routines). Primarily useful for determinant
enumeration and debugging utilities.

**Files:** `src/ormas_ci/_ext/bitutils.cpp` (new)

### Task 3.4: Custom C-Level RDM for ORMAS Spaces

If the Phase 2 RDM approach (expanding to full CAS and calling PySCF's
RDM routines) is too expensive for small ORMAS fractions, implement
a custom C-level RDM construction.

**Work:**
- Implement `ormas_make_rdm12()` in C using the ORMAS link tables:
  ```cpp
  void ormas_make_rdm12(
      const double* ci, int n_det,
      const int64_t* alpha_strings, const int64_t* beta_strings,
      const int* link_a, const int* link_b,
      int norb,
      double* rdm1, double* rdm2
  );
  ```
- The algorithm follows PySCF's `FCIrdm12_drv` pattern but operates
  on the restricted determinant set with ORMAS link tables
- This is the most complex C extension and should be implemented last

**Expected speedup:** 50-200x on RDM construction compared to the
Python implementation. Critical for CASSCF workflows where RDMs are
computed at every orbital optimization step.

**Files:** `src/ormas_ci/_ext/ormas_rdm.cpp` (new)

### Task 3.5: Build System for C Extensions

Set up the build infrastructure for the C/C++ extensions.

**Work:**
- Add pybind11 to build dependencies in `pyproject.toml`
- Configure `setuptools` to compile the C extensions:
  ```toml
  [build-system]
  requires = ["setuptools>=68.0", "wheel", "pybind11>=2.11"]
  ```
- Add a `setup.py` or `pyproject.toml` ext_modules configuration
- Ensure the extensions are optional: the pure Python fallback
  must remain functional when the C extensions are not compiled
  (e.g., `try: from ormas_ci._ext import ...; except ImportError: ...`)
- Add CI job to build and test the C extensions on Linux, macOS,
  and Windows
- Add pre-built wheels for common platforms via `cibuildwheel`

**Files:** `pyproject.toml`, `setup.py` (if needed),
`.github/workflows/ci.yml`

---

## Phase 4: Advanced Algorithms (Production-Grade)

**Goal:** Implement algorithmic improvements that go beyond the
current explicit-space CI approach. Target: handle CAS(14,14)+ with
ORMAS restrictions (>100,000 determinants) in seconds.

### Task 4.1: Davidson with Deflation for Multi-Root

Implement proper Davidson deflation for computing multiple roots
simultaneously. The current multi-root implementation calls
`eigsh(k=nroots)` which computes all roots at once but with
limited control over convergence.

**Work:**
- Use PySCF's `davidson()` (multi-root version) with the custom
  sigma function from Phase 2
- Implement root-following for state-specific CASSCF optimization
- Add spin-state targeting via the existing `spin_square()` method
  as a penalty function

**Files:** `src/ormas_ci/solver.py`, `src/ormas_ci/fcisolver.py`

### Task 4.2: Perturbative Correction (ORMAS-ENPT2)

Implement Epstein-Nesbet perturbation theory (ENPT2) to estimate
the energy contribution from the excluded determinant space.

**Work:**
- After the ORMAS-CI solve, compute the second-order energy
  correction from determinants that are excluded by the ORMAS
  constraints but connected to the CI space by single and double
  excitations
- The correction is:
  ```
  E_PT2 = sum_I |<I|H|Psi_0>|^2 / (E_0 - <I|H|I>)
  ```
  where I runs over excluded determinants
- Use the ORMAS link tables to enumerate which excluded determinants
  are connected to the CI space
- Report the PT2 correction alongside the variational energy to
  give a better estimate of the full CAS energy from a restricted
  calculation

**Significance:** PT2 corrections allow much more aggressive ORMAS
restrictions (smaller CI space, faster calculation) while maintaining
chemical accuracy. This is especially valuable for quantum resource
estimation, where the classical PT2 correction quantifies how much
error the quantum computer would need to recover.

**Files:** `src/ormas_ci/perturbation.py` (new),
`src/ormas_ci/fcisolver.py`

### Task 4.3: Automatic Subspace Partitioning

Implement heuristics for automatically determining ORMAS subspace
boundaries from orbital character analysis.

**Work:**
- Use PySCF's Mulliken population analysis or intrinsic atomic
  orbital (IAO) decomposition to classify active orbitals by
  atomic character
- Cluster orbitals into subspaces based on:
  - Spatial symmetry (sigma/pi/delta)
  - Atomic center (metal d vs ligand pi)
  - Orbital energy grouping
- Set occupation bounds based on chemical intuition:
  - Core-like orbitals: min_electrons close to max
  - Valence: flexible bounds
  - Virtual-like: max_electrons close to 0
- Integrate with QDK/Chemistry's AVAS active space selection:
  AVAS identifies which AOs contribute to each active MO, which
  directly maps to subspace assignments

**Files:** `src/ormas_ci/auto_partition.py` (new),
`src/ormas_ci/fcisolver.py`

### Task 4.4: Orbital Optimization Acceleration for CASSCF

Optimize the CASSCF orbital optimization loop which calls `kernel()`,
`make_rdm12()`, and `contract_2e()` repeatedly.

**Work:**
- Cache and reuse link tables across CASSCF iterations (the
  determinant strings change only if the active space changes,
  not during orbital rotation)
- Implement efficient CI vector transformation under orbital
  rotation using the existing `transform_ci_for_orbital_rotation()`
  as a warm start for the Davidson solver
- Profile the full CASSCF loop to identify remaining bottlenecks
  (integral transformation, orbital gradient computation, etc.)

**Files:** `src/ormas_ci/fcisolver.py`

### Task 4.5: Parallel Sigma Vector Computation

Parallelize the sigma vector computation across multiple CPU cores.

**Work:**
- The sigma vector computation over link tables is embarrassingly
  parallel: different determinant blocks can be processed
  independently
- Use OpenMP in the C extensions (Phase 3) for thread-level
  parallelism:
  ```cpp
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_det; i++) {
      // Process links for determinant i
  }
  ```
- Alternatively, use Python multiprocessing for the pure-Python
  fallback path (less efficient due to GIL and serialization)

**Expected speedup:** Linear in core count (4-16x on modern hardware).

**Files:** `src/ormas_ci/_ext/ormas_contract.cpp`

---

## Phase 5: Ecosystem and Release

**Goal:** Package the high-performance solver for distribution and
establish it as a production tool for quantum chemistry workflows.

### Task 5.1: Comprehensive Benchmark Suite

Expand benchmarks to cover a wider range of chemically relevant
systems and active space sizes.

**Work:**
- Add transition metal complexes: Fe(II) porphyrin, Cr2 dimer
- Add organic photochemistry: ethylene excited states, butadiene
- Scaling study from CAS(4,4) to CAS(16,16) with ORMAS restrictions
- Memory usage benchmarks alongside timing
- Compare against other restricted CI codes: GAMESS ORMAS, Forte
  (if available), OpenMolcas RASSCF

**Files:** `benchmarks/`

### Task 5.2: Documentation for Production Use

Update documentation to reflect the high-performance implementation.

**Work:**
- Update `docs/design/performance.md` with Phase 2-4 benchmarks
- Add installation guide for C extensions
- Add CASSCF workflow tutorial with ORMAS
- Document the automatic subspace partitioning heuristics
- Add quantum resource estimation workflow documentation

**Files:** `docs/`

### Task 5.3: Version 1.0 Release

Prepare for a stable release.

**Work:**
- Update version to 1.0.0 in `pyproject.toml`
- Update development status classifier from Alpha to Production/Stable
- Publish to PyPI with pre-built wheels
- Write release notes summarizing performance improvements
- Update CHANGELOG.md

**Files:** `pyproject.toml`, `CHANGELOG.md`

### Task 5.4: Paper / Technical Report

Write a technical report or paper describing the ORMAS-CI solver,
its integration with PySCF and QDK/Chemistry, and the quantum
resource estimation results.

**Work:**
- Introduction: ORMAS-CI method and motivation
- Implementation: PySCF plugin architecture, determinant enumeration,
  Hamiltonian construction strategies
- Performance: benchmarks against PySCF native FCI, scaling analysis
- Quantum computing: resource estimation, subspace factorization,
  qubit reduction results
- Target venue: Journal of Chemical Theory and Computation (JCTC),
  Journal of Chemical Physics (JCP), or arXiv preprint

---

## Summary: Expected Performance at Each Phase

| Phase | CAS(8,8) Time | vs PySCF | Approach |
|-------|--------------|----------|----------|
| v0.1.0 (baseline) | 55s | 85x slower | Pure Python, explicit H |
| v0.2.0 (SCI backend) | 4.3s | ~15x slower | SCI sigma + eigsh |
| **v0.2.0 (current)** | **~0.3-0.5s** | **~1-2x** | **Davidson + einsum sigma** |
| Phase 3 | ~0.1-0.3s | ~1x (parity) | Custom C extensions |
| Phase 4 | <0.1s | Competitive | Direct CI + parallelism |

For ORMAS-restricted spaces (where n_det is smaller than full CAS),
the solver is **faster** than PySCF native FCI on the equivalent
full CAS problem, because it operates on the reduced space.
