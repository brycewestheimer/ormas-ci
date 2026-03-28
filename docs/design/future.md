# Development Roadmap

Remaining development priorities for ORMAS-CI, organized by category.
For completed work see the [CHANGELOG](https://github.com/brycewestheimer/ormas-ci/blob/main/CHANGELOG.md).

## Current State

ORMAS-CI v0.2.0 is a fully functional PySCF fcisolver plugin with:

- Three-path CI solver (explicit Hamiltonian, Davidson+einsum, PySCF SCI fallback)
- Approaching parity with PySCF native FCI for typical ORMAS space sizes
- Full CASCI/CASSCF integration with RDMs, spin diagnostics, orbital optimization
- QDK/Chemistry integration with real IQPE/VQE simulation, fault-tolerant
  resource estimation, and wavefunction-aware Hamiltonian filtering
- 183 tests, 6 example scripts, 7 benchmark model systems

### Known Approximations

- **Rotation depth**: The `rotationDepth` fed to the Azure Quantum Resource
  Estimator uses total rotation count as an upper bound (sequential circuit
  assumption). A DAG-based analysis would give a tighter bound.
- **Circuit depth**: The `circuit_depth` benchmark metric is
  `sum(gate_counts)`, not true critical-path depth.
- **Excitation operator proxy** (`bench_vs_pyscf.py`): Off-diagonal
  Hamiltonian element count is a classical proxy. Real Pauli term counts
  are in `bench_qdk_quantum.py`.

---

## Phase 1: Quantum Resource Estimation

Extend the quantum benchmark suite to provide more compelling
resource comparisons between ORMAS-restricted and full CAS spaces.

### 1.1 Double-Factorized Qubitization Resource Estimation

FCIDUMP generation is scaffolded via `_generate_fcidump()`. Full integration
requires embedding the QDK df-chemistry Q# sample code to produce physical
qubit requirements using the state-of-the-art algorithm for fault-tolerant
quantum chemistry (double-factorized qubitization + QPE).

**Work:**

1. Clone the QDK samples repository and extract `df-chemistry/` Q# code
2. Use `qsharp.compile()` + `qsharp.estimate()` to run from Python
3. Feed FCIDUMP files for both ORMAS-restricted and full CAS active spaces
4. Parse `EstimatorResult` for physical qubit, runtime, T-state counts
5. Report ORMAS vs CAS reduction in fault-tolerant physical resources

**Alternative:** Use PennyLane's `qml.estimator.DoubleFactorization` for
analytical Toffoli gate counts from one- and two-electron integrals,
without the Q# dependency.

**Files:** `benchmarks/bench_qdk_quantum.py`

### 1.2 PennyLane Toffoli Gate Counts

Add PennyLane's `DoubleFactorization` and `FirstQuantization` resource
estimators as an optional dependency. These compute Toffoli gate counts
and logical qubit requirements for qubitization-based QPE directly from
molecular integrals, providing an independent estimate complementary to
the QDK-based approach.

**Files:** `benchmarks/bench_qdk_quantum.py`, `pyproject.toml`

### 1.3 Per-Subspace Qubit Hamiltonians

Construct separate qubit Hamiltonians for each ORMAS subspace to quantify
resource savings from subspace factorization at the qubit level. Currently,
subspace factorization reports classical determinant reduction but does not
construct per-subspace quantum circuits.

**Files:** `benchmarks/bench_qdk_quantum.py`

### 1.4 True Circuit Depth (DAG-Based)

Replace the sequential gate count approximation for circuit depth with
proper DAG analysis. Parse the QASM circuit into a directed acyclic graph
where edges represent qubit dependencies, then compute the critical path
length. This would also provide a tighter `rotationDepth` for the Azure
Quantum Resource Estimator.

**Files:** `benchmarks/bench_qdk_quantum.py`

### 1.5 T-Gate Decomposition from Rotations

Decompose arbitrary rotation angles into T-gate sequences using
Ross-Selinger synthesis. Report the actual T-gate count for each rotation
precision target, providing a `tCount` for `LogicalCounts` instead of
relying solely on `rotationCount`.

**Files:** `benchmarks/bench_qdk_quantum.py`

---

## Phase 2: Compiled Extensions

Implement performance-critical inner loops in C/C++ with pybind11
bindings. Target: full parity with PySCF native FCI for equivalent-size
determinant spaces.

### 2.1 ORMAS-Specific Link Table Generation

Custom C implementation that exploits ORMAS subspace structure — excitations
within a subspace are always allowed while cross-subspace excitations may
be forbidden. Skip forbidden excitations during link table construction to
reduce table size.

```cpp
void ormas_cre_des_linkstr(
    const int64_t* strings, int n_strings, int norb, int nelec,
    const int* subspace_bounds, int n_subspaces,
    const int* min_occ, const int* max_occ,
    int* link_table, int* n_links_per_string
);
```

**Expected speedup:** 10-50x on link table generation (significant for
CASSCF where `kernel()` is called many times).

**Files:** `src/ormas_ci/_ext/ormas_linkstr.cpp` (new), `pyproject.toml`

### 2.2 ORMAS-Aware Sigma Vector

Custom sigma vector computation exploiting the block structure created by
ORMAS subspace constraints: intra-subspace excitations are dense while
cross-subspace excitations are sparse. Better cache locality than PySCF's
generic `selected_ci.contract_2e()`.

**Expected speedup:** 2-5x beyond PySCF's generic selected\_ci sigma.

**Files:** `src/ormas_ci/_ext/ormas_contract.cpp` (new)

### 2.3 Custom C-Level RDM Construction

If the current approach (delegating to PySCF's `selected_ci` RDM routines
via 1D/2D mapping) becomes a bottleneck for small ORMAS fractions of full
CAS, implement ORMAS-native RDM construction in C using the ORMAS link
tables directly.

**Expected speedup:** 50-200x vs pure-Python RDM construction.

**Files:** `src/ormas_ci/_ext/ormas_rdm.cpp` (new)

### 2.4 Build Infrastructure

- Add pybind11 to build dependencies
- Configure setuptools for C extension compilation
- Ensure pure-Python fallback remains functional (`try: from ._ext import ...`)
- CI job for building/testing on Linux, macOS, Windows
- Pre-built wheels via `cibuildwheel`

**Files:** `pyproject.toml`, `.github/workflows/ci.yml`

---

## Phase 3: Advanced Algorithms

Algorithmic improvements beyond the current explicit-space CI. Target:
handle CAS(14,14)+ with ORMAS restrictions (>100,000 determinants) in
seconds.

### 3.1 Perturbative Correction (ORMAS-ENPT2)

Epstein-Nesbet second-order perturbation theory to estimate the energy
contribution from determinants excluded by ORMAS constraints:

```
E_PT2 = sum_I |<I|H|Psi_0>|^2 / (E_0 - <I|H|I>)
```

where I runs over excluded determinants connected to the CI space by
single and double excitations. This allows more aggressive ORMAS
restrictions while maintaining chemical accuracy — especially valuable
for quantum resource estimation, where the classical PT2 correction
quantifies how much error the quantum computer would need to recover.

**Files:** `src/ormas_ci/perturbation.py` (new), `src/ormas_ci/fcisolver.py`

### 3.2 Automatic Subspace Partitioning

Heuristics for automatically determining ORMAS subspace boundaries:

- Mulliken population / intrinsic atomic orbital (IAO) analysis to
  classify active orbitals by atomic character
- Clustering by spatial symmetry (sigma/pi/delta), atomic center
  (metal d vs ligand pi), or orbital energy grouping
- Integration with QDK/Chemistry's AVAS active space selection

**Files:** `src/ormas_ci/auto_partition.py` (new)

### 3.3 GAS-CI (Cumulative Constraints)

Generalized Active Space CI with cumulative electron constraints.
Requires modifying the feasibility check in the recursive distribution
enumeration to track cumulative counts instead of per-subspace bounds.
The rest of the pipeline (solver, RDMs, etc.) is identical to ORMAS.

**Files:** `src/ormas_ci/determinants.py`, `src/ormas_ci/subspaces.py`

### 3.4 ORMAS + Selected CI Hybrid

Within the ORMAS-restricted space, apply selected CI (CIPSI/ASCI) to
further prune unimportant determinants. Combines ORMAS's structural
constraints with selected CI's numerical efficiency.

### 3.5 Parallel Sigma Vector

Parallelize the sigma vector over multiple CPU cores. The computation
over link tables is embarrassingly parallel — different determinant
blocks can be processed independently. Use OpenMP in C extensions or
Python multiprocessing for the pure-Python path.

**Expected speedup:** Linear in core count (4-16x on modern hardware).

### 3.6 Davidson with Deflation for Multi-Root

Proper Davidson deflation for computing multiple roots simultaneously
with better control over near-degenerate root convergence. Add
root-following for state-specific CASSCF and spin-state targeting via
`spin_square()` penalty.

**Files:** `src/ormas_ci/davidson.py`, `src/ormas_ci/fcisolver.py`

---

## Phase 4: Ecosystem and Release

### 4.1 Expanded Benchmark Suite

- Transition metal complexes: Fe(II) porphyrin, Cr2 dimer
- Organic photochemistry: ethylene excited states
- Scaling study from CAS(4,4) to CAS(16,16)
- Memory usage benchmarks
- Comparison against GAMESS ORMAS, Forte, OpenMolcas RASSCF

### 4.2 Direct QDK/Chemistry Plugin

Register ORMAS-CI as a QDK/Chemistry plugin through its factory system,
making it available as a first-class CI method alongside CASCI and ASCI.

### 4.3 State-Averaged CASCI/CASSCF

Construct state-averaged RDMs for multi-root orbital optimization.

### 4.4 Version 1.0 Release

- Update version and classifiers
- Publish to PyPI with pre-built wheels
- Write release notes and update CHANGELOG

### 4.5 Paper / Technical Report

Technical report covering the ORMAS-CI solver, PySCF/QDK integration,
and quantum resource estimation results. Target: JCTC, JCP, or arXiv.

---

## Performance Targets

| Milestone | CAS(8,8) Time | vs PySCF | Approach |
|-----------|--------------|----------|----------|
| v0.1.0 | 55s | 85x slower | Pure Python, explicit H |
| v0.2.0 (SCI) | 4.3s | ~15x slower | SCI sigma + eigsh |
| **v0.2.0 (current)** | **~0.3-0.5s** | **~1-2x** | **Davidson + einsum sigma** |
| Phase 2 | ~0.1-0.3s | ~1x (parity) | Custom C extensions |
| Phase 3 | <0.1s | Competitive | Direct CI + parallelism |

For ORMAS-restricted spaces (where n\_det is smaller than full CAS),
the solver is already **faster** than PySCF native FCI on the equivalent
full CAS problem because it operates on the reduced space.
