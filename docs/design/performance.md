# Performance Considerations

## Solver Architecture

ORMAS-CI uses three solver paths, selected automatically:

**Small spaces (n_det <= 200):** Explicit Hamiltonian matrix constructed
via Slater-Condon rules, diagonalized with `numpy.linalg.eigh`. Gives
exact eigenvalues. Vectorized excitation precomputation skips >50% of
determinant pairs.

**Medium spaces (n_det > 200, <= ~300 unique strings/channel):**
Pure-Python Davidson-Liu eigensolver (`davidson.py`) with einsum-based
sigma vector (`sigma.py`). Precomputes dense single-excitation operator
matrices and ERI-contracted intermediates once per solve, then computes
each sigma call via NumPy einsum contractions. This eliminates the
per-call ERI preprocessing overhead of PySCF's `selected_ci.contract_2e`
and reduces iteration count vs ARPACK Lanczos.

**Large spaces (> 300 unique strings/channel):** Falls back to PySCF's
C-level `pyscf.fci.selected_ci.contract_2e` for the sigma vector with
`scipy.sparse.linalg.eigsh` (ARPACK Lanczos). Used when the dense
excitation matrices would exceed memory limits.

RDMs, `make_hdiag`, and `spin_square` delegate to PySCF's C-level
`selected_ci` routines across all paths. No explicit Hamiltonian
matrix is stored for the medium or large paths.

Note: we use PySCF's `selected_ci` **module** as a computational
backend for operating on arbitrary determinant string sets — not as a
Selected CI method (CIPSI, ASCI). The determinant selection is entirely
ORMAS constraint-based.

The thresholds are configurable via `ORMASFCISolver.direct_ci_threshold`
(explicit H vs iterative, default 200) and
`ORMASFCISolver.einsum_string_threshold` (einsum vs SCI fallback,
default 300).

## Benchmarks

A benchmark suite is provided in `benchmarks/bench_vs_pyscf.py` that
compares ORMAS-CI against PySCF's native FCI solver. Note: the benchmark
script measures the explicit Hamiltonian path components directly; it
does not exercise the matrix-free SCI path used by `kernel()`.

```bash
python benchmarks/bench_vs_pyscf.py
```

### Performance vs PySCF Native FCI (full CAS, via kernel())

| System | n_det | PySCF | ORMAS-CI | Ratio |
|--------|-------|-------|----------|-------|
| H2 CAS(2,2) | 4 | 0.02s | 0.02s | ~1x |
| H2O CAS(4,4) | 36 | 0.001s | 0.002s | ~2x |
| N2 CAS(6,6) | 400 | 0.24s | 0.02s | 0.1x |
| N2 CAS(8,8) | 4,900 | 0.32s | 4.6s | ~15x |

At small sizes, both solvers are dominated by PySCF CASCI framework
overhead. At CAS(8,8), ORMAS-CI is ~15x slower due to Python-level
CI vector marshalling between ORMAS 1D and PySCF `selected_ci` 2D formats on
each sigma call.

### ORMAS-Restricted Performance (the real use case)

The value of ORMAS-CI is determinant space reduction. Restricted spaces
solve faster because the iterative solver operates on fewer determinants:

| System | Full CAS | ORMAS n_det | Reduction | ORMAS time | dE (mHa) |
|--------|----------|-------------|-----------|------------|----------|
| N2 CAS(6,6) sigma/pi | 400 | 381 | 1.1x | 0.016s | +0.08 |
| N2 CAS(8,8) RAS | 4,900 | 3,355 | 1.5x | 3.8s | +0.02 |
| H2O CAS(8,8) sigma/pi | 4,900 | 3,355 | 1.5x | 3.7s | +0.00 |

## Current Bottleneck

For the Davidson+einsum path (medium spaces), the bottleneck is the
per-sigma einsum contraction cost (~10-15ms per call for CAS(8,8)),
with ~20 Davidson iterations. The 1D↔2D CI vector conversion is
negligible (~0.025ms). Total iterative solve: ~0.3-0.5s for CAS(8,8).

For the SCI fallback path (large spaces), the bottleneck is PySCF's
per-call ERI preprocessing inside `selected_ci.contract_2e()` (~150ms
per call, of which <1ms is the actual C kernel). With ~41 ARPACK
iterations, total: ~6s for CAS(8,8).

For the explicit Hamiltonian path (n_det <= 200), the construction cost
is negligible at these sizes.

## Memory

The matrix-free path stores only the CI vector (n_det floats) and
link tables (~n_det * n_orbitals^2 ints). For CAS(8,8), this is ~5 MB
vs ~192 MB for the explicit Hamiltonian.

The explicit Hamiltonian path stores the full n_det x n_det matrix:
- n_det = 100: 80 KB
- n_det = 200: 320 KB

## Acceleration Paths

The Davidson+einsum implementation (Phase 2.3) addressed the dominant
bottleneck for typical ORMAS spaces. Further performance gains require:

1. **Custom C extensions** (pybind11) for ORMAS-specific link table
   generation and sigma vector computation that exploit subspace block
   structure (Phase 3).

2. **Numba JIT** (optional) for accelerating excitation matrix
   construction and RDM inner loops without requiring C compilation.

3. **OpenMP parallelism** in custom C extensions for thread-level
   parallelism on the sigma vector computation (Phase 4).

See `FUTURE_DEVELOPMENT.md` (Phases 3-4) for details.

## Quantum Resource Metrics

Real quantum computing metrics from QDK/Chemistry v1.0.2. All values
computed from actual QDK Jordan-Wigner mapping and Trotter circuit
construction -- no analytical estimates or proxy metrics.

```bash
python benchmarks/bench_qdk_quantum.py --all
```

### Active-Space Qubit Hamiltonians (Jordan-Wigner)

| System | Basis | Active Space | Qubits | Pauli Terms | Real CNOTs | Real Gates |
|--------|-------|-------------|--------|-------------|------------|------------|
| Ethylene | cc-pVDZ | CAS(2,2) | 4 | 15 | 36 | 99 |
| Formaldehyde | cc-pVDZ | CAS(4,3) | 6 | 62 | 262 | 684 |
| Ethylene | cc-pVTZ | CAS(4,4) | 8 | 61 | 264 | 613 |
| Butadiene | cc-pVDZ | CAS(4,4) | 8 | 105 | 576 | 1,329 |
| Ozone | cc-pVDZ | CAS(6,5) | 10 | 228 | 1,594 | 3,670 |

Gate counts are from real QASM3 circuit parsing of QDK's Trotter
decomposition + PauliSequenceMapper, not analytical formulas.

### Benchmark Comparison: CASCI vs RASCI vs ORMAS

**RAS systems** (occupied/frontier/virtual separation):

| System | Method | n_det | dE (mHa) | Notes |
|--------|--------|-------|----------|-------|
| Ethylene/cc-pVTZ CAS(4,4) | CASCI | 36 | - | Reference |
| | RASCI | 9 | +14.3 | 4x det reduction |
| | ORMAS | 35 | +0.001 | sigma/pi split, minimal error |
| Ozone/cc-pVDZ CAS(6,5) | CASCI | 100 | - | Reference |
| | RASCI | 55 | +0.04 | 1.8x det reduction |
| Formaldehyde/cc-pVDZ CAS(4,3) | CASCI | 9 | - | Reference |
| | RASCI | 5 | +22.0 | n/pi/pi* RAS |

**ORMAS systems** (each demonstrates a different partitioning strategy):

| System | Partitioning | n_det (CAS/ORMAS) | Reduction | dE (mHa) |
|--------|-------------|-------------------|-----------|----------|
| N2/cc-pVDZ CAS(6,6) | sigma/pi symmetry | 400/176 | 2.3x | +2.4 |
| FeO/cc-pVDZ CAS(8,8) | Fe 3d / O 2p metal/ligand | 784/225 | 3.5x | +0.4 |
| Butadiene/cc-pVDZ CAS(4,4) | C1=C2 / C3=C4 spatial | 36/18 | 2.0x | +271 |

FeO demonstrates excellent accuracy with tight charge-transfer constraints
(+/- 1 electron). Butadiene shows the expected cost of breaking conjugation.

See `benchmarks/bench_qdk_quantum.py` for full results including IQPE
simulation, VQE-style energy estimation, and method-specific state prep
circuit metrics.
