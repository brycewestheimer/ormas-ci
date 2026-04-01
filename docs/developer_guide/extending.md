# Extending ORMAS-CI

This guide covers how to add new functionality to ORMAS-CI, focusing
on the most likely extension points.

## Adding a New Solver Path

The solver path selection happens in `fcisolver.py` inside `kernel()`.
To add a new path:

### 1. Implement a Sigma Engine

Create a new sigma vector class following the pattern in `sigma.py`.
A sigma engine must support:

```python
class SigmaMyEngine:
    def __init__(self, h1e, h2e, alpha_strings, beta_strings, ...):
        """Precompute anything needed for repeated H * c calls."""
        ...

    def __call__(self, c):
        """Compute sigma = H @ c and return as a 1D array."""
        ...
```

The engine is constructed once per `kernel()` call and then invoked
repeatedly by the iterative eigensolver (Davidson or Lanczos).

### 2. Add Path Selection Logic

In `kernel()`, add a condition for when your new path should be used:

```python
if n_det <= self.direct_ci_threshold:
    # existing dense path
elif n_unique_strings <= self.einsum_string_threshold:
    # existing Davidson + einsum path
elif your_condition:
    # your new path
else:
    # existing SCI fallback path
```

### 3. Add a Threshold Attribute

If your path has a configurable threshold, add it as an instance
attribute in `__init__()` alongside `direct_ci_threshold` and
`einsum_string_threshold`.

### 4. Test Against References

Your new path must reproduce the same energy as the existing paths.
Write tests comparing against:
- PySCF native CASCI (gold standard)
- The dense `eigh` path (for small systems)

## Adding a New Configuration Class

To support a new constraint model (e.g., GAS-CI with cumulative
constraints):

1. Define the new config dataclass in `subspaces.py`
2. Add a `to_ormas_config()` method if the new model can be converted
   to ORMAS-style local constraints, or modify `determinants.py` if
   the enumeration algorithm needs to change
3. Export from `__init__.py`
4. Add tests and documentation

## Benchmarking

Use the existing benchmark suite to validate performance:

```bash
python benchmarks/bench_vs_pyscf.py
```

This compares ORMAS-CI against PySCF's native FCI solver. Add new
benchmark systems by following the existing patterns in the script.

For quantum resource benchmarks:

```bash
python benchmarks/bench_qdk_quantum.py --all
```

See [Performance Considerations](../design/performance.md) for current
benchmark results and performance targets.

## Interface Contracts

### `kernel()` Signature

```python
def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
    """Returns (energy, ci_vector) or ([energies], [ci_vectors])."""
```

- `h1e`: 2D array of one-electron integrals
- `eri`: compressed or 4D two-electron integrals
- `norb`: number of active orbitals
- `nelec`: `(n_alpha, n_beta)` or total electron count
- Returns: scalar energy + 1D CI vector (single root), or lists
  (multi-root)

### RDM Methods

```python
def make_rdm1(self, ci, norb, nelec) -> np.ndarray:
    """Returns (norb, norb) 1-particle density matrix."""

def make_rdm12(self, ci, norb, nelec) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rdm1, rdm2) where rdm2 is (norb, norb, norb, norb)."""
```

These delegate to PySCF's `selected_ci` C routines via the SCI
mapping.

## Development Roadmap

See the [Development Roadmap](../design/future.md) for planned
extensions including compiled C extensions (Phase 2), perturbative
corrections (Phase 3), and automatic subspace partitioning.
