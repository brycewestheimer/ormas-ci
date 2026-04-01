# Performance Tuning

## Estimating Cost Before Running

Use `count_determinants()` to estimate the size of your CI space
before starting a calculation:

```python
from pyscf.ormas_ci import count_determinants, casci_determinant_count

n_det = count_determinants(config)
n_cas = casci_determinant_count(ncas, nelecas)
print(f"ORMAS determinants: {n_det}")
print(f"Full CAS determinants: {n_cas}")
print(f"Reduction factor: {n_cas / n_det:.1f}x")
```

This is essentially free and helps you decide whether the constraints
are worth the complexity.

## Solver Path Selection

The solver automatically selects one of three paths based on the
determinant space size:

| Path | Condition | Character |
|------|-----------|-----------|
| Dense `eigh` | n_det <= `direct_ci_threshold` (200) | Exact, fast for small spaces |
| Davidson + einsum | n_det > 200 and <= `einsum_string_threshold` (300) unique strings/channel | Iterative, pure-Python |
| PySCF SCI + ARPACK | > 300 unique strings/channel | Iterative, C-level sigma vector |

You can adjust these thresholds:

```python
solver = ORMASFCISolver(config)
solver.direct_ci_threshold = 500   # use dense path for larger spaces
solver.einsum_string_threshold = 400  # extend einsum range
```

**Dense path** gives exact results but stores an n_det x n_det matrix:
- 200 dets: ~320 KB
- 500 dets: ~2 MB
- 1000 dets: ~8 MB

**Davidson+einsum** is typically the fastest iterative path for moderate
spaces, converging in ~20 iterations vs ~40 for ARPACK.

## Reducing Determinant Count

If your calculation is too slow, reduce the determinant count:

1. **Tighten occupation bounds.** Allow fewer inter-subspace excitations.
   Start with +/- 2 electrons from the "natural" occupation and try +/- 1.
   See [Choosing Subspaces](../guides/choosing_subspaces.md).

2. **Add more subspaces.** Splitting a large subspace into two with
   tighter constraints reduces the combinatorial explosion.

3. **Reduce the active space.** Fewer active orbitals means fewer
   determinants. Ensure weakly correlated orbitals are excluded.

4. **Use RAS partitioning.** The classic RAS(1/2/3) scheme with limited
   holes and particles is often an effective default.

## When ORMAS-CI Is Faster vs Slower

**Faster than PySCF native FCI:** When ORMAS constraints significantly
reduce the determinant count. An ORMAS-restricted CAS(8,8) with 1,000
determinants solves faster than PySCF's full 4,900-determinant CAS(8,8).

**Comparable to PySCF:** For small active spaces (CAS(4,4) and below),
both solvers are dominated by PySCF's CASCI framework overhead.

**Slower than PySCF:** When running as unrestricted full CAS (no
reduction), the Python-level overhead in ORMAS-CI adds cost. Use PySCF's
native solver for full CAS calculations.

## The 63-Orbital Limit

Determinant strings are stored as 64-bit integers, limiting the active
space to 63 orbitals. In practice, active spaces above ~30 orbitals
produce determinant counts too large for explicit enumeration, so this
limit is never the binding constraint.

See [Bit String Representation](../architecture/bitstrings.md) for
details.

## Memory Considerations

| Solver Path | Memory Usage |
|-------------|-------------|
| Dense | n_det x n_det x 8 bytes (explicit Hamiltonian matrix) |
| Davidson + einsum | ~n_det x n_orbitals^2 (excitation matrices) |
| PySCF SCI + ARPACK | ~n_det x n_orbitals^2 (link tables) |

For detailed benchmarks, see
[Performance Considerations](../design/performance.md).
