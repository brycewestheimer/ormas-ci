# Interpreting Results

## Energy

The total energy returned by `mc.kernel()` includes both the active
space CI energy and the core (frozen) electron contribution:

```python
e_tot, ci = mc.kernel()
print(f"Total energy: {e_tot:.10f} Hartree")
```

### Comparing Against CASCI

Always compare your ORMAS-CI energy against unrestricted CASCI to
assess the quality of your constraints:

```python
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
e_ref = mc_ref.kernel()[0]
print(f"ORMAS - CASCI: {(e_tot - e_ref) * 1000:.3f} mHa")
```

The ORMAS energy should be **at or above** the CASCI energy (variational
principle). Guidelines:

| Difference | Interpretation |
|------------|---------------|
| < 0.01 mHa | Negligible -- constraints exclude only unimportant configurations |
| 0.01 -- 1 mHa | Acceptable for most applications |
| 1 -- 10 mHa | Significant -- consider loosening constraints |
| > 10 mHa | Constraints are likely too tight |

See [Choosing Subspaces](../guides/choosing_subspaces.md) for guidance
on adjusting constraints.

### Determinant Reduction

Check how many determinants your constraints eliminated:

```python
from pyscf.ormas_ci import count_determinants, casci_determinant_count

n_ormas = count_determinants(config)
n_casci = casci_determinant_count(ncas, nelecas)
print(f"ORMAS: {n_ormas} / CASCI: {n_casci} ({n_ormas/n_casci:.1%})")
```

A good ORMAS partitioning achieves significant determinant reduction
(2x or more) with minimal energy loss (< 1 mHa).

## Natural Orbital Analysis

PySCF's `mc.analyze()` prints natural orbital occupations:

```python
mc.analyze()
```

Occupation numbers close to 2.0 or 0.0 indicate weakly correlated
orbitals that could potentially be moved to a more restricted subspace.
Occupations far from integer values (e.g., 1.3, 0.7) indicate strong
correlation -- these orbitals should be in the most flexible subspace.

## Spin Diagnostics

Check the spin state with `spin_square()`:

```python
ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, nelecas)
print(f"<S^2> = {ss:.6f}, 2S+1 = {mult:.4f}")
```

| Expected S^2 | State |
|-------------|-------|
| 0.000 | Singlet |
| 0.750 | Doublet |
| 2.000 | Triplet |

Deviations from expected values indicate spin contamination, which can
occur if the ORMAS constraints break spin symmetry. For spin-flip
calculations, verify that the target state has the expected multiplicity.

## Multi-Root Output

When `nroots > 1`, `mc.kernel()` returns lists:

```python
solver.nroots = 3
e_tot, ci = mc.kernel()
for i, e in enumerate(e_tot):
    ss, mult = mc.fcisolver.spin_square(ci[i], ncas, nelecas)
    print(f"Root {i}: E = {e:.10f}, <S^2> = {ss:.4f}, 2S+1 = {mult:.2f}")
```

Roots are ordered by energy. For systems with near-degenerate roots,
tighten `conv_tol` (e.g., 1e-12) to ensure reliable root ordering.

## Numerical Precision

The level of agreement with PySCF's native FCI depends on the solver
path used:

| Solver Path | Energy Agreement | When Used |
|------------|-----------------|-----------|
| Dense `eigh` | Exact (machine precision) | n_det <= 200 |
| Davidson + einsum | ~1e-10 Ha | 200 < n_det, <= 300 strings/channel |
| PySCF SCI + ARPACK | ~1e-10 Ha | > 300 strings/channel |

For full details, see
[Differences from Native PySCF](../design/pyscf_differences.md).
