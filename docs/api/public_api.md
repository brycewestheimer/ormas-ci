# Public API Reference

The public API consists of the solver class, configuration classes, and utility functions.
All are importable directly from `pyscf.ormas_ci`:

```python
from pyscf.ormas_ci import (
    ORMASFCISolver,
    SFORMASFCISolver,
    ORMASConfig,
    RASConfig,
    SFORMASConfig,
    SFRASConfig,
    Subspace,
    build_determinant_list,
    casci_determinant_count,
    count_determinants,
    generate_sf_determinants,
    count_sf_determinants,
    validate_reference_consistency,
)
```

## Solver

```{autoclass} pyscf.ormas_ci.ORMASFCISolver
:members: __init__, kernel, make_rdm1, make_rdm1s, make_rdm12, make_rdm12s, spin_square, transform_ci_for_orbital_rotation, large_ci
```

### Solver Tuning

The solver automatically selects an eigensolver path based on the
determinant space size.  These attributes can be adjusted on the
`ORMASFCISolver` instance after construction:

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `direct_ci_threshold` | 200 | Determinant count below which the explicit Hamiltonian + dense `eigh` path is used. Increase if you have memory to spare and want exact diagonalisation for larger spaces. |
| `einsum_string_threshold` | 300 | Unique-string-per-spin-channel count below which the Davidson + `SigmaEinsum` (pure-Python) iterative path is used. Above this the solver falls back to PySCF's C-level `selected_ci` sigma with ARPACK. |
| `conv_tol` | 1e-12 | Davidson convergence tolerance on the residual norm. Loosen for faster but less precise iterative solves. |
| `max_cycle` | 100 | Maximum Davidson iterations before the solver declares non-convergence. |
| `max_space` | 12 | Maximum subspace dimension per root before Davidson restarts. |
| `level_shift` | 0.001 | Preconditioner level shift for Davidson. Increase if convergence is oscillatory. |
| `lindep` | 1e-14 | Linear-dependence threshold for Davidson subspace expansion. |

### PySCF Compatibility Methods

The following methods implement the full PySCF FCI solver interface,
enabling compatibility with PySCF addons such as `fix_spin_` and
iterative CASSCF workflows.  They are part of the public API but are
primarily called by PySCF internals rather than by user code directly.

```{autoclass} pyscf.ormas_ci.ORMASFCISolver
:members: absorb_h1e, contract_1e, contract_2e, contract_ss, make_hdiag, pspace, get_init_guess, dump_flags
:noindex:
```

---

## Spin-Flip Solver

```{autoclass} pyscf.ormas_ci.SFORMASFCISolver
:members: __init__, kernel
```

`SFORMASFCISolver` inherits all methods from {class}`~pyscf.ormas_ci.ORMASFCISolver`
including `make_rdm1`, `make_rdm12`, `spin_square`, and all PySCF compatibility methods.

---

## Configuration

```{autoclass} pyscf.ormas_ci.subspaces.Subspace
:members:
```

```{autoclass} pyscf.ormas_ci.subspaces.ORMASConfig
:members:
```

```{autoclass} pyscf.ormas_ci.subspaces.RASConfig
:members:
```

```{autoclass} pyscf.ormas_ci.subspaces.SFORMASConfig
:members:
```

```{autoclass} pyscf.ormas_ci.subspaces.SFRASConfig
:members:
```

---

## Determinant Utilities

```{autofunction} pyscf.ormas_ci.determinants.build_determinant_list
```

```{autofunction} pyscf.ormas_ci.determinants.count_determinants
```

```{autofunction} pyscf.ormas_ci.determinants.casci_determinant_count
```

---

## Spin-Flip Utilities

```{autofunction} pyscf.ormas_ci.spinflip.generate_sf_determinants
```

```{autofunction} pyscf.ormas_ci.spinflip.count_sf_determinants
```

```{autofunction} pyscf.ormas_ci.spinflip.validate_reference_consistency
```
