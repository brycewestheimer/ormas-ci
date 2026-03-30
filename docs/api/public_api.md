# Public API Reference

The public API consists of the solver class, configuration classes, and utility functions.
All are importable directly from `pyscf.ormas_ci`:

```python
from pyscf.ormas_ci import (
    ORMASFCISolver,
    ORMASConfig,
    RASConfig,
    Subspace,
    build_determinant_list,
    count_determinants,
)
```

## Solver

```{autoclass} pyscf.ormas_ci.ORMASFCISolver
:members: __init__, kernel, make_rdm1, make_rdm1s, make_rdm12, make_rdm12s, spin_square, transform_ci_for_orbital_rotation, large_ci
```

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

---

## Determinant Utilities

```{autofunction} pyscf.ormas_ci.determinants.build_determinant_list
```

```{autofunction} pyscf.ormas_ci.determinants.count_determinants
```

```{autofunction} pyscf.ormas_ci.determinants.casci_determinant_count
```
