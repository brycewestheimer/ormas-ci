# Migration Guide

This page helps users coming from other ORMAS/RAS implementations
translate their existing workflows to ORMAS-CI.

## From PySCF Native CASCI

If you are already using PySCF's CASCI, switching to ORMAS-CI requires
two additions: defining a configuration and setting the solver.

**Before (PySCF native):**

```python
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.kernel()
```

**After (ORMAS-CI):**

```python
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

config = ORMASConfig(
    subspaces=[
        Subspace("group_a", [0, 1, 2], min_electrons=2, max_electrons=5),
        Subspace("group_b", [3, 4, 5], min_electrons=1, max_electrons=4),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
mc.kernel()
```

Everything else (molecule setup, SCF, result access via `mc.e_tot`,
`mc.ci`, `mc.analyze()`) remains identical.

## From GAMESS ORMAS

GAMESS specifies ORMAS via `$ORMAS` input group keywords. Here is
the mapping:

| GAMESS | ORMAS-CI Equivalent |
|--------|-------------------|
| `NSPACE=N` | Number of `Subspace` objects in `ORMASConfig.subspaces` |
| `MSTART(1)=...` / `MEND(1)=...` | `Subspace.orbital_indices` (0-indexed) |
| `MINE(1)=...` | `Subspace.min_electrons` |
| `MAXE(1)=...` | `Subspace.max_electrons` |

**Key differences:**

- **Orbital indexing:** GAMESS uses 1-based indices into the full MO
  space. ORMAS-CI uses 0-based indices into the active space only. If
  your GAMESS active space starts at MO 5, then GAMESS orbital 5 maps
  to ORMAS-CI index 0.

- **Electron specification:** GAMESS specifies total active electrons.
  ORMAS-CI specifies `(n_alpha, n_beta)` separately. For a closed-shell
  system with 6 active electrons, use `nelecas=(3, 3)`.

- **Solver selection:** GAMESS uses its own CI solver. ORMAS-CI uses
  PySCF as the computational framework. Results should be compared at
  the same basis set and geometry.

## From Forte / Psi4

Forte supports RASCI, GASCI, and ORMAS-CI through Psi4. The constraint
model is similar but the interface differs.

| Forte | ORMAS-CI Equivalent |
|-------|-------------------|
| `GAS1 [0,1,2]` | `Subspace("gas1", [0,1,2], min_e, max_e)` |
| `GAS_MIN [2,1]` | `min_electrons` on each `Subspace` |
| `GAS_MAX [4,3]` | `max_electrons` on each `Subspace` |

**Key differences:**

- **Constraint model:** Forte supports both local (ORMAS) and cumulative
  (GAS) constraints. ORMAS-CI currently supports only local constraints.
  See [ORMAS vs GAS](../theory/gas.md).

- **Integration target:** Forte integrates with Psi4. ORMAS-CI integrates
  with PySCF and QDK/Chemistry.

- **Orbital indices:** Forte uses Psi4's orbital ordering. ORMAS-CI uses
  active-space-relative indices. See
  [Subspace Model](../design/subspace_model.md) for the indexing
  convention.

For a detailed comparison, see
[Forte/Psi4 Comparison](../integration/forte_comparison.md).

## Notation Reference

| Concept | ORMAS-CI | GAMESS | Forte/Psi4 |
|---------|----------|--------|------------|
| Active orbitals | `n_active_orbitals` | `NACT` | `ACTIVE` |
| Active electrons | `nelecas=(n_a, n_b)` | `NELT` | `ACTIVE_ELECTRONS` |
| Orbital indices | 0-based, active-space-relative | 1-based, full-MO | 0-based, Psi4 ordering |
| Subspace count | `len(config.subspaces)` | `NSPACE` | Number of `GASn` entries |
| Min occupation | `Subspace.min_electrons` | `MINE(i)` | `GAS_MIN[i]` |
| Max occupation | `Subspace.max_electrons` | `MAXE(i)` | `GAS_MAX[i]` |
