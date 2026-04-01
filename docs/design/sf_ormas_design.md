# SF-ORMAS Design

## Design Goal

Extend ORMAS-CI with spin-flip capability by modifying only the front end
(configuration + determinant enumeration). The core computational engine
(Hamiltonian, eigensolver, sigma vectors, RDMs) must remain unchanged and
be reused as-is.

## Architecture Impact

SF-ORMAS is primarily a front-end extension. The core engine is reused unchanged.

**Modified modules:**

| Module | Change |
|--------|--------|
| `subspaces.py` | New `SFORMASConfig` and `SFRASConfig` dataclasses |
| `fcisolver.py` | New `SFORMASFCISolver` subclass |
| `__init__.py` | Export new public symbols |

**New module:**

| Module | Purpose |
|--------|---------|
| `spinflip.py` | SF determinant enumeration, reference analysis, config translation |

**Unchanged modules** (all operate on generic determinant lists or integrals):
`utils.py`, `slater_condon.py`, `hamiltonian.py`, `solver.py`, `davidson.py`,
`sigma.py`, `rdm.py`.

## Configuration Model

**`SFORMASConfig`** is the main config class. Key fields:

- `ref_spin` (int): 2S of ROHF reference (e.g., 2 for triplet)
- `target_spin` (int): 2S of target CI state (e.g., 0 for singlet)
- `n_spin_flips` (int): Number of alpha→beta flips
- `n_active_orbitals`, `n_active_electrons`: Active space size
- `subspaces`: List of `Subspace` objects (same as standard ORMAS)

Key methods: `nelecas_reference` → `(N_alpha_ref, N_beta_ref)` for ROHF,
`nelecas_target` → `(N_alpha_target, N_beta_target)` for CI,
`to_ormas_config()` → standard `ORMASConfig` in target M_s sector.

**`SFRASConfig`** is a convenience for 3-subspace RAS-style SF. Mirrors
`RASConfig` with added spin-flip parameters. Has `to_sf_ormas_config()`.

**`single_sf_diradical()`** is a convenience constructor for the common
case: triplet→singlet single SF. Usage example:

```python
sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("pi", [0, 1], 0, 4)],
)
ormas_config = sf_config.to_ormas_config()  # Standard ORMAS in M_s=0 sector
```

## Determinant Enumeration Strategy

The key implementation insight: SF-ORMAS determinant enumeration reduces
to standard ORMAS enumeration in the target M_s sector.

`spinflip.py` does NOT reimplement enumeration. It:

1. Takes an `SFORMASConfig`
2. Calls `config.to_ormas_config()` for target-sector electron counts
3. Passes that to existing `build_determinant_list()`
4. Returns the result

Additional responsibilities: `validate_reference_consistency()` checks ROHF
occupation vs. subspace assignments, `build_reference_determinant()` constructs
reference bit strings, and `count_sf_determinants()` counts without full
enumeration.

## PySCF Integration

`SFORMASFCISolver` inherits from `ORMASFCISolver`. See
{doc}`pyscf_integration` for the base integration contract.

- Constructor takes `SFORMASConfig`, translates to `ORMASConfig` via
  `to_ormas_config()`
- Stores SF metadata (ref_spin, target_spin, n_spin_flips, nelecas_ref,
  nelecas_target)
- Overrides `kernel()` to add SF-specific validation and logging
- Validates that PySCF's nelecas matches target sector
- Delegates to parent `kernel()` for actual solve

User workflow:

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='sto-3g', spin=2)
mf = scf.ROHF(mol).run()

sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("sigma", [0, 1], 0, 4)],
)

n_a, n_b = sf_config.nelecas_target
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
mc.kernel()
```

**ROHF → CASCI orbital handling:** PySCF's CASCI extracts MO coefficients
from the SCF object. When `mf` is ROHF with `spin=2S_ref` and we pass
`nelecas=(N_alpha_target, N_beta_target)`, PySCF doesn't care about the
mismatch — it uses ROHF orbitals as-is and does CI in the target M_s
sector. This is exactly correct.

## CASSCF Compatibility

SF solver works with CASSCF orbital optimization:

```python
mc = mcscf.CASSCF(mf, ncas, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
mc.kernel()
```

Works because `SFORMASFCISolver` produces valid RDMs (inherited from
parent).

## Multi-Root Support

Inherited unchanged from parent. Critical for excited states:

```python
mc.fcisolver.nroots = 4
mc.kernel()
for i, (e, ci) in enumerate(zip(mc.e_tot, mc.ci)):
    ss, mult = mc.fcisolver.spin_square(ci, ncas, (n_a, n_b))
    print(f"Root {i}: E={e:.8f}, <S^2>={ss:.4f}, 2S+1={mult:.2f}")
```

Spin diagnostic is essential — the M_s=0 sector contains states of
multiple multiplicities.

## Dependency Graph

Module dependency graph with `spinflip.py` added:

```
subspaces.py  [+SFORMASConfig]
    ↓
determinants.py
    ↓
spinflip.py  [NEW: SF enumeration + reference analysis]
    ↓
fcisolver.py  [+SFORMASFCISolver]
```

## Known Limitations

- No state-averaged CASCI with SF (inherited limitation).
- No analytical gradients (inherited).
- UHF references not supported (by design — ROHF required).
