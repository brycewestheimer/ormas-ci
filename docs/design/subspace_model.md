# Subspace Configuration Model

## Design Philosophy

The subspace configuration carries all information needed to restrict
the CI expansion. It is defined by the user, validated at construction
time, and consumed by the solver's internal modules. External code
(PySCF, QDK/Chemistry) never sees it.

The model is deliberately simple: dataclasses with validation, no
inheritance hierarchy, no abstract interfaces. The user constructs a
config, passes it to the solver, and doesn't interact with it again.

## Classes

### Subspace

A single orbital subspace with occupation constraints.

```python
@dataclass
class Subspace:
    name: str               # Human-readable label
    orbital_indices: list[int]  # 0-based indices into active space
    min_electrons: int      # Minimum total occupation (alpha + beta)
    max_electrons: int      # Maximum total occupation (alpha + beta)
```

The orbital indices are positions within the active orbital space, not
the full MO space. If PySCF's CASCI has ncas=10 active orbitals, valid
indices are 0 through 9.

The `name` field is for logging and debugging only. It has no
algorithmic effect.

### ORMASConfig

The complete specification for an ORMAS-CI calculation.

```python
@dataclass
class ORMASConfig:
    subspaces: list[Subspace]
    n_active_orbitals: int
    nelecas: tuple[int, int]    # (n_alpha, n_beta)
```

Validation checks:
- Each subspace is internally consistent (min <= max, indices non-negative, etc.)
- Orbital indices are non-overlapping across subspaces
- Orbital indices cover exactly [0, n_active_orbitals)
- The global electron count is feasible given subspace constraints

### RASConfig

A convenience constructor for the traditional 3-subspace RAS.

```python
@dataclass
class RASConfig:
    ras1_orbitals: list[int]
    ras2_orbitals: list[int]
    ras3_orbitals: list[int]
    max_holes_ras1: int
    max_particles_ras3: int
    nelecas: tuple[int, int]
```

`to_ormas_config()` converts to an ORMASConfig by computing the
appropriate min/max electron bounds for each RAS subspace.

## Orbital Index Convention

Orbital indices in the config refer to positions within the active
space, numbered 0 through ncas-1. They do NOT refer to the full MO
space or the AO space.

The mapping from active-space indices to full MO indices is handled
by PySCF's CASCI during integral transformation. Our solver never
needs to know which full MOs the active orbitals correspond to.

This means the user must define subspaces in terms of the active
orbital ordering, which depends on how PySCF selected the active space.
If using PySCF's default orbital selection (lowest-energy HF orbitals),
the active orbitals are ordered by HF orbital energy.

## Why Local Constraints

ORMAS uses local (per-subspace) constraints rather than cumulative
(GAS-style) constraints. This is a deliberate design choice:

1. Local constraints are easier to reason about. Each subspace is
   independent: you can think about subspace A without considering B.

2. Local constraints enable subspace factorization for quantum computing.
   Because subspace A's allowed configurations don't depend on subspace B,
   the subspaces can (in principle) be solved independently on separate
   QPUs.

3. Local constraints are what ORMAS uses. GAS-style cumulative constraints
   could be added as a future extension by modifying the distribution
   enumeration feasibility check.

## Validation Timing

Validation happens at two points:

1. When `config.validate()` is called explicitly (or implicitly by
   `ORMASFCISolver.__init__`). This catches configuration errors early.

2. When `fcisolver.kernel()` is called. This checks consistency between
   the config and PySCF's active space parameters (ncas, nelecas).

Catching errors early (at config creation time) gives the user better
error messages than letting invalid configs propagate to the determinant
enumeration stage.
