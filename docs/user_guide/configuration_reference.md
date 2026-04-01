# Configuration Reference

This page describes all configuration classes and solver tuning
attributes. For the full autodoc API, see the
[API Reference](../api/public_api.md).

## Subspace

A single group of active orbitals with occupation bounds.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable label (for logging only) |
| `orbital_indices` | `list[int]` | 0-based indices into the active orbital space |
| `min_electrons` | `int` | Minimum total electrons (alpha + beta) in this subspace |
| `max_electrons` | `int` | Maximum total electrons (alpha + beta) in this subspace |

Orbital indices refer to positions within the active space (0 to
ncas-1), not the full MO space. See
[Subspace Model](../design/subspace_model.md) for details on this
convention.

## ORMASConfig

The main configuration for an ORMAS-CI calculation.

| Field | Type | Description |
|-------|------|-------------|
| `subspaces` | `list[Subspace]` | Orbital subspaces with occupation constraints |
| `n_active_orbitals` | `int` | Total number of active orbitals (must equal `ncas` in PySCF) |
| `nelecas` | `tuple[int, int]` | Active electrons as `(n_alpha, n_beta)` |

Validation checks run automatically when the config is passed to the
solver. Orbital indices must be non-overlapping, cover exactly
`[0, n_active_orbitals)`, and the electron count must be feasible given
the subspace bounds.

**Convenience method:** `ORMASConfig.unrestricted(ncas, nelecas)` creates
a single-subspace config equivalent to full CASCI (useful for reference
comparisons).

## RASConfig

Convenience wrapper for classic 3-subspace RASCI.

| Field | Type | Description |
|-------|------|-------------|
| `ras1_orbitals` | `list[int]` | Mostly-occupied orbitals (restricted holes) |
| `ras2_orbitals` | `list[int]` | Full-CI orbitals (unrestricted) |
| `ras3_orbitals` | `list[int]` | Mostly-empty orbitals (restricted particles) |
| `max_holes_ras1` | `int` | Maximum holes allowed in RAS1 |
| `max_particles_ras3` | `int` | Maximum particles allowed in RAS3 |
| `nelecas` | `tuple[int, int]` | Active electrons as `(n_alpha, n_beta)` |

Call `ras.to_ormas_config()` to convert to an `ORMASConfig` before
passing to the solver.

## SFORMASConfig

Configuration for spin-flip ORMAS-CI calculations.

| Field | Type | Description |
|-------|------|-------------|
| `ref_spin` | `int` | Reference state spin (2S, e.g., 2 for triplet) |
| `target_spin` | `int` | Target state spin (2S, e.g., 0 for singlet) |
| `n_spin_flips` | `int` | Number of spin flips (typically 1) |
| `n_active_orbitals` | `int` | Total active orbitals |
| `n_active_electrons` | `int` | Total active electrons |
| `subspaces` | `list[Subspace]` | Subspace definitions |
| `max_hole_excitations` | `int` (optional) | Max holes for SF-RAS-style restrictions |
| `max_particle_excitations` | `int` (optional) | Max particles for SF-RAS-style restrictions |

**Convenience method:** `SFORMASConfig.single_sf_diradical(...)` creates
a standard 1-3 subspace single-SF configuration.

Use `sf_config.nelecas_target` to get the `(n_alpha, n_beta)` tuple for
PySCF's CASCI setup.

## SFRASConfig

Convenience wrapper for 3-subspace SF-RASCI. Converts to `SFORMASConfig`
via `to_sf_ormas_config()`.

## Solver Tuning Attributes

These attributes can be set on `ORMASFCISolver` (or `SFORMASFCISolver`)
after construction to control solver behavior.

| Attribute | Default | When to Change |
|-----------|---------|----------------|
| `direct_ci_threshold` | 200 | Increase if you have memory to spare and want exact diagonalization for larger spaces. Dense path stores an n_det x n_det matrix. |
| `einsum_string_threshold` | 300 | Increase if the Davidson+einsum path converges well for your system. Decrease if you observe memory issues from the dense excitation matrices. |
| `conv_tol` | 1e-12 | Loosen (e.g., 1e-10) for faster iterative solves when high precision is not needed. Tighten for near-degenerate roots. |
| `max_cycle` | 100 | Increase if the Davidson solver reports non-convergence. |
| `max_space` | 12 | Increase for better convergence on difficult systems (costs more memory per Davidson iteration). |
| `level_shift` | 0.001 | Increase if Davidson convergence is oscillatory. |
| `lindep` | 1e-14 | Rarely needs adjustment. |
| `nroots` | 1 | Set > 1 for multi-root calculations (excited states). |

```python
solver = ORMASFCISolver(config)
solver.nroots = 3              # compute 3 roots
solver.conv_tol = 1e-10        # slightly looser convergence
solver.direct_ci_threshold = 500  # use dense path up to 500 dets
mc.fcisolver = solver
```

For details on how the solver selects its internal path, see
[Performance Considerations](../design/performance.md).
