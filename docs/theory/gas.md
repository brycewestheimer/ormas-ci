# Generalized Active Space CI (GAS-CI)

## Definition

GAS divides the active orbitals into K subspaces, similar to ORMAS. The
distinction is in how the occupation constraints are specified.

In ORMAS, each subspace has its own independent (local) min/max electron count.

In GAS, the constraints are **cumulative**: you specify the min/max total
electron count for the first subspace, then for the first two subspaces
combined, then for the first three combined, and so on.

Formally, determinant |D> is included iff for all k = 1, ..., K:
    cum_min_k <= sum_{j=1}^{k} n_j(D) <= cum_max_k
where n_j(D) is the total electron count in subspace j.

## Comparison with ORMAS

ORMAS and GAS generate different CI expansions for the same orbital
partitioning. The difference is subtle but matters in practice.

Consider 2 subspaces (A, B) with 6 total electrons. Suppose we want to
allow at most 1 electron of charge transfer between the subspaces.

**ORMAS formulation:**
- Subspace A: min=2, max=4
- Subspace B: min=2, max=4
This allows distributions (2,4), (3,3), (4,2): at most 1 electron
transferred relative to the (3,3) default. But it also allows certain
internal rearrangements that don't affect the total.

**GAS formulation:**
- Cumulative through A: min=2, max=4
- Cumulative through A+B: min=6, max=6 (total is fixed at 6)
This also constrains A to have 2-4 electrons, but the cumulative
structure means the constraint on B is implicit (B = 6 - A).

For two subspaces, the formulations are often equivalent. For three or
more subspaces, they diverge because cumulative constraints couple the
subspaces: the allowed occupation of subspace C depends on what happened
in subspaces A and B.

## Implications for Quantum Computing

ORMAS's local constraints are generally more useful for quantum circuit
construction because each subspace can be treated independently. GAS's
cumulative constraints couple the subspaces, making it harder to factor
the problem into independent QPU calculations.

GAS is more flexible for controlling the total number of inter-subspace
excitations globally (e.g., "at most 2 electrons can leave the first 3
subspaces combined"), which ORMAS can't express directly.

## Implementation Status

This package currently implements ORMAS-style local constraints only.
GAS support (cumulative constraints) is listed as a future direction.
The determinant enumeration algorithm could be extended to support
cumulative constraints by modifying the feasibility check in the
recursive distribution enumeration.

## Key References

Ma, D.; Li Manni, G.; Gagliardi, L. "The generalized active space
concept in multiconfigurational self-consistent field methods."
J. Chem. Phys. 2011, 135, 044128.

Olsen, J. (1988). Original GAS formulation (unpublished, referenced
in Ma et al. 2011).
