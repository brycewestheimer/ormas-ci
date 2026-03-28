# Occupation-Restricted Multiple Active Spaces CI (ORMAS-CI)

## Motivation

RAS is limited to three subspaces (RAS1/RAS2/RAS3) with specific roles:
core-like, fully flexible, and virtual-like. Many systems have correlation
structures that don't fit this pattern. A transition metal complex might
have strongly correlated d-orbitals, a separate set of ligand pi-orbitals,
and sigma bonding orbitals that each need their own treatment.

ORMAS generalizes RAS to an arbitrary number of subspaces, each with its
own independent occupation constraints.

## Definition

The active orbitals are partitioned into K subspaces. Each subspace k has:
- A set of orbital indices (non-overlapping, covering the full active space)
- A minimum total electron occupation: min_k
- A maximum total electron occupation: max_k

A determinant is included in the ORMAS-CI expansion if and only if the
total electron count (alpha + beta) in every subspace simultaneously
satisfies its min/max bounds.

Formally, determinant |D> is included iff for all k = 1, ..., K:
    min_k <= n_alpha_k(D) + n_beta_k(D) <= max_k
where n_alpha_k(D) is the number of alpha electrons in subspace k's orbitals.

## Key Property: Local Constraints

ORMAS uses **local** constraints: each subspace has its own independent
min/max bounds. Whether a determinant is allowed depends on the occupation
of each subspace checked independently. This is in contrast to GAS, which
uses cumulative constraints (see [gas.md](gas.md)).

Local constraints have an important practical consequence: the subspace
problems are structurally independent. The set of allowed determinants
within subspace A does not depend on what's happening in subspace B
(beyond the global electron count constraint). This enables potential
factorization of the CI problem into independent subspace calculations,
which is relevant for quantum computing (see
[quantum_computing.md](quantum_computing.md)).

## Relationship to Other Methods

**CASCI** is ORMAS with a single subspace and no restrictions (min=0, max=2*m).

**RASCI** is ORMAS with exactly three subspaces where RAS1 has a high minimum
(allowing limited holes), RAS2 has min=0 and max=2*m (fully flexible), and
RAS3 has a low maximum (allowing limited particles).

**GAS** is similar to ORMAS but uses cumulative rather than local constraints.
See [comparison.md](comparison.md) for a detailed comparison.

## Example: Transition Metal Complex

Consider an Fe(II) complex with 10 active orbitals:
- 5 metal d-orbitals (orbitals 0-4)
- 5 ligand pi/sigma orbitals (orbitals 5-9)

A CASCI(10,10) calculation has C(10,5)^2 = 63,504 determinants.

An ORMAS partitioning:
- Subspace "metal_d": orbitals [0,1,2,3,4], min=4, max=8
- Subspace "ligand": orbitals [5,6,7,8,9], min=2, max=6

This restricts to configurations where the metal has 4-8 electrons (allowing
up to 2 electrons of charge transfer in either direction) and the ligand has
2-6 electrons. Configurations with extreme charge transfer (e.g., all 10
electrons on the metal) are excluded because they're physically unreasonable.

## Implementation in This Package

ORMAS-CI is the primary method implemented by `ORMASFCISolver`. The user
defines subspaces and constraints through the `ORMASConfig` class. The
solver enumerates all determinants satisfying the constraints, builds the
CI Hamiltonian, and diagonalizes. See [subspace_model.md](../design/subspace_model.md)
for the data model and [determinant_enumeration.md](../architecture/determinant_enumeration.md)
for the enumeration algorithm.

## Key Reference

Ivanic, J. "Direct configuration interaction and multiconfigurational
self-consistent-field method for multiple active spaces with variable
occupations. I. Method." J. Chem. Phys. 2003, 119, 9364.
