# Comparison of Active Space CI Methods

## Method Summary

| Method   | Subspaces | Constraint Type | Orbital Optimization | Key Use Case |
|----------|-----------|-----------------|----------------------|--------------|
| CASCI    | 1         | None            | No                   | Small active spaces, reference calculations |
| CASSCF   | 1         | None            | Yes                  | Accurate multiconfigurational wavefunctions |
| RASCI    | 3 (fixed) | Local (holes/particles) | No          | Large active spaces with core/virtual periphery |
| RASSCF   | 3 (fixed) | Local (holes/particles) | Yes         | Same as RASCI with orbital relaxation |
| ORMAS-CI | Arbitrary | Local (per-subspace min/max) | No     | Multiple distinct correlation domains |
| GAS-CI   | Arbitrary | Cumulative (accumulated min/max) | No | Controlling total inter-subspace excitation |
| ASCI     | 1         | Adaptive (numerical selection) | No   | Large active spaces, automatic truncation |

## What This Package Implements

ORMAS-CI (with RASCI as a special case). Fixed orbitals, local per-subspace
constraints, arbitrary number of subspaces. CASCI is recovered as a single
unrestricted subspace.

## ORMAS vs Selected CI (ASCI)

These solve the same problem (CASCI is too big) through fundamentally
different strategies:

**ORMAS:** Determinants are selected by structural rules (which orbitals,
how many electrons). The user defines subspaces based on chemical intuition.
The determinant space has regular structure that can be predicted before
running the calculation.

**ASCI:** Determinants are selected by numerical importance (which
configurations have large CI coefficients). The algorithm discovers the
important determinants adaptively. The determinant space has no regular
structure; it depends on the specific wavefunction.

**Tradeoffs:**
- ASCI is typically more accurate per determinant (it keeps exactly the
  important ones). ORMAS may include some unimportant determinants and
  exclude some important ones if the subspace boundaries don't perfectly
  align with the correlation structure.
- ORMAS gives the user explicit control and produces a predictable
  determinant space. ASCI is automatic but its output depends on
  convergence behavior.
- ORMAS's structured determinant space maps to quantum circuit ansatze.
  ASCI's irregular space does not. See
  [quantum_computing.md](quantum_computing.md) for details.
- ORMAS enables subspace factorization for quantum computing. ASCI does not.

## ORMAS vs GAS

Both partition orbitals into arbitrary subspaces with occupation constraints.
The difference is local (ORMAS) vs cumulative (GAS) constraints.

**When they give the same result:** For two subspaces with a fixed total
electron count, ORMAS and GAS are equivalent (the constraint on one
subspace fully determines the other).

**When they differ:** For three or more subspaces, cumulative constraints
couple the subspaces. In GAS, the allowed occupation of subspace C
depends on the combined occupation of A and B. In ORMAS, subspace C's
constraints are independent.

**Which is better for quantum computing:** ORMAS, generally. Its local
constraints enable independent treatment of subspaces, which is the
basis for subspace factorization (sequential/parallel QPU evaluation).
GAS's coupling makes factorization harder.

**Which is more flexible:** GAS can express constraints that ORMAS cannot,
such as "at most 2 electrons total can move between any subspaces."
ORMAS can express some constraints that GAS cannot cleanly handle,
particularly when subspaces have very different character.

## Variational Hierarchy

For the same active space, the ground state energies satisfy:

    E(CASCI) <= E(RASCI)
    E(CASCI) <= E(ORMAS)

Any restriction on the determinant space raises the energy relative
to the unrestricted CASCI (variational principle). Additionally,
E(restricted) <= E(HF) holds when the HF reference determinant is
included in the restricted determinant space, which is the typical
case but not guaranteed for all ORMAS/RASCI configurations. The
quality of the approximation depends on whether the excluded
determinants were important for the target property.
