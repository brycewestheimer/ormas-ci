# Restricted Active Space Configuration Interaction (RASCI)

## Motivation

CASCI becomes intractable for large active spaces because every possible
electron configuration is included. In many systems, however, only a subset
of the active orbitals participate in strong correlation. The remaining
orbitals are "mostly occupied" or "mostly empty" and only need limited
excitations to capture their contribution.

RAS exploits this by dividing the active space into three subspaces with
occupation restrictions, keeping full flexibility where it matters while
cutting the combinatorial cost of the periphery.

## The Three Subspaces

**RAS1:** Contains orbitals that are mostly doubly occupied in the reference.
Think of these as "almost core" orbitals that participate in some correlation
but rarely have more than a few electrons removed. A maximum number of holes
(missing electrons relative to full occupation) is allowed.

**RAS2:** The fully flexible subspace. All occupations from 0 to 2 electrons
per orbital are allowed with no restrictions. This is equivalent to a small
CAS within the larger active space and should contain the orbitals with the
strongest correlation effects.

**RAS3:** Contains orbitals that are mostly empty in the reference. These are
"almost virtual" orbitals that accept limited excitations. A maximum number
of particles (electrons present) is allowed.

## How Restrictions Work

A determinant is included in the RASCI expansion if and only if:
1. The number of holes in RAS1 does not exceed max_holes
2. RAS2 has no restrictions
3. The number of particles in RAS3 does not exceed max_particles

"Holes in RAS1" means the number of electrons missing relative to full
occupation (2 * n_ras1_orbitals). If RAS1 has 4 orbitals (normally 8
electrons) and max_holes=2, then RAS1 must have at least 6 electrons.

"Particles in RAS3" is simply the number of electrons present, since
the default occupation is 0.

## Determinant Count Reduction

Consider CAS(14,14) which has about 11.8 million determinants.
RAS with 4 orbitals in RAS1, 6 in RAS2, 4 in RAS3, max_holes=2,
max_particles=2 might have only 500,000 determinants: a 20x reduction.

The reduction comes from excluding configurations with 3+ holes in RAS1
or 3+ particles in RAS3. These correspond to high-order excitations from
the core or into the virtual space, which are usually chemically unimportant.

## Convergence to CASCI

As max_holes and max_particles increase, the RASCI expansion approaches
full CASCI:
- max_holes=0, max_particles=0: only the RAS2 orbitals are flexible (a smaller CAS)
- max_holes=1, max_particles=1: adds single excitations from RAS1 and into RAS3
- max_holes=2, max_particles=2: adds double excitations
- max_holes=n_ras1*2, max_particles=n_ras3*2: equivalent to full CASCI

RASCI is variational: the energy is always at or above the CASCI energy for
the same active space, since RASCI uses a subset of the CASCI determinants.

## Relationship to ORMAS

RAS is a special case of ORMAS with exactly three subspaces and specific
constraint semantics. In this package, RAS configurations are specified
through the `RASConfig` convenience class, which internally converts to
an `ORMASConfig` with appropriate min/max occupation bounds. See
[ormas.md](ormas.md) for the generalization.

## Key Reference

Malmqvist, P.A.; Rendell, A.; Roos, B.O. "The restricted active space
self-consistent-field method, implemented with a split graph unitary group
approach." J. Phys. Chem. 1990, 94, 5477.
