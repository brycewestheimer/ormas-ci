# Slater-Condon Rules

## Overview

The Slater-Condon rules define how to compute Hamiltonian matrix elements
between Slater determinants. They are the fundamental building block of
any determinant-based CI code, including this one.

Given two determinants |I> and |J> and the molecular Hamiltonian H, the
matrix element <I|H|J> depends on how many spin-orbitals differ between
the two determinants.

## The Electronic Hamiltonian

In second quantization, the electronic Hamiltonian is:

    H = sum_{pq} h1e[p,q] a+_p a_q + (1/2) sum_{pqrs} h2e[p,q,r,s] a+_p a+_r a_s a_q

where h1e[p,q] are one-electron integrals (kinetic energy + nuclear attraction)
and h2e[p,q,r,s] = (pq|rs) are two-electron integrals in chemists' notation.

This package uses chemists' notation throughout, matching PySCF's convention.

## The Three Cases

### Zero differences (diagonal: I = J)

    <I|H|I> = sum_p h1e[p,p] * n_p
            + sum_{p>q, same spin} [h2e[p,p,q,q] - h2e[p,q,q,p]]
            + sum_{p(alpha), q(beta)} h2e[p,p,q,q]

The first term sums orbital energies over all occupied orbitals. The
second adds Coulomb minus exchange for same-spin occupied pairs. The
third adds Coulomb only for opposite-spin pairs (no exchange for
opposite-spin electrons).

### One spin-orbital difference (single excitation)

If |J> is obtained from |I> by replacing orbital p with orbital q
in spin channel sigma:

    <I|H|J> = sign * [ h1e[p,q]
                      + sum_{r in occ_sigma, r!=p} (h2e[p,q,r,r] - h2e[p,r,r,q])
                      + sum_{r in occ_other_sigma} h2e[p,q,r,r] ]

The sign is the fermionic phase factor (see Phase Factors below).

### Two spin-orbital differences (double excitation)

**Same-spin double (two alpha or two beta orbitals differ):**
If p,q -> r,s in the same spin channel:

    <I|H|J> = sign * [h2e[p,r,q,s] - h2e[p,s,q,r]]

**Opposite-spin double (one alpha and one beta orbital differ):**
If alpha p -> r and beta q -> s:

    <I|H|J> = sign_alpha * sign_beta * h2e[p,r,q,s]

No exchange term for opposite-spin doubles.

### Three or more differences

    <I|H|J> = 0

The Hamiltonian contains at most two-body operators, so it can connect
determinants that differ by at most two spin-orbitals.

## Phase Factors

The fermionic sign arises from anticommutation of creation/annihilation
operators. When moving an electron from orbital q to orbital p, the sign is:

    sign = (-1)^n

where n is the number of occupied orbitals (in the same spin channel)
strictly between positions min(p,q) and max(p,q).

For single excitations, the phase is computed directly from the original
occupation string.

For same-spin double excitations, the phase is the product of two single
excitation phases. The second phase must be computed using the intermediate
determinant (after the first excitation has been applied), not the original
determinant. See [phase_factors.md](../architecture/phase_factors.md) for
implementation details.

For opposite-spin double excitations, the alpha and beta phases are
independent because they operate on separate occupation strings.

## Implementation

The implementation in `slater_condon.py` dispatches based on the
excitation level (0, 1, or 2 differences), computes the appropriate
integral contributions, and applies the phase factor. The top-level
function `matrix_element()` handles the dispatch.
