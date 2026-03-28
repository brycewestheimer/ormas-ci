# Complete Active Space Configuration Interaction (CASCI)

## The Electronic Structure Problem

The central problem in quantum chemistry is solving the electronic Schrodinger
equation for a molecular system. The exact solution within a given basis set
is Full Configuration Interaction (FCI): distribute all N electrons among all
M molecular orbitals in every possible way, build the Hamiltonian matrix in that
determinant basis, and diagonalize. The result is exact for that basis set.

FCI is intractable for all but the smallest systems. The number of Slater
determinants grows combinatorially: C(M, N_alpha) * C(M, N_beta). For a
modest molecule in a decent basis set, this reaches billions.

## The Active Space Approximation

The key insight: not all orbitals contribute equally to electron correlation.
Core orbitals (deep inner shells) are always doubly occupied. High-lying
virtual orbitals are always empty. The chemically interesting action happens
in a small set of frontier orbitals near the HOMO-LUMO gap.

CASCI exploits this by dividing the molecular orbitals into three groups:

**Inactive (frozen core):** Always doubly occupied. These contribute a constant
core energy E_core and are removed from the CI problem.

**Active:** The orbitals where correlation matters. The active electrons are
distributed among active orbitals in all possible ways. This is where the
CI expansion lives.

**External (frozen virtual):** Always empty. Excluded from the CI. Can be
recovered later through perturbation theory (CASPT2, NEVPT2).

The active space straddles the HF occupied-virtual boundary: it includes
the highest occupied and lowest virtual orbitals from the HF reference.

## The CASCI Expansion

For n_active electrons in m_active orbitals, CASCI generates all determinants
distributing those electrons among those orbitals. The determinant count is:

    N_det = C(m, n_alpha) * C(m, n_beta)

For CAS(10,10) (10 electrons in 10 orbitals with 5 alpha and 5 beta), this
is C(10,5)^2 = 63,504 determinants. For CAS(16,16), about 166 million. For
CAS(20,20), about 34 billion.

The Hamiltonian matrix is built using the Slater-Condon rules (see
[slater_condon.md](slater_condon.md)) and diagonalized to obtain the ground
state energy and wavefunction.

## CASCI vs CASSCF

**CASCI** uses fixed orbitals (from HF or another prior calculation). The
CI coefficients are optimized but the orbitals are not.

**CASSCF** optimizes both the CI coefficients and the orbital shapes
simultaneously in a self-consistent loop. This captures orbital relaxation
due to correlation and generally gives better results, but is more expensive
and can have convergence issues.

This package implements CI-level methods (fixed orbitals), consistent with
QDK/Chemistry's approach. Orbital optimization can be handled externally
by PySCF's CASSCF if desired.

## Limitations

The exponential scaling of determinant count with active space size is the
fundamental limitation. CAS(14,14) is roughly the practical limit for exact
CASCI on a classical computer. Many chemically important systems require
larger active spaces, which motivates the restricted active space methods
described in the companion documents.

## Notation

CAS(n, m) denotes n active electrons in m active orbitals. Some references
use CAS(m, n) with reversed order; this package follows the (electrons, orbitals)
convention matching PySCF's `mcscf.CASCI(mf, ncas, nelecas)` interface.
