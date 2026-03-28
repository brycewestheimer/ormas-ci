# Integral Conventions

## Notation

This package uses **chemists' notation** for two-electron integrals
throughout, matching PySCF's internal convention:

    h2e[p,q,r,s] = (pq|rs)
                  = integral dr1 dr2 phi_p(r1) phi_q(r1) (1/r12) phi_r(r2) phi_s(r2)

The two electrons are indexed by (r1, r2). Orbitals p,q share coordinate
r1; orbitals r,s share coordinate r2. The 1/r12 is the electron-electron
repulsion operator.

## Chemists' vs Physicists' Notation

Some references and codes use physicists' notation:

    <pq|rs> = integral dr1 dr2 phi_p(r1) phi_q(r2) (1/r12) phi_r(r1) phi_s(r2)

The relationship is:

    (pq|rs)_chemists = <pr|qs>_physicists

Mixing notations is a common source of bugs in CI code. This package
exclusively uses chemists' notation. If consulting references that use
physicists' notation, convert before applying formulas.

## Common Integral Combinations

**Coulomb integral:**
    J[p,q] = (pp|qq) = h2e[p,p,q,q]

This is the classical electrostatic repulsion between an electron in
orbital p and an electron in orbital q.

**Exchange integral:**
    K[p,q] = (pq|qp) = h2e[p,q,q,p]

This is a purely quantum mechanical term arising from the antisymmetry
of the electronic wavefunction. It only appears between same-spin
electron pairs.

**Coulomb-exchange combination (same spin):**
    J[p,q] - K[p,q] = h2e[p,p,q,q] - h2e[p,q,q,p]

This appears in the diagonal Hamiltonian element for same-spin pairs
and in the summation terms of single excitation elements.

## How PySCF Passes Integrals

PySCF's CASCI calls `fcisolver.kernel(h1e, eri, ncas, nelecas)` where:

**h1e** is always a full (ncas, ncas) array. These are the one-electron
integrals (kinetic + nuclear attraction + frozen core Fock contribution)
in the active-space MO basis.

**eri** can arrive in multiple formats:
- Full 4-index: (ncas, ncas, ncas, ncas). Ready to use directly.
- Compressed 2D: (ncas*(ncas+1)/2, ncas*(ncas+1)/2). Uses the 4-fold
  symmetry (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq) to store only unique
  elements. Needs `ao2mo.restore(1, eri, ncas)` to expand.
- Compressed 1D: flattened version of the 2D form. Same restore call.

Our fcisolver.kernel() detects the format by checking `eri.ndim` and
converts to full 4-index before passing to the Hamiltonian builder.

## Integral Symmetries

The chemists' notation integrals have 4-fold symmetry for real orbitals:

    (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr)

And the additional particle exchange symmetry:

    (pq|rs) = (rs|pq)

Together these give 8-fold symmetry. PySCF's compressed formats exploit
these symmetries. Our Hamiltonian builder uses the full 4-index tensor
for simplicity, at the cost of storing redundant elements.
