#!/usr/bin/env python3
"""Systematic constraint relaxation: accuracy vs determinant count tradeoff.

Shows how to find the optimal ORMAS constraints by progressively
relaxing occupation bounds and tracking the energy error.

System: N2 with sigma/pi symmetry partitioning
  - sigma orbitals: indices [0, 5] (bonding + antibonding sigma)
  - pi orbitals:    indices [1, 2, 3, 4] (bonding + antibonding pi)

At equilibrium, the sigma pair holds ~2 electrons and the pi system
holds ~4. We systematically allow more charge transfer between them.
"""

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count, count_determinants


def main():
    # N2 at equilibrium
    mol = gto.M(atom="N 0 0 0; N 0 0 1.098", basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 6, (3, 3)

    # Orbital ordering from PySCF CASCI:
    #   0: sigma_g (bonding sigma)
    #   1: pi_y    (bonding pi)
    #   2: pi_x    (bonding pi)
    #   3: pi_x*   (antibonding pi)
    #   4: pi_y*   (antibonding pi)
    #   5: sigma_u* (antibonding sigma)

    # Full CASCI reference
    mc_ref = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_ref.verbose = 0
    e_casci = mc_ref.kernel()[0]
    n_det_casci = casci_determinant_count(ncas, nelecas)

    # Sweep: allow 0, 1, 2, 3 electrons to transfer between sigma and pi
    # Nominal: sigma has 2 electrons, pi has 4
    print("Convergence Study: N2/cc-pVDZ CAS(6,6) sigma/pi split")
    print("=" * 70)
    print(f"CASCI reference: E = {e_casci:.8f} Ha, {n_det_casci} determinants")
    print()
    print(
        f"{'Allowance':>9} {'sigma e- range':>15} {'pi e- range':>13} "
        f"{'n_det':>6} {'Reduction':>9} {'dE (mHa)':>10}"
    )
    print("-" * 70)

    for allowance in range(4):
        # sigma: nominal 2 electrons, allow +/- allowance
        sig_min = max(0, 2 - allowance)
        sig_max = min(4, 2 + allowance)  # 2 orbitals -> max 4 electrons
        # pi: nominal 4 electrons, allow +/- allowance
        pi_min = max(0, 4 - allowance)
        pi_max = min(8, 4 + allowance)  # 4 orbitals -> max 8 electrons

        config = ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 5], min_electrons=sig_min, max_electrons=sig_max),
                Subspace("pi", [1, 2, 3, 4], min_electrons=pi_min, max_electrons=pi_max),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )

        mc = mcscf.CASCI(mf, ncas, sum(nelecas))
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e = mc.kernel()[0]
        n_det = count_determinants(config)
        de = (e - e_casci) * 1000
        reduction = f"{n_det_casci / n_det:.1f}x" if n_det < n_det_casci else "-"

        print(
            f"{allowance:>9} {f'[{sig_min}, {sig_max}]':>15} "
            f"{f'[{pi_min}, {pi_max}]':>13} {n_det:>6} {reduction:>9} {de:>+10.3f}"
        )

    print()
    print("Key insight: allowance=1 gives most of the accuracy recovery")
    print("with significant determinant reduction. Diminishing returns")
    print("for larger allowances -- a common pattern in ORMAS.")


if __name__ == "__main__":
    main()
