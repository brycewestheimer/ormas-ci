#!/usr/bin/env python3
"""Basic ORMAS-CI: partition H2O's active space into subspaces.

Demonstrates the core workflow:
1. Run PySCF RHF + CASCI as a reference
2. Define ORMAS subspaces with occupation constraints
3. Plug ORMASFCISolver into PySCF's CASCI
4. Compare energy accuracy and determinant reduction
"""

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count, count_determinants


def main():
    # Build H2O and run RHF
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVDZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    # Define active space: 6 electrons in 6 orbitals
    ncas, nelecas = 6, (3, 3)

    # --- Full CASCI reference ---
    mc_ref = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_ref.verbose = 0
    e_casci = mc_ref.kernel()[0]
    n_det_casci = casci_determinant_count(ncas, nelecas)

    # --- ORMAS-CI with 3-subspace partitioning ---
    # Partition by orbital character: lone pairs, bonding, antibonding
    config = ORMASConfig(
        subspaces=[
            Subspace("lone_pair", [0, 1], min_electrons=2, max_electrons=4),
            Subspace("bonding", [2, 3], min_electrons=1, max_electrons=3),
            Subspace("antibonding", [4, 5], min_electrons=0, max_electrons=1),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )

    mc_ormas = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_ormas.verbose = 0
    mc_ormas.fcisolver = ORMASFCISolver(config)  # one-line swap
    e_ormas = mc_ormas.kernel()[0]
    n_det_ormas = count_determinants(config)

    # --- Results ---
    print("ORMAS-CI: H2O/cc-pVDZ CAS(6,6)")
    print("=" * 50)
    print(f"CASCI energy:    {e_casci:.10f} Ha  ({n_det_casci} determinants)")
    print(f"ORMAS energy:    {e_ormas:.10f} Ha  ({n_det_ormas} determinants)")
    print(f"Energy error:    {(e_ormas - e_casci) * 1000:+.3f} mHa")
    print(f"Det reduction:   {n_det_casci}/{n_det_ormas} = {n_det_casci / n_det_ormas:.1f}x")
    print(f"Variational:     {'PASS' if e_ormas >= e_casci - 1e-10 else 'FAIL'}")
    print()
    print("Subspace partitioning:")
    for sub in config.subspaces:
        print(
            f"  {sub.name:>12}: orbitals {sub.orbital_indices}, "
            f"electrons [{sub.min_electrons}, {sub.max_electrons}]"
        )


if __name__ == "__main__":
    main()
