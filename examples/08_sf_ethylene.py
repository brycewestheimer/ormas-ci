#!/usr/bin/env python3
"""SF-ORMAS twisted ethylene excited states.

Demonstrates spin-flip ORMAS-CI on a classic diradical: ethylene at 90-degree
torsion. At this geometry, singlet and triplet are near-degenerate. SF-ORMAS
captures both roots correctly with confirmed spin purity.
"""

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace


def main():
    # 90-degree twisted ethylene geometry (Angstrom)
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.340  0.000  0.000
        H -0.500  0.930  0.000
        H -0.500 -0.930  0.000
        H  1.840  0.000  0.930
        H  1.840  0.000 -0.930
        """,
        basis="6-31g",
        spin=2,
        verbose=0,
    )

    # --- Triplet ROHF reference ---
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    # --- PySCF CASCI reference (2 roots) ---
    mc_ref = mcscf.CASCI(mf, 2, (1, 1))
    mc_ref.verbose = 0
    mc_ref.fcisolver.nroots = 2
    mc_ref.kernel()
    e_casci = mc_ref.e_tot

    # --- SF-ORMAS CAS(2,2), single SF, 2 roots ---
    sf_config = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    )
    mc_sf = mcscf.CASCI(mf, 2, (1, 1))
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    mc_sf.fcisolver.nroots = 2
    mc_sf.kernel()
    e_sf = mc_sf.e_tot

    # --- Analyze spin for each SF-ORMAS root ---
    print("SF-ORMAS Twisted Ethylene (90-degree, 6-31G)")
    print("=" * 70)
    print()
    print("SF-ORMAS roots:")
    print(f"  {'Root':>4}  {'Energy (Ha)':>16}  {'<S^2>':>8}  {'2S+1':>6}  {'State':>10}")
    print("  " + "-" * 56)

    spin_labels = []
    for i in range(2):
        ss, mult = mc_sf.fcisolver.spin_square(mc_sf.ci[i], 2, (1, 1))
        s_val = (-1 + (1 + 4 * ss) ** 0.5) / 2
        label = "singlet" if round(s_val) == 0 else "triplet"
        spin_labels.append(label)
        print(f"  {i:>4}  {e_sf[i]:16.10f}  {ss:8.4f}  {mult:6.2f}  {label:>10}")

    print()

    # --- Compare with PySCF CASCI ---
    print("PySCF CASCI reference:")
    print(f"  {'Root':>4}  {'Energy (Ha)':>16}")
    print("  " + "-" * 26)
    for i in range(2):
        print(f"  {i:>4}  {e_casci[i]:16.10f}")

    print()

    # --- Singlet-triplet gap ---
    singlet_idx = spin_labels.index("singlet")
    triplet_idx = spin_labels.index("triplet")
    gap_ha = e_sf[singlet_idx] - e_sf[triplet_idx]
    gap_kcal = gap_ha * 627.509
    print(f"Singlet-triplet gap: {gap_ha:+.10f} Ha = {gap_kcal:+.4f} kcal/mol")
    print()

    # --- Energy comparison ---
    print("SF-ORMAS vs CASCI energy differences:")
    for i in range(2):
        diff_mh = (e_sf[i] - e_casci[i]) * 1000.0
        print(f"  Root {i}: {diff_mh:+.3e} mHa")


if __name__ == "__main__":
    main()
