#!/usr/bin/env python3
"""SF-ORMAS H2 dissociation curve.

Demonstrates spin-flip ORMAS-CI on the simplest diradical: H2 bond breaking.
Compares RHF, PySCF CASCI, and SF-ORMAS energies across bond lengths.
SF-ORMAS with full CAS bounds reproduces CASCI exactly.
"""

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace


def main():
    # Bond lengths to scan (Angstrom)
    distances = [0.5, 0.74, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Collect results
    results: list[dict[str, float]] = []

    for d in distances:
        atom = f"H 0 0 0; H 0 0 {d}"

        # --- RHF (singlet) ---
        mol_rhf = gto.M(atom=atom, basis="6-31g", verbose=0)
        mf_rhf = scf.RHF(mol_rhf)
        mf_rhf.verbose = 0
        mf_rhf.run()
        e_rhf = mf_rhf.e_tot

        # --- ROHF (triplet reference for SF and CASCI) ---
        mol_trip = gto.M(atom=atom, basis="6-31g", spin=2, verbose=0)
        mf_rohf = scf.ROHF(mol_trip)
        mf_rohf.verbose = 0
        mf_rohf.run()

        # --- PySCF native CASCI(2, (1,1)) ---
        mc_ref = mcscf.CASCI(mf_rohf, 2, (1, 1))
        mc_ref.verbose = 0
        e_casci = mc_ref.kernel()[0]

        # --- SF-ORMAS CAS(2,2), single SF, full CAS bounds ---
        sf_config = SFORMASConfig(
            ref_spin=2,
            target_spin=0,
            n_spin_flips=1,
            n_active_orbitals=2,
            n_active_electrons=2,
            subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        )
        mc_sf = mcscf.CASCI(mf_rohf, 2, (1, 1))
        mc_sf.verbose = 0
        mc_sf.fcisolver = SFORMASFCISolver(sf_config)
        e_sf = mc_sf.kernel()[0]

        diff_mh = (e_sf - e_casci) * 1000.0
        results.append({"d": d, "rhf": e_rhf, "casci": e_casci, "sf": e_sf, "diff": diff_mh})

    # --- Print results ---
    print("SF-ORMAS H2 Dissociation Curve (6-31G)")
    print("=" * 75)
    print(
        f"{'R (A)':>7}  {'E(RHF)':>14}  {'E(CASCI)':>14}  {'E(SF-ORMAS)':>14}  {'diff (mHa)':>10}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['d']:7.2f}  {r['rhf']:14.10f}  {r['casci']:14.10f}"
            f"  {r['sf']:14.10f}  {r['diff']:+10.3e}"
        )
    print("-" * 75)
    print()
    print("Notes:")
    print("  - SF-ORMAS with full CAS bounds reproduces CASCI to machine precision.")
    print("  - RHF fails qualitatively at large bond lengths (Coulson-Fischer point).")
    print("  - SF-ORMAS and CASCI correctly dissociate H2 to two neutral atoms.")


if __name__ == "__main__":
    main()
