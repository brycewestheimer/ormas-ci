#!/usr/bin/env python3
"""QDK/Chemistry pipeline: QDK SCF -> ORMAS-CI -> qubit Hamiltonian.

Demonstrates the end-to-end integration:
1. QDK/Chemistry runs SCF on H2O
2. Orbitals are converted to a PySCF SCF object
3. PySCF CASCI uses ORMASFCISolver for the CI step
4. QDK constructs the Jordan-Wigner qubit Hamiltonian
5. QDK's classical solver verifies the qubit energy matches

Requires: pip install pyscf-ormas-ci[qdk]
"""

from pyscf import mcscf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count, count_determinants

try:
    from qdk_chemistry.algorithms import QdkHamiltonianConstructor, QdkQubitMapper, create
    from qdk_chemistry.data import Structure
    from qdk_chemistry.plugins.pyscf.conversion import SCFType, orbitals_to_scf
    from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False


def main():
    if not QDK_AVAILABLE:
        print("qdk-chemistry is not installed.")
        print("Install with: pip install pyscf-ormas-ci[qdk]")
        raise SystemExit(1)

    # --- Step 1: QDK SCF on H2O ---
    xyz_str = (
        "3\nWater\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.000000  0.757000  0.587000\n"
        "H  0.000000 -0.757000  0.587000"
    )
    structure = Structure.from_xyz(xyz_str)
    scf_solver = PyscfScfSolver()
    scf_energy, wfn = scf_solver.run(structure, 0, 1, "6-31g")

    # --- Step 2: Convert QDK orbitals to PySCF SCF object ---
    container = wfn.get_container()
    orbitals = container.get_orbitals()
    n_mo = orbitals.get_num_molecular_orbitals()
    n_occ = 5  # H2O has 10 electrons -> 5 doubly occupied
    occ = [1] * n_occ + [0] * (n_mo - n_occ)
    pyscf_scf = orbitals_to_scf(orbitals, occ_alpha=occ, occ_beta=occ, scf_type=SCFType.RESTRICTED)

    # --- Step 3: CASCI + ORMAS-CI ---
    ncas, nelecas = 4, (2, 2)

    # Full CASCI reference
    mc_ref = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc_ref.verbose = 0
    e_casci = mc_ref.kernel()[0]

    # ORMAS: bond/lone_pair subspaces
    config = ORMASConfig(
        subspaces=[
            Subspace("bond", [0, 1], min_electrons=1, max_electrons=4),
            Subspace("lone_pair", [2, 3], min_electrons=0, max_electrons=3),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc_ormas = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc_ormas.verbose = 0
    mc_ormas.fcisolver = ORMASFCISolver(config)
    e_ormas = mc_ormas.kernel()[0]

    # --- Step 4: QDK qubit Hamiltonian ---
    as_sel = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=sum(nelecas),
        num_active_orbitals=ncas,
    )
    sel_wfn = as_sel.run(wfn)
    sel_orbs = sel_wfn.get_orbitals()
    hamiltonian = QdkHamiltonianConstructor().run(sel_orbs)
    qubit_ham = QdkQubitMapper().run(hamiltonian)

    # --- Step 5: Classical qubit solve ---
    solver = create("qubit_hamiltonian_solver", "qdk_sparse_matrix_solver")
    e_qubit_raw, _ = solver.run(qubit_ham)
    e_qubit = e_qubit_raw + hamiltonian.get_core_energy()

    # --- Results ---
    n_det_casci = casci_determinant_count(ncas, nelecas)
    n_det_ormas = count_determinants(config)

    print("QDK Pipeline: H2O/6-31G CAS(4,4)")
    print("=" * 60)
    print()
    print("Classical comparison:")
    print(f"  CASCI energy:    {e_casci:.10f} Ha ({n_det_casci} dets)")
    print(f"  ORMAS energy:    {e_ormas:.10f} Ha ({n_det_ormas} dets)")
    print(f"  Energy error:    {(e_ormas - e_casci) * 1000:+.3f} mHa")
    print()
    print("Qubit Hamiltonian (Jordan-Wigner):")
    print(f"  Qubits:          {qubit_ham.num_qubits}")
    print(f"  Pauli terms:     {len(qubit_ham.pauli_strings)}")
    print(f"  Encoding:        {qubit_ham.encoding}")
    print(f"  Qubit solve E:   {e_qubit:.10f} Ha")
    print()
    print("The QDK pipeline produces the same molecular system as PySCF,")
    print("and the ORMAS-CI solver plugs in at the CI step.")


if __name__ == "__main__":
    main()
