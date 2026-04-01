#!/usr/bin/env python3
"""Real IQPE and VQE simulation with method-specific state preparation.

Demonstrates the quantum computing value proposition of ORMAS-CI:
1. Build qubit Hamiltonian from the active space
2. Construct state prep circuits for CASCI and ORMAS wavefunctions
3. Run real iterative QPE via QDK's full-state simulator
4. Run real shot-based VQE energy estimation
5. Compare circuit costs and quantum energies between methods

The key insight: ORMAS restricts the determinant space, which produces
different (often shallower) state preparation circuits for quantum
simulation, while targeting the same qubit Hamiltonian.

Requires: pip install pyscf-ormas-ci[qdk]
"""

import numpy as np
from pyscf.fci import cistring

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count, count_determinants

try:
    from qdk_chemistry._core.data import (
        CasWavefunctionContainer,
        Configuration,
        Wavefunction,
    )
    from qdk_chemistry.algorithms import QdkHamiltonianConstructor, QdkQubitMapper, create
    from qdk_chemistry.algorithms.state_preparation.sparse_isometry import (
        SparseIsometryGF2XStatePreparation,
    )
    from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
    from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper import (
        PauliSequenceMapper,
    )
    from qdk_chemistry.data import Structure
    from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False


def bitstrings_to_configurations(alpha_strings, beta_strings, ncas):
    """Convert ORMAS alpha/beta bitstrings to QDK Configuration objects."""
    configs = []
    for a, b in zip(alpha_strings, beta_strings):
        occ = ""
        for orb in range(ncas):
            has_a = bool(int(a) & (1 << orb))
            has_b = bool(int(b) & (1 << orb))
            if has_a and has_b:
                occ += "2"
            elif has_a:
                occ += "u"
            elif has_b:
                occ += "d"
            else:
                occ += "0"
        configs.append(Configuration(occ))
    return configs


def build_state_prep(ci_vector, alpha_strings, beta_strings, ncas, qdk_orbs):
    """Build a QDK state prep circuit from a CI wavefunction."""
    configs = bitstrings_to_configurations(alpha_strings, beta_strings, ncas)
    container = CasWavefunctionContainer(ci_vector, configs, qdk_orbs)
    wfn = Wavefunction(container)
    sp = SparseIsometryGF2XStatePreparation()
    return sp.run(wfn)


def main():
    if not QDK_AVAILABLE:
        print("qdk-chemistry is not installed.")
        print("Install with: pip install pyscf-ormas-ci[qdk]")
        raise SystemExit(1)

    # --- Setup: Formaldehyde n/pi/pi* system ---
    mol = gto.M(
        atom="C 0 0 0; O 0 0 1.208; H 0 0.943 -0.561; H 0 -0.943 -0.561",
        basis="cc-pVDZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 3, (2, 2)

    # Full CASCI
    mc_cas = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_cas.verbose = 0
    mc_cas.kernel()
    e_casci = mc_cas.e_tot

    # ORMAS: n_O / pi,pi*
    ras_config = ORMASConfig(
        subspaces=[
            Subspace("n_O", [0], min_electrons=1, max_electrons=2),
            Subspace("pi", [1, 2], min_electrons=1, max_electrons=3),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc_ormas = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_ormas.verbose = 0
    mc_ormas.fcisolver = ORMASFCISolver(ras_config)
    mc_ormas.kernel()
    e_ormas = mc_ormas.e_tot

    # --- Build QDK qubit Hamiltonian ---
    xyz_str = "4\nCH2O\nC 0 0 0\nO 0 0 1.208\nH 0 0.943 -0.561\nH 0 -0.943 -0.561"
    _, wfn = PyscfScfSolver().run(Structure.from_xyz(xyz_str), 0, 1, "cc-pVDZ")
    as_sel = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=sum(nelecas),
        num_active_orbitals=ncas,
    )
    sel_wfn = as_sel.run(wfn)
    qdk_orbs = sel_wfn.get_orbitals()
    hamiltonian = QdkHamiltonianConstructor().run(qdk_orbs)
    qubit_ham = QdkQubitMapper().run(hamiltonian)
    core_energy = hamiltonian.get_core_energy()

    print(
        f"Formaldehyde CAS(4,3)/cc-pVDZ: {qubit_ham.num_qubits} qubits, "
        f"{len(qubit_ham.pauli_strings)} Pauli terms"
    )

    # --- Build method-specific state prep circuits ---
    # CASCI: flatten PySCF's 2D CI matrix to 1D + determinant arrays
    alpha_cas = cistring.gen_strings4orblist(range(ncas), nelecas[0])
    beta_cas = cistring.gen_strings4orblist(range(ncas), nelecas[1])
    ci_flat = []
    a_flat = []
    b_flat = []
    for ia, a in enumerate(alpha_cas):
        for ib, b in enumerate(beta_cas):
            ci_flat.append(mc_cas.ci[ia, ib])
            a_flat.append(a)
            b_flat.append(b)

    cas_prep = build_state_prep(
        np.array(ci_flat),
        np.array(a_flat, dtype=np.int64),
        np.array(b_flat, dtype=np.int64),
        ncas,
        qdk_orbs,
    )

    # ORMAS: use the restricted CI vector directly
    ormas_prep = build_state_prep(
        mc_ormas.fcisolver.ci,
        mc_ormas.fcisolver._alpha_strings,
        mc_ormas.fcisolver._beta_strings,
        ncas,
        qdk_orbs,
    )

    print(f"CAS state prep:   {len(cas_prep.qasm):>6} QASM chars")
    print(f"ORMAS state prep: {len(ormas_prep.qasm):>6} QASM chars")

    # --- Run IQPE ---
    coeffs = np.abs(qubit_ham.coefficients)
    pauli_1_norm = float(np.sum(coeffs))
    evolution_time = 2 * np.pi / (2 * pauli_1_norm)

    trotter = Trotter()
    mapper = PauliSequenceMapper()
    executor = create("circuit_executor", "qdk_full_state_simulator")

    print("\nRunning 6-bit IQPE...")
    for label, prep in [("CASCI", cas_prep), ("ORMAS", ormas_prep)]:
        pe = create(
            "phase_estimation",
            "iterative",
            num_bits=6,
            evolution_time=evolution_time,
            shots_per_bit=100,
        )
        result = pe.run(
            prep,
            qubit_ham,
            evolution_builder=trotter,
            circuit_mapper=mapper,
            circuit_executor=executor,
        )
        e_iqpe = result.to_dict()["raw_energy"] + core_energy
        print(f"  {label} IQPE energy: {e_iqpe:.8f} Ha")

    # --- Run VQE energy estimation ---
    print("\nRunning VQE (1000 shots)...")
    grouped = qubit_ham.group_commuting()
    estimator = create("energy_estimator", "qdk_base_simulator")
    for label, prep in [("CASCI", cas_prep), ("ORMAS", ormas_prep)]:
        energy_result, _ = estimator.run(prep, grouped, total_shots=1000)
        e_vqe = energy_result.to_dict()["energy_expectation_value"] + core_energy
        print(f"  {label} VQE energy:  {e_vqe:.8f} Ha")

    # --- Summary ---
    n_det_cas = casci_determinant_count(ncas, nelecas)
    n_det_ormas = count_determinants(ras_config)

    print(f"\n{'=' * 55}")
    print("Summary: Formaldehyde CAS(4,3)/cc-pVDZ")
    print(f"{'=' * 55}")
    print(f"{'Method':<8} {'n_det':>6} {'E_classical':>16} {'dE(mHa)':>10}")
    print(f"{'CASCI':<8} {n_det_cas:>6} {e_casci:>16.8f} {'-':>10}")
    print(f"{'ORMAS':<8} {n_det_ormas:>6} {e_ormas:>16.8f} {(e_ormas - e_casci) * 1000:>+10.3f}")
    print()
    print("ORMAS produces a different state prep circuit from the same")
    print("qubit Hamiltonian. The quantum simulation results (IQPE, VQE)")
    print("reflect the quality of each method's wavefunction overlap with")
    print("the true ground state.")


if __name__ == "__main__":
    main()
