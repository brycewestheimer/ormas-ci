"""SF-ORMAS + QDK/Chemistry integration tests.

Validates the full pipeline: ROHF -> SF-ORMAS CASCI -> QDK resource estimation.
These tests are skipped if qdk-chemistry is not installed.
"""

import pytest

try:
    from qdk_chemistry._core.data import (
        CanonicalFourCenterHamiltonianContainer,
        ModelOrbitals,
    )
    from qdk_chemistry._core.data import Hamiltonian as QdkHamiltonian
    from qdk_chemistry.algorithms import QdkQubitMapper

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QDK_AVAILABLE, reason="qdk-chemistry not installed")


def _twisted_ethylene_rohf():
    """Build and run ROHF for twisted ethylene (triplet reference)."""
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.340  0.000  0.000
        H -0.500  0.930  0.000
        H -0.500 -0.930  0.000
        H  1.840  0.000  0.930
        H  1.840  0.000 -0.930
        """,
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mf


def _build_sf_qubit_hamiltonian(mf, ncas: int, nelecas: tuple[int, int]):
    """Build QDK qubit Hamiltonian from PySCF ROHF via ModelOrbitals bridge.

    QDK's active-space selectors don't support open-shell systems, so we
    extract integrals from PySCF CASCI and construct the QDK Hamiltonian
    using CanonicalFourCenterHamiltonianContainer with model orbitals.

    Args:
        mf: Converged PySCF ROHF object.
        ncas: Number of active orbitals.
        nelecas: Tuple of (n_alpha, n_beta) active electrons.

    Returns:
        QDK qubit Hamiltonian object.
    """
    import numpy as np

    from pyscf import ao2mo as ao2mo_mod
    from pyscf import mcscf

    mc_tmp = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_tmp.verbose = 0
    mc_tmp.kernel()
    h1e, ecore = mc_tmp.get_h1eff()
    eri = mc_tmp.get_h2eff()
    if eri.ndim != 4:
        eri = ao2mo_mod.restore(1, eri, ncas)
    eri_flat = eri.reshape(-1)
    fock = np.zeros((ncas, ncas))

    model_orbs = ModelOrbitals(ncas, True)
    container = CanonicalFourCenterHamiltonianContainer(h1e, eri_flat, model_orbs, ecore, fock)
    hamiltonian = QdkHamiltonian(container)
    qubit_ham = QdkQubitMapper().run(hamiltonian)
    return qubit_ham


def test_sf_ormas_qdk_pipeline() -> None:
    """Full pipeline: ROHF -> SF-ORMAS CASCI -> QDK qubit Hamiltonian."""
    import math

    from pyscf import mcscf
    from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace

    mf = _twisted_ethylene_rohf()

    sf_config = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("pi", [0, 1], 0, 4)],
    )

    mc = mcscf.CASCI(mf, 2, (1, 1))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config)
    e_tot = mc.kernel()[0]

    assert math.isfinite(e_tot), f"Energy is not finite: {e_tot}"

    qubit_ham = _build_sf_qubit_hamiltonian(mf, 2, (1, 1))

    assert qubit_ham.num_qubits > 0, "Qubit Hamiltonian has no qubits"
    assert len(qubit_ham.pauli_strings) > 0, "Qubit Hamiltonian has no Pauli strings"


def test_sf_ormas_qdk_multiroot() -> None:
    """Multi-root SF-ORMAS with QDK qubit Hamiltonian."""
    from pyscf import mcscf
    from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace

    mf = _twisted_ethylene_rohf()

    sf_config = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("pi", [0, 1], 0, 4)],
    )

    mc = mcscf.CASCI(mf, 2, (1, 1))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config)
    mc.fcisolver.nroots = 2
    mc.kernel()

    assert len(mc.e_tot) == 2, f"Expected 2 roots, got {len(mc.e_tot)}"
    assert mc.ci[0] is not None, "CI vector for root 0 is None"
    assert mc.ci[1] is not None, "CI vector for root 1 is None"

    qubit_ham = _build_sf_qubit_hamiltonian(mf, 2, (1, 1))

    # 2 spatial orbitals x 2 spin channels = 4 qubits (Jordan-Wigner)
    assert qubit_ham.num_qubits == 4, (
        f"Expected 4 qubits for 2 spatial orbitals, got {qubit_ham.num_qubits}"
    )


def test_sf_ormas_qdk_resource_reduction() -> None:
    """ORMAS restrictions reduce determinant count vs full CAS."""
    from pyscf import gto, mcscf, scf
    from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace
    from pyscf.ormas_ci.determinants import casci_determinant_count
    from pyscf.ormas_ci.spinflip import count_sf_determinants

    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.350  0.000  0.000
        C -0.675  1.169  0.000
        C -0.675 -1.169  0.000
        H  1.944  0.930  0.000
        H  1.944 -0.930  0.000
        H -1.269  2.099  0.000
        H -0.081  2.099  0.000
        H -1.269 -2.099  0.000
        H -0.081 -2.099  0.000
        """,
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config_restricted = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=4,
        n_active_electrons=4,
        subspaces=[
            Subspace("pi1", [0, 1], min_electrons=1, max_electrons=3),
            Subspace("pi2", [2, 3], min_electrons=1, max_electrons=3),
        ],
    )

    mc = mcscf.CASCI(mf, 4, (2, 2))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config_restricted)
    mc.kernel()

    n_det_restricted = count_sf_determinants(sf_config_restricted)
    n_det_full = casci_determinant_count(4, (2, 2))

    assert n_det_restricted <= n_det_full, (
        f"Restricted ({n_det_restricted}) exceeds full CAS ({n_det_full})"
    )
    assert n_det_restricted < n_det_full, (
        f"Expected strict reduction: restricted ({n_det_restricted}) "
        f"should be < full CAS ({n_det_full})"
    )

    qubit_ham = _build_sf_qubit_hamiltonian(mf, 4, (2, 2))

    assert qubit_ham.num_qubits > 0, "Qubit Hamiltonian pipeline failed"
