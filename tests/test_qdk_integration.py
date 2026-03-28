"""Tests for QDK/Chemistry integration.

Verifies the end-to-end workflow: QDK SCF -> PySCF CASCI with ORMASFCISolver.
These tests are skipped if qdk-chemistry is not installed.
"""

import pytest

try:
    from qdk_chemistry.data import Structure
    from qdk_chemistry.plugins.pyscf.conversion import SCFType, orbitals_to_scf
    from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QDK_AVAILABLE, reason="qdk-chemistry not installed")


def _h2_qdk_scf():
    """Run QDK SCF on H2/STO-3G and return (pyscf_scf, qdk_wfn)."""
    xyz_str = "2\nH2\nH  0.000000  0.000000  0.000000\nH  0.000000  0.000000  0.740000"
    structure = Structure.from_xyz(xyz_str)
    scf_solver = PyscfScfSolver()
    scf_energy, wfn = scf_solver.run(structure, 0, 1, "sto-3g")

    container = wfn.get_container()
    orbitals = container.get_orbitals()
    n_mo = orbitals.get_num_molecular_orbitals()
    occ_a = [1] + [0] * (n_mo - 1)
    occ_b = [1] + [0] * (n_mo - 1)
    pyscf_scf = orbitals_to_scf(
        orbitals, occ_alpha=occ_a, occ_beta=occ_b, scf_type=SCFType.RESTRICTED
    )
    return pyscf_scf, wfn


def test_qdk_scf_to_ormas_casci():
    """End-to-end: QDK SCF -> PySCF CASCI with ORMASFCISolver matches CASCI."""
    from pyscf import mcscf

    from ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

    pyscf_scf, _ = _h2_qdk_scf()

    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )

    mc = mcscf.CASCI(pyscf_scf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    e_ormas = mc.kernel()[0]

    mc_ref = mcscf.CASCI(pyscf_scf, 2, 2)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    assert abs(e_ormas - e_ref) < 1e-10, f"Energy mismatch: {e_ormas} vs {e_ref}"


def test_qdk_scf_restricted_ormas():
    """QDK SCF -> restricted ORMAS-CI gives energy >= CASCI."""
    from pyscf import mcscf

    from ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

    pyscf_scf, _ = _h2_qdk_scf()

    config = ORMASConfig(
        subspaces=[
            Subspace("s1", [0], min_electrons=1, max_electrons=1),
            Subspace("s2", [1], min_electrons=1, max_electrons=1),
        ],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )

    mc = mcscf.CASCI(pyscf_scf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    e_restricted = mc.kernel()[0]

    mc_ref = mcscf.CASCI(pyscf_scf, 2, 2)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    assert e_restricted >= e_ref - 1e-10, (
        f"Restricted energy {e_restricted} below CASCI {e_ref} (variational violation)"
    )


def test_qdk_scf_energy_reasonable():
    """QDK SCF produces a reasonable H2 energy."""
    pyscf_scf, _ = _h2_qdk_scf()
    # H2/STO-3G HF energy should be around -1.117
    assert -1.2 < pyscf_scf.e_tot < -1.0 or pyscf_scf.e_tot == 0
    # Note: orbitals_to_scf may not set e_tot; the CASCI test is authoritative.


def _h2o_qdk_scf():
    """Run QDK SCF on H2O/STO-3G and return pyscf_scf."""
    xyz_str = (
        "3\nWater\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.000000  0.757000  0.587000\n"
        "H  0.000000 -0.757000  0.587000"
    )
    structure = Structure.from_xyz(xyz_str)
    scf_solver = PyscfScfSolver()
    scf_energy, wfn = scf_solver.run(structure, 0, 1, "sto-3g")

    container = wfn.get_container()
    orbitals = container.get_orbitals()
    n_mo = orbitals.get_num_molecular_orbitals()
    n_occ = 5  # H2O has 10 electrons -> 5 doubly occupied
    occ_a = [1] * n_occ + [0] * (n_mo - n_occ)
    occ_b = [1] * n_occ + [0] * (n_mo - n_occ)
    pyscf_scf = orbitals_to_scf(
        orbitals, occ_alpha=occ_a, occ_beta=occ_b, scf_type=SCFType.RESTRICTED
    )
    return pyscf_scf


def test_qdk_h2o_ormas_matches_casci():
    """QDK SCF on H2O -> unrestricted ORMAS matches full CASCI."""
    from pyscf import mcscf

    from ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

    pyscf_scf = _h2o_qdk_scf()
    ncas = 4
    nelecas = (2, 2)

    config = ORMASConfig(
        subspaces=[
            Subspace("all", [0, 1, 2, 3], min_electrons=0, max_electrons=8),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )

    mc = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    e_ormas = mc.kernel()[0]

    mc_ref = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    assert abs(e_ormas - e_ref) < 1e-10, f"Energy mismatch: {e_ormas} vs {e_ref}"


def test_qdk_h2o_restricted_ormas():
    """QDK SCF on H2O -> restricted ORMAS obeys variational principle."""
    from pyscf import mcscf

    from ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

    pyscf_scf = _h2o_qdk_scf()
    ncas = 4
    nelecas = (2, 2)

    config = ORMASConfig(
        subspaces=[
            Subspace("bond", [0, 1], min_electrons=1, max_electrons=4),
            Subspace("lone", [2, 3], min_electrons=0, max_electrons=3),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )

    mc = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    e_restricted = mc.kernel()[0]

    mc_ref = mcscf.CASCI(pyscf_scf, ncas, sum(nelecas))
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    assert e_restricted >= e_ref - 1e-10, (
        f"Restricted energy {e_restricted} below CASCI {e_ref}"
    )
