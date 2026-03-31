"""Integration tests for SF-ORMAS-CI with PySCF CASCI/CASSCF.

These tests run actual quantum chemistry calculations to validate that
SFORMASFCISolver works as a drop-in replacement inside PySCF's CASCI
framework for spin-flip calculations.

Gold standard: SF-ORMAS with full CAS bounds must match PySCF CASCI.
"""

import pytest

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace


def test_sf_ormas_matches_casci_h2():
    """SF-ORMAS with full CAS bounds must match PySCF CASCI (H2 stretched)."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    # PySCF native CASCI (singlet in M_s=0 sector)
    mc_ref = mcscf.CASCI(mf, 2, (1, 1))
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # SF-ORMAS (triplet ref -> singlet target, no restrictions)
    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )
    mc_sf = mcscf.CASCI(mf, 2, (1, 1))
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]

    assert abs(e_ref - e_sf) < 1e-10, (
        f"Energy mismatch: PySCF={e_ref}, SF-ORMAS={e_sf}, "
        f"diff={abs(e_ref - e_sf)}"
    )


def test_sf_ormas_matches_casci_h2_equilibrium():
    """SF-ORMAS matches PySCF CASCI at H2 equilibrium geometry."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    # PySCF native CASCI
    mc_ref = mcscf.CASCI(mf, 2, (1, 1))
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # SF-ORMAS
    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )
    mc_sf = mcscf.CASCI(mf, 2, (1, 1))
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]

    assert abs(e_ref - e_sf) < 1e-10, (
        f"Energy mismatch: PySCF={e_ref}, SF-ORMAS={e_sf}, "
        f"diff={abs(e_ref - e_sf)}"
    )


def test_sf_ormas_nelecas_mismatch_raises():
    """CASCI with wrong nelecas must raise ValueError."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )

    # Construct with reference nelecas (2,0) instead of target (1,1)
    mc_sf = mcscf.CASCI(mf, 2, (2, 0))
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)

    with pytest.raises(ValueError, match="target"):
        mc_sf.kernel()


def test_sf_ormas_spin_pure_h2():
    """H2 CAS(2,2) nroots=2: each root must be a spin eigenstate."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )
    mc = mcscf.CASCI(mf, 2, (1, 1))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config)
    mc.fcisolver.nroots = 2
    mc.kernel()

    s_values = []
    for i in range(2):
        ss, mult = mc.fcisolver.spin_square(mc.ci[i], 2, (1, 1))
        s_val = (-1 + (1 + 4 * ss) ** 0.5) / 2
        assert abs(s_val - round(s_val)) < 1e-6, (
            f"Root {i}: S={s_val} is not a spin eigenstate"
        )
        s_values.append(round(s_val))

    # One singlet (S=0) and one triplet (S=1) component
    assert 0 in s_values, f"No singlet root found, S values: {s_values}"
    assert 1 in s_values, f"No triplet root found, S values: {s_values}"


def test_sf_ormas_spin_pure_larger():
    """4-electron/4-orbital SF: verify spin purity across multiple roots."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=4, n_active_electrons=4,
        subspaces=[Subspace("all", list(range(4)), 0, 8)],
    )
    mc = mcscf.CASCI(mf, 4, (2, 2))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config)
    mc.fcisolver.nroots = 4
    mc.kernel()

    for i in range(len(mc.ci)):
        ss, mult = mc.fcisolver.spin_square(mc.ci[i], 4, (2, 2))
        s_val = (-1 + (1 + 4 * ss) ** 0.5) / 2
        assert abs(s_val - round(s_val)) < 1e-6, (
            f"Root {i}: S={s_val:.6f} is not a spin eigenstate "
            f"(<S^2>={ss:.6f})"
        )


def test_sf_ormas_multiroot_energy_ordering():
    """H2 stretched, nroots=2: E[0] <= E[1]."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )
    mc = mcscf.CASCI(mf, 2, (1, 1))
    mc.verbose = 0
    mc.fcisolver = SFORMASFCISolver(sf_config)
    mc.fcisolver.nroots = 2
    mc.kernel()
    e_tot = mc.e_tot

    assert e_tot[0] <= e_tot[1] + 1e-12, (
        f"Energy ordering violated: E[0]={e_tot[0]}, E[1]={e_tot[1]}"
    )


def test_sf_ormas_casscf():
    """SF-ORMAS with CASSCF orbital optimization on H2."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], 0, 4)],
    )

    # CASCI reference energy
    mc_ci = mcscf.CASCI(mf, 2, (1, 1))
    mc_ci.verbose = 0
    mc_ci.fcisolver = SFORMASFCISolver(sf_config)
    e_casci = mc_ci.kernel()[0]

    # CASSCF with SF-ORMAS solver
    mc_scf = mcscf.CASSCF(mf, 2, (1, 1))
    mc_scf.verbose = 0
    mc_scf.fcisolver = SFORMASFCISolver(sf_config)
    e_casscf = mc_scf.kernel()[0]

    assert mc_scf.converged, "CASSCF did not converge"
    assert e_casscf <= e_casci + 1e-10, (
        f"CASSCF energy ({e_casscf}) should be <= CASCI energy ({e_casci})"
    )
