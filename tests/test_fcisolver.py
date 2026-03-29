"""Tests for ormas_ci.fcisolver -- PySCF CASCI integration.

These are the critical integration tests that validate ORMASFCISolver as a
drop-in replacement for PySCF's default FCI solver inside CASCI calculations.

The gold standard test is test_unrestricted_matches_casci: when ORMAS has no
occupation restrictions, it must reproduce PySCF CASCI energy exactly.
"""

import logging

import numpy as np

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci.fcisolver import ORMASFCISolver
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace


def test_unrestricted_matches_casci():
    """Unrestricted ORMAS must match PySCF CASCI exactly (H2/STO-3G).

    This is the GOLD STANDARD test. A single subspace spanning all active
    orbitals with no occupation restrictions is mathematically identical
    to full CASCI. The energies must agree to machine precision.
    """
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    # PySCF reference
    mc_ref = mcscf.CASCI(mf, 2, 2)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # Our solver, no restrictions
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    mc_test = mcscf.CASCI(mf, 2, 2)
    mc_test.verbose = 0
    mc_test.fcisolver = ORMASFCISolver(config)
    e_test = mc_test.kernel()[0]

    assert abs(e_ref - e_test) < 1e-10, (
        f"Energy mismatch: PySCF={e_ref}, ORMAS={e_test}, "
        f"diff={abs(e_ref - e_test)}"
    )


def test_unrestricted_matches_casci_h2o():
    """Unrestricted ORMAS matches CASCI on H2O/cc-pVTZ (s/p/d/f coverage).

    Uses cc-pVTZ basis to exercise the solver with basis functions of all
    angular momentum types (s, p, d, f). A (4,6) active space is used to
    keep the test tractable while still being a meaningful validation.
    """
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVTZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 4, (3, 3)

    # PySCF reference
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # Our solver, unrestricted (single subspace, full range)
    config = ORMASConfig(
        subspaces=[
            Subspace("all", list(range(ncas)), 0, 2 * ncas),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc_test = mcscf.CASCI(mf, ncas, nelecas)
    mc_test.verbose = 0
    mc_test.fcisolver = ORMASFCISolver(config)
    e_test = mc_test.kernel()[0]

    assert abs(e_ref - e_test) < 1e-10, (
        f"H2O energy mismatch: PySCF={e_ref}, ORMAS={e_test}, "
        f"diff={abs(e_ref - e_test)}"
    )


def test_restricted_energy_above_casci():
    """Variational principle: restricted ORMAS energy >= CASCI energy.

    Restricting the determinant space removes variational freedom, so the
    restricted energy must be at or above the full CASCI energy. Uses
    H2O/cc-pVTZ with two subspaces, each requiring at least 2 electrons.
    """
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVTZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 4, (3, 3)

    # PySCF reference (full CASCI)
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # Restricted ORMAS: two subspaces with occupation bounds
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], min_electrons=2, max_electrons=4),
            Subspace("B", [2, 3], min_electrons=2, max_electrons=4),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc_test = mcscf.CASCI(mf, ncas, nelecas)
    mc_test.verbose = 0
    mc_test.fcisolver = ORMASFCISolver(config)
    e_test = mc_test.kernel()[0]

    # Allow tiny numerical noise below reference
    assert e_test >= e_ref - 1e-10, (
        f"Variational violation: restricted={e_test} < CASCI={e_ref}, "
        f"diff={e_test - e_ref}"
    )


def test_rdm1_from_fcisolver():
    """make_rdm1 works through the fcisolver interface after kernel().

    After running kernel(), calling make_rdm1 on the solver must return
    a valid 1-RDM whose trace equals the total number of electrons.
    """
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()

    rdm1 = mc.fcisolver.make_rdm1(mc.ci, 2, (1, 1))
    assert abs(np.trace(rdm1) - 2.0) < 1e-10, (
        f"RDM1 trace = {np.trace(rdm1)}, expected 2.0"
    )


def test_determinant_reduction_logged(caplog):
    """Solver should log determinant count information during kernel()."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)

    with caplog.at_level(logging.INFO, logger="pyscf.ormas_ci.fcisolver"):
        mc.kernel()

    assert any("determinant" in r.message.lower() for r in caplog.records), (
        "Expected log message containing 'determinant', got: "
        + str([r.message for r in caplog.records])
    )
