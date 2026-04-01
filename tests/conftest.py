"""Shared pytest fixtures for ORMAS-CI test suite."""

import pytest

from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.ormas_ci.fcisolver import ORMASFCISolver
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace

# Standard tolerances for new tests
ENERGY_ATOL = 1e-10
RDM_ATOL = 1e-10
S2_ATOL = 1e-6


@pytest.fixture
def h2_mol():
    """H2 molecule at equilibrium, 6-31G basis."""
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)


@pytest.fixture
def h2_rhf(h2_mol):
    """Converged RHF for H2/6-31G."""
    mf = scf.RHF(h2_mol)
    mf.verbose = 0
    mf.run()
    return mf


@pytest.fixture
def h2_casci(h2_rhf):
    """PySCF CASCI(2,2) reference for H2/6-31G."""
    mc = mcscf.CASCI(h2_rhf, 2, 2)
    mc.verbose = 0
    mc.kernel()
    return mc


@pytest.fixture
def h2_ormas_config():
    """Unrestricted ORMAS config for CAS(2,2)."""
    return ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )


@pytest.fixture
def h2_ormas_solver(h2_rhf, h2_ormas_config):
    """Converged ORMAS CASCI(2,2) for H2/6-31G."""
    mc = mcscf.CASCI(h2_rhf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(h2_ormas_config)
    mc.kernel()
    return mc


@pytest.fixture
def h2_active_integrals(h2_rhf):
    """Active-space integrals for H2 CAS(2,2).

    Returns (h1e, h2e_4d, ecore, e_fci) where h2e_4d is the full
    4-index ERI array and e_fci is PySCF's FCI ground-state energy.
    """
    mc = mcscf.CASCI(h2_rhf, 2, 2)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    h2e = ao2mo.restore(1, h2e, 2)
    e_fci, _ = fci.direct_spin1.kernel(h1e, h2e, 2, 2)
    return h1e, h2e, ecore, e_fci
