"""Tests for PySCF drop-in compatibility.

Verifies that ORMASFCISolver works as a complete PySCF FCI solver
replacement, including addon compatibility (fix_spin_), StreamObject
inheritance, and all required interface methods.
"""

import numpy as np
import pytest

from pyscf import ao2mo, gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def h2_setup():
    """H2/6-31G with unrestricted ORMAS config (single subspace)."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run()
    config = ORMASConfig(
        subspaces=[Subspace('all', [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    return mol, mf, config


@pytest.fixture()
def h2_solver_and_integrals(h2_setup):
    """Return solver after kernel() with active-space integrals."""
    mol, mf, config = h2_setup
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    solver = ORMASFCISolver(config)
    mc.fcisolver = solver
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    eri = mc.get_h2eff()
    return solver, h1e, eri, ecore


# ---------------------------------------------------------------------------
# StreamObject inheritance
# ---------------------------------------------------------------------------

class TestStreamObjectInheritance:
    def test_has_copy(self, h2_setup):
        _, _, config = h2_setup
        solver = ORMASFCISolver(config)
        assert hasattr(solver, 'copy')
        assert callable(solver.copy)

    def test_copy_returns_new_object(self, h2_setup):
        _, _, config = h2_setup
        solver = ORMASFCISolver(config)
        solver.nroots = 3
        copied = solver.copy()
        assert copied is not solver
        assert copied.nroots == 3
        assert copied.config is solver.config

    def test_has_stdout(self, h2_setup):
        _, _, config = h2_setup
        solver = ORMASFCISolver(config)
        assert hasattr(solver, 'stdout')

    def test_has_max_memory(self, h2_setup):
        _, _, config = h2_setup
        solver = ORMASFCISolver(config)
        assert hasattr(solver, 'max_memory')
        assert solver.max_memory > 0

    def test_has_verbose(self, h2_setup):
        _, _, config = h2_setup
        solver = ORMASFCISolver(config)
        assert hasattr(solver, 'verbose')


# ---------------------------------------------------------------------------
# State variables after kernel()
# ---------------------------------------------------------------------------

class TestKernelStateVariables:
    def test_sets_eci(self, h2_setup):
        _, mf, config = h2_setup
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        solver = ORMASFCISolver(config)
        mc.fcisolver = solver
        mc.kernel()
        assert solver.eci is not None
        assert isinstance(solver.eci, float)

    def test_sets_ci(self, h2_setup):
        _, mf, config = h2_setup
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        solver = ORMASFCISolver(config)
        mc.fcisolver = solver
        mc.kernel()
        assert solver.ci is not None
        assert isinstance(solver.ci, np.ndarray)

    def test_sets_norb_nelec(self, h2_setup):
        _, mf, config = h2_setup
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        solver = ORMASFCISolver(config)
        mc.fcisolver = solver
        mc.kernel()
        assert solver.norb == 2
        assert solver.nelec == (1, 1)

    def test_sets_converged(self, h2_setup):
        _, mf, config = h2_setup
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        solver = ORMASFCISolver(config)
        mc.fcisolver = solver
        mc.kernel()
        assert solver.converged is True


# ---------------------------------------------------------------------------
# make_hdiag
# ---------------------------------------------------------------------------

class TestMakeHdiag:
    def test_matches_hamiltonian_diagonal(self, h2_solver_and_integrals):
        solver, h1e, eri, ecore = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        hdiag = solver.make_hdiag(h1e, eri_4d, 2, (1, 1))

        from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
        h_ci = build_ci_hamiltonian(
            solver._alpha_strings, solver._beta_strings, h1e, eri_4d
        )
        if hasattr(h_ci, 'toarray'):
            h_ci = h_ci.toarray()
        expected = np.diag(h_ci)
        np.testing.assert_allclose(hdiag, expected, atol=1e-12)

    def test_correct_length(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        hdiag = solver.make_hdiag(h1e, eri_4d, 2, (1, 1))
        assert len(hdiag) == len(solver._alpha_strings)


# ---------------------------------------------------------------------------
# pspace
# ---------------------------------------------------------------------------

class TestPspace:
    def test_returns_subblock_and_addresses(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        h_sub, addr = solver.pspace(h1e, eri_4d, 2, (1, 1), np_=2)
        assert h_sub.shape == (2, 2)
        assert len(addr) == 2

    def test_full_space(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        n_det = len(solver._alpha_strings)
        h_sub, addr = solver.pspace(h1e, eri_4d, 2, (1, 1), np_=n_det + 10)
        assert h_sub.shape == (n_det, n_det)
        assert len(addr) == n_det


# ---------------------------------------------------------------------------
# get_init_guess
# ---------------------------------------------------------------------------

class TestGetInitGuess:
    def test_returns_unit_vectors(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        hdiag = solver.make_hdiag(h1e, eri_4d, 2, (1, 1))
        ci0 = solver.get_init_guess(2, (1, 1), 1, hdiag)
        assert len(ci0) == 1
        assert np.count_nonzero(ci0[0]) == 1
        assert ci0[0][np.argmin(hdiag)] == 1.0

    def test_multiple_roots(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        eri_4d = ao2mo.restore(1, eri, 2)
        hdiag = solver.make_hdiag(h1e, eri_4d, 2, (1, 1))
        ci0 = solver.get_init_guess(2, (1, 1), 3, hdiag)
        assert len(ci0) == 3
        for v in ci0:
            assert np.count_nonzero(v) == 1


# ---------------------------------------------------------------------------
# contract_1e
# ---------------------------------------------------------------------------

class TestContract1e:
    def test_matches_1e_hamiltonian(self, h2_solver_and_integrals):
        solver, h1e, eri, _ = h2_solver_and_integrals
        ci = solver._ci_vector
        sigma = solver.contract_1e(h1e, ci, 2, (1, 1))
        # Build 1e-only Hamiltonian for reference
        from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
        h2e_zero = np.zeros((2, 2, 2, 2))
        h1_only = build_ci_hamiltonian(
            solver._alpha_strings, solver._beta_strings, h1e, h2e_zero
        )
        if hasattr(h1_only, 'toarray'):
            h1_only = h1_only.toarray()
        expected = h1_only @ ci
        np.testing.assert_allclose(sigma, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# contract_ss
# ---------------------------------------------------------------------------

class TestContractSs:
    def test_singlet_eigenvalue(self, h2_solver_and_integrals):
        """For a singlet state, <Psi|S^2|Psi> should be ~0."""
        solver, _, _, _ = h2_solver_and_integrals
        ci = solver._ci_vector
        sigma = solver.contract_ss(ci, 2, (1, 1))
        ss = np.dot(ci, sigma)
        assert abs(ss) < 1e-8, f"<S^2> = {ss}, expected ~0 for singlet"

    def test_matches_spin_square(self, h2_solver_and_integrals):
        """contract_ss expectation value matches spin_square result."""
        solver, _, _, _ = h2_solver_and_integrals
        ci = solver._ci_vector
        sigma = solver.contract_ss(ci, 2, (1, 1))
        ss_from_contract = np.dot(ci, sigma)
        ss_from_method, _ = solver.spin_square(ci, 2, (1, 1))
        np.testing.assert_allclose(ss_from_contract, ss_from_method, atol=1e-10)


# ---------------------------------------------------------------------------
# fix_spin_ addon
# ---------------------------------------------------------------------------

class TestFixSpin:
    def test_fix_spin_does_not_crash(self, h2_setup):
        _, _, config = h2_setup
        from pyscf.fci.addons import fix_spin_
        solver = ORMASFCISolver(config)
        fix_spin_(solver, shift=0.1)

    def test_fix_spin_energy(self, h2_setup):
        """fix_spin_ should give same energy for a singlet ground state."""
        _, mf, config = h2_setup
        from pyscf.fci.addons import fix_spin_

        solver = ORMASFCISolver(config)
        fix_spin_(solver, shift=0.1, ss=0)

        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        mc.fcisolver = solver
        e_fixed = mc.kernel()[0]

        mc_ref = mcscf.CASCI(mf, 2, 2)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        assert abs(e_fixed - e_ref) < 1e-8


# ---------------------------------------------------------------------------
# Hamiltonian caching
# ---------------------------------------------------------------------------

class TestHamiltonianCache:
    def test_contract_2e_uses_cache(self, h2_solver_and_integrals):
        """contract_2e should not rebuild Hamiltonian after kernel()."""
        solver, h1e, eri, _ = h2_solver_and_integrals
        assert solver._h_ci_cached is not None
        cached_id = id(solver._h_ci_cached)
        eri_4d = ao2mo.restore(1, eri, 2)
        solver.contract_2e(eri_4d, solver._ci_vector, 2, (1, 1))
        assert id(solver._h_ci_cached) == cached_id
