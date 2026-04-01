"""Tests verifying the matrix-free path matches the explicit Hamiltonian path.

The matrix-free path uses PySCF's `pyscf.fci.selected_ci` C-level routines
as a computational backend (not as a Selected CI method). These tests create
two solver instances for the same system — one using the matrix-free path
(default for n_det > 200) and one forced to the explicit Hamiltonian path
via a high direct_ci_threshold — then compare energies, RDMs, spin, and
sigma vectors.
"""

import numpy as np
import pytest

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

# ---------- fixtures ----------

@pytest.fixture(scope="module")
def n2_mf():
    """N2/6-31G RHF — CAS(6,6) gives 400 determinants."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="6-31g", verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope="module")
def n2_cas66_config():
    """Full-CAS config for N2 CAS(6,6)."""
    return ORMASConfig(
        subspaces=[Subspace("full", list(range(6)), 0, 12)],
        n_active_orbitals=6,
        nelecas=(3, 3),
    )


def _run_solver(mf, config, ncas, nelecas, *, threshold, nroots=1):
    """Run CASCI with ORMAS solver at a given direct_ci_threshold."""
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.fcisolver.direct_ci_threshold = threshold
    mc.fcisolver.nroots = nroots
    mc.kernel()
    return mc


# ---------- single-root tests ----------

class TestSCIvsExplicitSingleRoot:
    """Compare SCI and explicit paths for ground-state properties."""

    @pytest.fixture(autouse=True)
    def setup(self, n2_mf, n2_cas66_config):
        self.mc_sci = _run_solver(
            n2_mf, n2_cas66_config, 6, 6, threshold=50,
        )
        self.mc_explicit = _run_solver(
            n2_mf, n2_cas66_config, 6, 6, threshold=999999,
        )
        # Verify the two paths are actually different
        assert self.mc_sci.fcisolver._use_sci
        assert self.mc_explicit.fcisolver._h_ci_cached is not None

    def test_energy_agreement(self):
        assert abs(self.mc_sci.e_tot - self.mc_explicit.e_tot) < 1e-10

    def test_rdm1_agreement(self):
        rdm1_sci = self.mc_sci.fcisolver.make_rdm1(
            self.mc_sci.ci, 6, (3, 3),
        )
        rdm1_exp = self.mc_explicit.fcisolver.make_rdm1(
            self.mc_explicit.ci, 6, (3, 3),
        )
        assert np.allclose(rdm1_sci, rdm1_exp, atol=1e-10)

    def test_rdm12_agreement(self):
        rdm1_sci, rdm2_sci = self.mc_sci.fcisolver.make_rdm12(
            self.mc_sci.ci, 6, (3, 3),
        )
        rdm1_exp, rdm2_exp = self.mc_explicit.fcisolver.make_rdm12(
            self.mc_explicit.ci, 6, (3, 3),
        )
        assert np.allclose(rdm1_sci, rdm1_exp, atol=1e-10)
        assert np.allclose(rdm2_sci, rdm2_exp, atol=1e-10)

    def test_rdm1s_agreement(self):
        rdm1a_sci, rdm1b_sci = self.mc_sci.fcisolver.make_rdm1s(
            self.mc_sci.ci, 6, (3, 3),
        )
        rdm1a_exp, rdm1b_exp = self.mc_explicit.fcisolver.make_rdm1s(
            self.mc_explicit.ci, 6, (3, 3),
        )
        assert np.allclose(rdm1a_sci, rdm1a_exp, atol=1e-10)
        assert np.allclose(rdm1b_sci, rdm1b_exp, atol=1e-10)

    def test_spin_square_agreement(self):
        ss_sci, mult_sci = self.mc_sci.fcisolver.spin_square(
            self.mc_sci.ci, 6, (3, 3),
        )
        ss_exp, mult_exp = self.mc_explicit.fcisolver.spin_square(
            self.mc_explicit.ci, 6, (3, 3),
        )
        assert abs(ss_sci - ss_exp) < 1e-10
        assert abs(mult_sci - mult_exp) < 1e-10

    def test_make_hdiag_agreement(self):
        h1e = self.mc_sci.fcisolver._h1e
        eri = self.mc_sci.fcisolver._eri
        hdiag_sci = self.mc_sci.fcisolver.make_hdiag(h1e, eri, 6, (3, 3))
        hdiag_exp = self.mc_explicit.fcisolver.make_hdiag(h1e, eri, 6, (3, 3))
        assert np.allclose(hdiag_sci, hdiag_exp, atol=1e-10)

    def test_contract_2e_agreement(self):
        """Sigma vectors from both paths should match."""
        ci = self.mc_explicit.ci
        h1e = self.mc_explicit.fcisolver._h1e
        eri = self.mc_explicit.fcisolver._eri
        eri_abs = self.mc_explicit.fcisolver.absorb_h1e(
            h1e, eri, 6, (3, 3), fac=0.5,
        )
        sigma_sci = self.mc_sci.fcisolver.contract_2e(
            eri_abs, ci, 6, (3, 3),
        )
        sigma_exp = self.mc_explicit.fcisolver.contract_2e(
            eri_abs, ci, 6, (3, 3),
        )
        assert np.allclose(sigma_sci, sigma_exp, atol=1e-12)


# ---------- multi-root tests ----------

class TestSCIvsExplicitMultiRoot:
    """Compare SCI and explicit paths for multi-root calculations.

    Note: eigsh (ARPACK Lanczos) can occasionally miss excited states,
    especially near degeneracies. Only the ground state is guaranteed
    to match reliably. For robust multi-root, PySCF's Davidson solver
    would be needed (future work).
    """

    @pytest.fixture(autouse=True)
    def setup(self, n2_mf, n2_cas66_config):
        self.mc_sci = _run_solver(
            n2_mf, n2_cas66_config, 6, 6,
            threshold=50, nroots=3,
        )
        self.mc_explicit = _run_solver(
            n2_mf, n2_cas66_config, 6, 6,
            threshold=999999, nroots=3,
        )

    def test_multi_root_ground_state_energy(self):
        """Ground state (root 0) should always agree between paths."""
        assert abs(self.mc_sci.e_tot[0] - self.mc_explicit.e_tot[0]) < 1e-8

    def test_multi_root_all_are_eigenvalues(self):
        """All SCI roots should be valid eigenvalues of H, even if
        eigsh finds different excited states than dense eigh."""
        solver = self.mc_sci.fcisolver
        for i in range(3):
            ci = self.mc_sci.ci[i]
            sigma = solver._sigma_sci(ci)
            e_i = np.dot(ci, sigma)
            residual = np.linalg.norm(sigma - e_i * ci)
            assert residual < 1e-6, (
                f"Root {i} residual {residual:.2e} too large"
            )


# ---------- edge case tests ----------

class TestSCIEdgeCases:
    """Edge cases for the SCI path."""

    def test_nroots_ge_ndet_falls_back(self, n2_mf):
        """When nroots >= n_det, SCI path falls back to explicit H."""
        config = ORMASConfig(
            subspaces=[
                Subspace("s1", [0, 1], min_electrons=2, max_electrons=4),
                Subspace("s2", [2, 3], min_electrons=0, max_electrons=2),
            ],
            n_active_orbitals=4,
            nelecas=(2, 2),
        )
        # This config gives a small number of determinants
        mc = mcscf.CASCI(n2_mf, 4, 4)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.direct_ci_threshold = 1  # Force SCI path attempt
        n_det_estimate = 27  # approximate
        mc.fcisolver.nroots = n_det_estimate + 10  # More roots than dets
        mc.kernel()
        # Should not crash — falls back to explicit H for full diag
        assert mc.fcisolver.converged

    def test_small_system_sci_path(self, n2_mf):
        """Very small system forced through SCI path still works."""
        config = ORMASConfig(
            subspaces=[Subspace("full", [0, 1], 0, 4)],
            n_active_orbitals=2,
            nelecas=(1, 1),
        )
        mc = mcscf.CASCI(n2_mf, 2, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.direct_ci_threshold = 1  # Force SCI
        mc.kernel()
        assert mc.fcisolver.converged
        assert mc.fcisolver._use_sci

    def test_python_fallback_rdm_matches_sci(self, n2_mf):
        """Python RDM path (SCI disabled) matches SCI RDM path."""
        config = ORMASConfig(
            subspaces=[Subspace("full", list(range(6)), 0, 12)],
            n_active_orbitals=6,
            nelecas=(3, 3),
        )
        mc = mcscf.CASCI(n2_mf, 6, 6)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        ci = mc.ci
        ncas = 6
        nelecas = (3, 3)
        solver = mc.fcisolver

        # Get SCI results
        rdm1_sci = solver.make_rdm1(ci, ncas, nelecas)
        rdm1a_sci, rdm1b_sci = solver.make_rdm1s(ci, ncas, nelecas)
        rdm1_12_sci, rdm2_sci = solver.make_rdm12(ci, ncas, nelecas)
        rdm12s_sci = solver.make_rdm12s(ci, ncas, nelecas)
        ss_sci, mult_sci = solver.spin_square(ci, ncas, nelecas)
        hdiag_sci = solver.make_hdiag(
            solver._h1e, solver._eri, ncas, nelecas,
        )

        # Disable SCI to force Python fallback
        saved_link_index = solver._sci_link_index
        solver._sci_link_index = None
        assert not solver._use_sci

        rdm1_py = solver.make_rdm1(ci, ncas, nelecas)
        rdm1a_py, rdm1b_py = solver.make_rdm1s(ci, ncas, nelecas)
        rdm1_12_py, rdm2_py = solver.make_rdm12(ci, ncas, nelecas)
        rdm12s_py = solver.make_rdm12s(ci, ncas, nelecas)
        ss_py, mult_py = solver.spin_square(ci, ncas, nelecas)
        hdiag_py = solver.make_hdiag(
            solver._h1e, solver._eri, ncas, nelecas,
        )

        # Restore
        solver._sci_link_index = saved_link_index

        assert np.allclose(rdm1_sci, rdm1_py, atol=1e-12)
        assert np.allclose(rdm1a_sci, rdm1a_py, atol=1e-12)
        assert np.allclose(rdm1b_sci, rdm1b_py, atol=1e-12)
        assert np.allclose(rdm1_12_sci, rdm1_12_py, atol=1e-12)
        assert np.allclose(rdm2_sci, rdm2_py, atol=1e-12)
        assert np.allclose(rdm12s_sci[0][0], rdm12s_py[0][0], atol=1e-12)
        assert np.allclose(rdm12s_sci[1][0], rdm12s_py[1][0], atol=1e-12)
        assert np.allclose(rdm12s_sci[1][1], rdm12s_py[1][1], atol=1e-12)
        assert np.allclose(rdm12s_sci[1][2], rdm12s_py[1][2], atol=1e-12)
        assert abs(ss_sci - ss_py) < 1e-10
        assert abs(mult_sci - mult_py) < 1e-10
        assert np.allclose(hdiag_sci, hdiag_py, atol=1e-10)

    def test_ormas_restricted_sci_matches_pyscf(self, n2_mf):
        """ORMAS-restricted space via SCI gives variational energy."""
        # Full CAS reference
        mc_ref = mcscf.CASCI(n2_mf, 6, 6)
        mc_ref.verbose = 0
        mc_ref.kernel()

        # ORMAS restricted
        config = ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
                Subspace("pi", [3, 4, 5], min_electrons=0, max_electrons=4),
            ],
            n_active_orbitals=6,
            nelecas=(3, 3),
        )
        mc = mcscf.CASCI(n2_mf, 6, 6)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        # ORMAS energy should be >= CASCI (variational principle)
        assert mc.e_tot >= mc_ref.e_tot - 1e-10
        # But close (within a few mHa for reasonable restrictions)
        assert abs(mc.e_tot - mc_ref.e_tot) < 0.01  # 10 mHa
