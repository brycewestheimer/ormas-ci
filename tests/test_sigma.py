"""Tests for ormas_ci.sigma -- excitation matrix and sigma vector engine.

Direct tests for _build_single_excitation_matrix() and SigmaEinsum,
covering diagonal occupancy, allowed/forbidden excitations, phase signs,
and agreement with the explicit Hamiltonian path.
"""

import numpy as np
import pytest

from pyscf import ao2mo, gto, mcscf, scf
from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
from pyscf.ormas_ci.sigma import SigmaEinsum, _build_single_excitation_matrix
from pyscf.ormas_ci.utils import compute_phase, generate_strings

# ---------------------------------------------------------------------------
# _build_single_excitation_matrix tests
# ---------------------------------------------------------------------------


class TestExcitationMatrixDiagonal:
    """E[p,p,a,a] should be 1 if orbital p is occupied in string a, else 0."""

    def test_two_orbital_single_electron(self):
        """2 orbitals, strings |01> and |10>."""
        strings = np.array([0b01, 0b10], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        # String 0b01: orbital 0 occupied, orbital 1 empty
        assert e_mat[0, 0, 0, 0] == 1.0  # orbital 0 occupied in string 0
        assert e_mat[1, 1, 0, 0] == 0.0  # orbital 1 NOT occupied in string 0

        # String 0b10: orbital 1 occupied, orbital 0 empty
        assert e_mat[0, 0, 1, 1] == 0.0
        assert e_mat[1, 1, 1, 1] == 1.0

    def test_fully_occupied(self):
        """Both orbitals occupied: |11>."""
        strings = np.array([0b11], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        assert e_mat[0, 0, 0, 0] == 1.0
        assert e_mat[1, 1, 0, 0] == 1.0

    def test_three_orbital_mixed(self):
        """3 orbitals, string |101>: orbitals 0 and 2 occupied."""
        strings = np.array([0b101], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=3)

        assert e_mat[0, 0, 0, 0] == 1.0  # orbital 0 occupied
        assert e_mat[1, 1, 0, 0] == 0.0  # orbital 1 empty
        assert e_mat[2, 2, 0, 0] == 1.0  # orbital 2 occupied


class TestExcitationMatrixOffDiagonal:
    """Test allowed and forbidden single excitations."""

    def test_allowed_excitation_0_to_1(self):
        """Excitation from orbital 0 to 1 in {|01>, |10>}."""
        strings = np.array([0b01, 0b10], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        # Annihilate orbital 0 from |01>, create at orbital 1 -> |10>
        # E[p=1, q=0, a'=1, a=0] should be nonzero with correct phase
        phase = compute_phase(0b01, 1, 0)
        assert e_mat[1, 0, 1, 0] == phase

    def test_allowed_excitation_1_to_0(self):
        """Reverse excitation: orbital 1 to 0 in {|01>, |10>}."""
        strings = np.array([0b01, 0b10], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        phase = compute_phase(0b10, 0, 1)
        assert e_mat[0, 1, 0, 1] == phase

    def test_forbidden_excitation_target_occupied(self):
        """Cannot excite into already-occupied orbital."""
        # |11> -> both occupied, no single excitations possible within same string
        strings = np.array([0b11], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        # Off-diagonal elements should all be zero (only one string)
        assert e_mat[1, 0, 0, 0] == 0.0
        assert e_mat[0, 1, 0, 0] == 0.0

    def test_forbidden_excitation_source_empty(self):
        """Cannot annihilate from unoccupied orbital."""
        strings = np.array([0b01, 0b10], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        # Try to annihilate orbital 1 from |01> where orbital 1 is empty
        # E[p, q=1, *, a=0] for any p should be 0
        assert e_mat[0, 1, 0, 0] == 0.0
        assert e_mat[0, 1, 1, 0] == 0.0


class TestExcitationMatrixPhases:
    """Phase sign verification for 3-orbital systems."""

    def test_phase_with_intervening_occupied(self):
        """Phase = -1 when one orbital lies between source and target.

        String |111> (0b111): excite orbital 0 -> orbital 2.
        Orbital 1 is between them and occupied, so phase = -1.
        """
        # Need strings that include both |111> and the result of 0->2 excitation
        # |111> with orbital 0 annihilated and 2 created... orbital 2 is already occupied
        # Use |011> instead: excite orbital 0 -> orbital 2
        # |011> (0b011) -> annihilate 0, create 2 -> |110> (0b110)
        strings = np.array([0b011, 0b110], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=3)

        # Phase for excitation 0 -> 2 in |011>
        # Intervening orbital 1 is occupied -> phase = -1
        expected_phase = compute_phase(0b011, 2, 0)
        assert expected_phase == -1  # sanity check
        assert e_mat[2, 0, 1, 0] == -1.0

    def test_phase_no_intervening(self):
        """Phase = +1 when no orbitals lie between source and target.

        String |001> (0b001): excite orbital 0 -> orbital 1.
        No orbitals between them, so phase = +1.
        """
        strings = np.array([0b001, 0b010], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=3)

        expected_phase = compute_phase(0b001, 1, 0)
        assert expected_phase == 1
        assert e_mat[1, 0, 1, 0] == 1.0

    def test_three_orbital_all_single_electron_strings(self):
        """All 1-electron strings in 3 orbitals: |001>, |010>, |100>."""
        strings = np.array([0b001, 0b010, 0b100], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=3)

        # Verify all allowed excitations have correct phases
        # |001> -> |010>: excite 0 -> 1, no intervening, phase = +1
        assert e_mat[1, 0, 1, 0] == compute_phase(0b001, 1, 0)

        # |001> -> |100>: excite 0 -> 2, no intervening, phase = +1
        assert e_mat[2, 0, 2, 0] == compute_phase(0b001, 2, 0)

        # |010> -> |001>: excite 1 -> 0, no intervening, phase = +1
        assert e_mat[0, 1, 0, 1] == compute_phase(0b010, 0, 1)

        # |010> -> |100>: excite 1 -> 2, no intervening, phase = +1
        assert e_mat[2, 1, 2, 1] == compute_phase(0b010, 2, 1)

        # |100> -> |001>: excite 2 -> 0, no intervening, phase = +1
        assert e_mat[0, 2, 0, 2] == compute_phase(0b100, 0, 2)

        # |100> -> |010>: excite 2 -> 1, no intervening, phase = +1
        assert e_mat[1, 2, 1, 2] == compute_phase(0b100, 1, 2)


class TestExcitationMatrixSymmetry:
    """For every nonzero E[p,q,a',a], the reverse E[q,p,a,a'] should exist."""

    def test_reverse_excitation_exists(self):
        """If a -> a' via q->p, then a' -> a via p->q should also be nonzero."""
        strings = np.array([0b01, 0b10, 0b11], dtype=np.int64)
        e_mat = _build_single_excitation_matrix(strings, norb=2)

        norb = 2
        n_str = len(strings)
        for p in range(norb):
            for q in range(norb):
                if p == q:
                    continue
                for a_prime in range(n_str):
                    for a in range(n_str):
                        if e_mat[p, q, a_prime, a] != 0:
                            # Reverse must also be nonzero
                            assert e_mat[q, p, a, a_prime] != 0, (
                                f"E[{p},{q},{a_prime},{a}]={e_mat[p, q, a_prime, a]} "
                                f"but reverse E[{q},{p},{a},{a_prime}]=0"
                            )


# ---------------------------------------------------------------------------
# SigmaEinsum vs explicit Hamiltonian
# ---------------------------------------------------------------------------


class TestSigmaVsExplicitHamiltonian:
    """SigmaEinsum.sigma(ci) must agree with H @ ci from explicit Hamiltonian."""

    @pytest.fixture
    def h2_integrals(self):
        """H2/6-31G active-space integrals."""
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.run()
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        h1e, ecore = mc.get_h1eff()
        h2e = mc.get_h2eff()
        h2e = ao2mo.restore(1, h2e, 2)
        return h1e, h2e

    def test_sigma_matches_explicit_h_times_ci(self, h2_integrals):
        """For H2, sigma(ci) == H @ ci for all unit vectors."""
        h1e, h2e = h2_integrals
        ncas = 2
        nelec = (1, 1)

        alpha_strings = generate_strings(ncas, nelec[0])
        beta_strings = generate_strings(ncas, nelec[1])

        # Build explicit determinant arrays (alpha x beta product)
        all_alpha = []
        all_beta = []
        for a in alpha_strings:
            for b in beta_strings:
                all_alpha.append(a)
                all_beta.append(b)
        all_alpha = np.array(all_alpha, dtype=np.int64)
        all_beta = np.array(all_beta, dtype=np.int64)

        # Build explicit Hamiltonian
        h_ci = build_ci_hamiltonian(all_alpha, all_beta, h1e, h2e)
        if hasattr(h_ci, "toarray"):
            h_ci = h_ci.toarray()

        # Build SigmaEinsum
        unique_alpha = np.array(alpha_strings, dtype=np.int64)
        unique_beta = np.array(beta_strings, dtype=np.int64)
        sigma_engine = SigmaEinsum(unique_alpha, unique_beta, h1e, h2e, nelec)

        n_det = len(all_alpha)
        n_ua = len(unique_alpha)
        n_ub = len(unique_beta)

        # Build 1D -> 2D mapping
        a_map = {int(v): i for i, v in enumerate(unique_alpha)}
        b_map = {int(v): i for i, v in enumerate(unique_beta)}
        det_row = np.array([a_map[int(s)] for s in all_alpha])
        det_col = np.array([b_map[int(s)] for s in all_beta])

        # Test each unit vector
        for k in range(n_det):
            ci_1d = np.zeros(n_det)
            ci_1d[k] = 1.0

            # Explicit path
            expected = h_ci @ ci_1d

            # SigmaEinsum path
            ci_2d = np.zeros((n_ua, n_ub))
            ci_2d[det_row, det_col] = ci_1d
            sigma_2d = sigma_engine.sigma(ci_2d)
            actual = sigma_2d[det_row, det_col]

            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-12,
                err_msg=f"Sigma mismatch for unit vector {k}",
            )

    def test_sigma_random_vector(self, h2_integrals):
        """sigma(random_ci) matches H @ random_ci."""
        h1e, h2e = h2_integrals
        ncas = 2
        nelec = (1, 1)

        alpha_strings = generate_strings(ncas, nelec[0])
        beta_strings = generate_strings(ncas, nelec[1])

        all_alpha = []
        all_beta = []
        for a in alpha_strings:
            for b in beta_strings:
                all_alpha.append(a)
                all_beta.append(b)
        all_alpha = np.array(all_alpha, dtype=np.int64)
        all_beta = np.array(all_beta, dtype=np.int64)

        h_ci = build_ci_hamiltonian(all_alpha, all_beta, h1e, h2e)
        if hasattr(h_ci, "toarray"):
            h_ci = h_ci.toarray()

        unique_alpha = np.array(alpha_strings, dtype=np.int64)
        unique_beta = np.array(beta_strings, dtype=np.int64)
        sigma_engine = SigmaEinsum(unique_alpha, unique_beta, h1e, h2e, nelec)

        n_det = len(all_alpha)
        n_ua = len(unique_alpha)
        n_ub = len(unique_beta)

        a_map = {int(v): i for i, v in enumerate(unique_alpha)}
        b_map = {int(v): i for i, v in enumerate(unique_beta)}
        det_row = np.array([a_map[int(s)] for s in all_alpha])
        det_col = np.array([b_map[int(s)] for s in all_beta])

        rng = np.random.default_rng(42)
        ci_1d = rng.standard_normal(n_det)
        ci_1d /= np.linalg.norm(ci_1d)

        expected = h_ci @ ci_1d

        ci_2d = np.zeros((n_ua, n_ub))
        ci_2d[det_row, det_col] = ci_1d
        sigma_2d = sigma_engine.sigma(ci_2d)
        actual = sigma_2d[det_row, det_col]

        np.testing.assert_allclose(actual, expected, atol=1e-12)


class TestSigmaMemoryEstimate:
    """memory_estimate_mb returns a positive value."""

    def test_positive_estimate(self):
        strings = np.array([0b01, 0b10], dtype=np.int64)
        norb = 2
        h1e = np.zeros((norb, norb))
        eri = np.zeros((norb, norb, norb, norb))
        engine = SigmaEinsum(strings, strings, h1e, eri, (1, 1))
        assert engine.memory_estimate_mb() > 0
