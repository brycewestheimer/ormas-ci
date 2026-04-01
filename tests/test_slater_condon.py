"""Tests for ormas_ci.slater_condon -- Slater-Condon matrix element evaluation.

The critical test is test_h2_full_hamiltonian_matches_pyscf, which builds the
full CI Hamiltonian for H2/6-31G using matrix_element() and verifies that
eigenvalues match PySCF's FCI solver.
"""

import numpy as np

from pyscf.ormas_ci.slater_condon import (
    _compute_double,
    _compute_single,
    excitation_info,
    matrix_element,
)
from pyscf.ormas_ci.utils import compute_phase, generate_strings


def _build_h2_hamiltonian(h1e, h2e):
    """Build the full H2 CI Hamiltonian using matrix_element().

    Returns:
        (H, det_list) where H is the Hamiltonian matrix and det_list is a
        list of (alpha, beta) determinant pairs.
    """
    alpha_strings = generate_strings(2, 1)  # [0b01, 0b10]
    beta_strings = generate_strings(2, 1)

    det_list = [(a, b) for a in alpha_strings for b in beta_strings]
    n_det = len(det_list)
    h_mat = np.zeros((n_det, n_det))

    for i, (ai, bi) in enumerate(det_list):
        for j, (aj, bj) in enumerate(det_list):
            h_mat[i, j] = matrix_element(ai, bi, aj, bj, h1e, h2e)

    return h_mat, det_list


def test_h2_full_hamiltonian_matches_pyscf(h2_active_integrals):
    """Build full H matrix for H2 and compare eigenvalues to PySCF FCI."""
    h1e, h2e, _, e_fci = h2_active_integrals
    h_mat, _ = _build_h2_hamiltonian(h1e, h2e)

    # Symmetry check
    assert np.allclose(h_mat, h_mat.T, atol=1e-14), "Hamiltonian not symmetric"

    # Compare ground state eigenvalue to PySCF FCI
    eigenvalues = np.sort(np.linalg.eigvalsh(h_mat))
    assert abs(eigenvalues[0] - e_fci) < 1e-10, (
        f"Ground state mismatch: {eigenvalues[0]} vs {e_fci}"
    )


def test_hermiticity(h2_active_integrals):
    """H[i,j] == H[j,i] for all pairs, verified element-by-element."""
    h1e, h2e, _, _ = h2_active_integrals
    strings = generate_strings(2, 1)
    dets = [(a, b) for a in strings for b in strings]

    for ai, bi in dets:
        for aj, bj in dets:
            hij = matrix_element(ai, bi, aj, bj, h1e, h2e)
            hji = matrix_element(aj, bj, ai, bi, h1e, h2e)
            assert abs(hij - hji) < 1e-14, (
                f"H[{(ai, bi)},{(aj, bj)}]={hij} != H[{(aj, bj)},{(ai, bi)}]={hji}"
            )


def test_zero_for_triple_excitation():
    """Three or more orbital differences should give zero."""
    # In a 6-orbital space: alpha |000111> vs |111000> differ by 3 orbitals
    alpha_i = 0b000111  # orbitals 0,1,2
    alpha_j = 0b111000  # orbitals 3,4,5
    beta_i = 0b000111  # same beta
    beta_j = 0b000111

    h1e = np.eye(6)
    h2e = np.zeros((6, 6, 6, 6))

    result = matrix_element(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)
    assert result == 0.0, f"Expected 0.0 for triple excitation, got {result}"


def test_diagonal_elements_real(h2_active_integrals):
    """Diagonal matrix elements should be real finite numbers."""
    h1e, h2e, _, _ = h2_active_integrals
    strings = generate_strings(2, 1)

    for a in strings:
        for b in strings:
            val = matrix_element(a, b, a, b, h1e, h2e)
            assert np.isfinite(val), f"Non-finite diagonal element for ({a}, {b})"


def test_h2_eigenvalue_count(h2_active_integrals):
    """H2/6-31G in (2,2) space has exactly 4 determinants and 4 eigenvalues."""
    h1e, h2e, _, _ = h2_active_integrals
    h_mat, det_list = _build_h2_hamiltonian(h1e, h2e)

    assert len(det_list) == 4, f"Expected 4 determinants, got {len(det_list)}"
    eigenvalues = np.linalg.eigvalsh(h_mat)
    assert len(eigenvalues) == 4, f"Expected 4 eigenvalues, got {len(eigenvalues)}"


# ---------------------------------------------------------------------------
# excitation_info() tests
# ---------------------------------------------------------------------------


class TestExcitationInfo:
    """Direct tests for excitation_info() helper."""

    def test_identical_strings(self):
        """Same string -> 0 excitations, no holes/particles."""
        n, holes, particles = excitation_info(0b0101, 0b0101)
        assert n == 0
        assert holes == ()
        assert particles == ()

    def test_single_excitation(self):
        """One orbital differs -> 1 excitation."""
        # |01> -> |10>: hole at 0, particle at 1
        n, holes, particles = excitation_info(0b01, 0b10)
        assert n == 1
        assert holes == (0,)
        assert particles == (1,)

    def test_double_excitation(self):
        """Two orbitals differ -> 2 excitations."""
        # |0011> -> |1100>: holes at 0,1; particles at 2,3
        n, holes, particles = excitation_info(0b0011, 0b1100)
        assert n == 2
        assert holes == (0, 1)
        assert particles == (2, 3)

    def test_triple_excitation(self):
        """Three orbitals differ -> 3 excitations."""
        n, holes, particles = excitation_info(0b000111, 0b111000)
        assert n == 3
        assert holes == (0, 1, 2)
        assert particles == (3, 4, 5)

    def test_partial_overlap(self):
        """Some orbitals shared, one differs."""
        # |011> vs |101>: both have orbital 0 (no, wait)
        # 0b011 = orbitals 0,1 occupied; 0b101 = orbitals 0,2 occupied
        n, holes, particles = excitation_info(0b011, 0b101)
        assert n == 1
        assert holes == (1,)
        assert particles == (2,)


# ---------------------------------------------------------------------------
# _compute_single() tests
# ---------------------------------------------------------------------------


class TestComputeSingle:
    """Direct tests for _compute_single() with hand-checkable integrals."""

    def test_alpha_single_one_electron_only(self):
        """Alpha single excitation with zero 2e integrals -> pure h1e[p,q]."""
        # 3 orbitals, 1 alpha electron, 1 beta electron
        # Alpha: |001> -> |010>  (excite orbital 0 -> 1)
        # Beta: |001> unchanged
        h1e = np.zeros((3, 3))
        h1e[0, 1] = 0.5  # <0|h|1>
        h1e[1, 0] = 0.5  # symmetric
        h2e = np.zeros((3, 3, 3, 3))

        alpha_i, beta_i = 0b001, 0b001
        alpha_j, beta_j = 0b010, 0b001

        result = _compute_single(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)
        phase = compute_phase(0b001, 0, 1)  # hole=0, particle=1
        expected = phase * h1e[0, 1]
        assert abs(result - expected) < 1e-14

    def test_beta_single_one_electron_only(self):
        """Beta single excitation with zero 2e integrals -> pure h1e[p,q]."""
        h1e = np.zeros((3, 3))
        h1e[0, 1] = 0.7
        h1e[1, 0] = 0.7
        h2e = np.zeros((3, 3, 3, 3))

        alpha_i, beta_i = 0b001, 0b001
        alpha_j, beta_j = 0b001, 0b010

        result = _compute_single(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)
        phase = compute_phase(0b001, 0, 1)
        expected = phase * h1e[0, 1]
        assert abs(result - expected) < 1e-14

    def test_alpha_single_with_coulomb(self):
        """Alpha single excitation includes Coulomb with beta electrons."""
        # Alpha: |01> -> |10>, Beta: |01> (orbital 0 occupied)
        h1e = np.zeros((2, 2))
        h1e[0, 1] = 0.3
        h1e[1, 0] = 0.3
        h2e = np.zeros((2, 2, 2, 2))
        h2e[0, 1, 0, 0] = 0.1  # J[p=0, q=1, r=0] where r is beta occupied

        alpha_i, beta_i = 0b01, 0b01
        alpha_j, beta_j = 0b10, 0b01

        result = _compute_single(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)
        phase = compute_phase(0b01, 0, 1)
        # hole=0, particle=1; no other alpha electrons; beta electron at orbital 0
        expected = phase * (h1e[0, 1] + h2e[0, 1, 0, 0])
        assert abs(result - expected) < 1e-14

    def test_no_excitation_returns_zero(self):
        """If neither spin channel has a single excitation, returns 0."""
        h1e = np.eye(2)
        h2e = np.zeros((2, 2, 2, 2))
        # Same determinant -> _compute_single should return 0
        result = _compute_single(0b01, 0b01, 0b01, 0b01, h1e, h2e)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _compute_double() tests
# ---------------------------------------------------------------------------


class TestComputeDouble:
    """Direct tests for _compute_double() with hand-checkable cases."""

    def test_alpha_alpha_double(self):
        """Same-spin alpha-alpha double with known ERI values.

        Alpha: |0011> -> |1100> (holes at 0,1; particles at 2,3)
        Beta unchanged.
        Slater-Condon: sign * [(p,r|q,s) - (p,s|q,r)]
        """
        h1e = np.zeros((4, 4))
        h2e = np.zeros((4, 4, 4, 4))
        h2e[0, 2, 1, 3] = 0.25  # (p=0,r=2|q=1,s=3) Coulomb
        h2e[0, 3, 1, 2] = 0.10  # (p=0,s=3|q=1,r=2) exchange

        alpha_i, beta_i = 0b0011, 0b0001
        alpha_j, beta_j = 0b1100, 0b0001

        result = _compute_double(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)

        # Compute expected phase manually
        # First excitation: hole=0 -> particle=2 on |0011>
        sign1 = compute_phase(0b0011, 0, 2)
        # Intermediate: |0011> ^ (1<<0) | (1<<2) = |0110>... wait:
        # 0b0011 XOR (1<<0) = 0b0010, OR (1<<2) = 0b0110
        alpha_mid = (0b0011 ^ (1 << 0)) | (1 << 2)
        # Second excitation: hole=1 -> particle=3 on intermediate
        sign2 = compute_phase(alpha_mid, 1, 3)
        sign = sign1 * sign2

        expected = sign * (h2e[0, 2, 1, 3] - h2e[0, 3, 1, 2])
        assert abs(result - expected) < 1e-14

    def test_beta_beta_double(self):
        """Same-spin beta-beta double, symmetric to alpha-alpha."""
        h1e = np.zeros((4, 4))
        h2e = np.zeros((4, 4, 4, 4))
        h2e[0, 2, 1, 3] = 0.25
        h2e[0, 3, 1, 2] = 0.10

        alpha_i, beta_i = 0b0001, 0b0011
        alpha_j, beta_j = 0b0001, 0b1100

        result = _compute_double(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)

        sign1 = compute_phase(0b0011, 0, 2)
        beta_mid = (0b0011 ^ (1 << 0)) | (1 << 2)
        sign2 = compute_phase(beta_mid, 1, 3)
        sign = sign1 * sign2

        expected = sign * (h2e[0, 2, 1, 3] - h2e[0, 3, 1, 2])
        assert abs(result - expected) < 1e-14

    def test_alpha_beta_double_no_exchange(self):
        """Opposite-spin alpha-beta double: Coulomb only, no exchange."""
        h1e = np.zeros((3, 3))
        h2e = np.zeros((3, 3, 3, 3))
        h2e[0, 1, 0, 2] = 0.4  # (p,r|q,s) Coulomb

        # Alpha: |01> -> |10> (hole=0, particle=1)
        # Beta: |01> -> |10> (hole=0, particle=2... no, let me be precise)
        # Alpha: |001> -> |010> (hole=0, particle=1)
        # Beta: |001> -> |100> (hole=0, particle=2)
        alpha_i, beta_i = 0b001, 0b001
        alpha_j, beta_j = 0b010, 0b100

        result = _compute_double(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)

        sign_a = compute_phase(0b001, 0, 1)  # alpha hole=0, particle=1
        sign_b = compute_phase(0b001, 0, 2)  # beta hole=0, particle=2

        expected = sign_a * sign_b * h2e[0, 1, 0, 2]
        assert abs(result - expected) < 1e-14

    def test_intermediate_determinant_matters(self):
        """Verify the intermediate determinant changes the phase.

        This test constructs a case where using the original string for
        the second phase would give the wrong sign.

        Alpha: |00111> -> |11100> (holes at 0,1,2 — wait that's triple)
        Let me use: |0111> -> |1110> only differs in orbitals 0 and 3
        That's a single excitation. For a double:

        |0011> -> |1100>: holes at 0,1; particles at 2,3
        Phase for (0->2): count occupied between 0 and 2 in |0011> = 1 (orbital 1)
        -> sign1 = -1
        Intermediate = |0110> (after removing 0, adding 2)
        Phase for (1->3): count occupied between 1 and 3 in |0110> = 1 (orbital 2)
        -> sign2 = -1
        Total = (-1)*(-1) = +1

        If we incorrectly used original |0011> for second phase:
        count occupied between 1 and 3 in |0011> = 0
        -> wrong_sign2 = +1
        wrong_total = (-1)*(+1) = -1  <-- WRONG SIGN
        """
        h1e = np.zeros((4, 4))
        h2e = np.zeros((4, 4, 4, 4))
        h2e[0, 2, 1, 3] = 1.0

        alpha_i = 0b0011  # orbitals 0,1 occupied
        alpha_j = 0b1100  # orbitals 2,3 occupied
        beta_i = beta_j = 0b0001

        result = _compute_double(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)

        # Correct computation using intermediate
        sign1 = compute_phase(0b0011, 0, 2)  # -1 (orbital 1 between)
        alpha_mid = (0b0011 ^ (1 << 0)) | (1 << 2)  # |0110>
        sign2 = compute_phase(alpha_mid, 1, 3)  # -1 (orbital 2 between)
        correct_sign = sign1 * sign2  # +1

        # Wrong computation using original for second phase
        wrong_sign2 = compute_phase(0b0011, 1, 3)  # +1 (nothing between in original)
        wrong_sign = sign1 * wrong_sign2  # -1

        assert correct_sign != wrong_sign, "Test setup error: phases should differ"
        expected = correct_sign * h2e[0, 2, 1, 3]
        assert abs(result - expected) < 1e-14, (
            f"Result {result} != expected {expected}; "
            f"intermediate determinant may not be used for second phase"
        )
