"""Tests for ormas_ci.slater_condon -- Slater-Condon matrix element evaluation.

The critical test is test_h2_full_hamiltonian_matches_pyscf, which builds the
full CI Hamiltonian for H2/STO-3G using matrix_element() and verifies that
eigenvalues match PySCF's FCI solver.
"""

import numpy as np

from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.ormas_ci.slater_condon import matrix_element
from pyscf.ormas_ci.utils import generate_strings


def _get_h2_integrals():
    """Get H2/STO-3G active space integrals from PySCF.

    Returns:
        (h1e, h2e, ecore, e_fci) where e_fci is the PySCF FCI ground state energy.
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    h2e = ao2mo.restore(1, h2e, 2)  # Ensure 4-index array

    e_fci, _ = fci.direct_spin1.kernel(h1e, h2e, 2, 2)

    return h1e, h2e, ecore, e_fci


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


def test_h2_full_hamiltonian_matches_pyscf():
    """Build full H matrix for H2 and compare eigenvalues to PySCF FCI."""
    h1e, h2e, _, e_fci = _get_h2_integrals()
    h_mat, _ = _build_h2_hamiltonian(h1e, h2e)

    # Symmetry check
    assert np.allclose(h_mat, h_mat.T, atol=1e-14), "Hamiltonian not symmetric"

    # Compare ground state eigenvalue to PySCF FCI
    eigenvalues = np.sort(np.linalg.eigvalsh(h_mat))
    assert abs(eigenvalues[0] - e_fci) < 1e-10, (
        f"Ground state mismatch: {eigenvalues[0]} vs {e_fci}"
    )


def test_hermiticity():
    """H[i,j] == H[j,i] for all pairs, verified element-by-element."""
    h1e, h2e, _, _ = _get_h2_integrals()
    strings = generate_strings(2, 1)
    dets = [(a, b) for a in strings for b in strings]

    for ai, bi in dets:
        for aj, bj in dets:
            hij = matrix_element(ai, bi, aj, bj, h1e, h2e)
            hji = matrix_element(aj, bj, ai, bi, h1e, h2e)
            assert abs(hij - hji) < 1e-14, (
                f"H[{(ai,bi)},{(aj,bj)}]={hij} != H[{(aj,bj)},{(ai,bi)}]={hji}"
            )


def test_zero_for_triple_excitation():
    """Three or more orbital differences should give zero."""
    # In a 6-orbital space: alpha |000111> vs |111000> differ by 3 orbitals
    alpha_i = 0b000111  # orbitals 0,1,2
    alpha_j = 0b111000  # orbitals 3,4,5
    beta_i = 0b000111   # same beta
    beta_j = 0b000111

    h1e = np.eye(6)
    h2e = np.zeros((6, 6, 6, 6))

    result = matrix_element(alpha_i, beta_i, alpha_j, beta_j, h1e, h2e)
    assert result == 0.0, f"Expected 0.0 for triple excitation, got {result}"


def test_diagonal_elements_real():
    """Diagonal matrix elements should be real finite numbers."""
    h1e, h2e, _, _ = _get_h2_integrals()
    strings = generate_strings(2, 1)

    for a in strings:
        for b in strings:
            val = matrix_element(a, b, a, b, h1e, h2e)
            assert np.isfinite(val), f"Non-finite diagonal element for ({a}, {b})"


def test_h2_eigenvalue_count():
    """H2/STO-3G in (2,2) space has exactly 4 determinants and 4 eigenvalues."""
    h1e, h2e, _, _ = _get_h2_integrals()
    h_mat, det_list = _build_h2_hamiltonian(h1e, h2e)

    assert len(det_list) == 4, f"Expected 4 determinants, got {len(det_list)}"
    eigenvalues = np.linalg.eigvalsh(h_mat)
    assert len(eigenvalues) == 4, f"Expected 4 eigenvalues, got {len(eigenvalues)}"
