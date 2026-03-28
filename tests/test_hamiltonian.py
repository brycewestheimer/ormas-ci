"""Tests for ormas_ci.hamiltonian -- CI Hamiltonian construction."""

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from ormas_ci.hamiltonian import build_ci_hamiltonian
from ormas_ci.utils import generate_strings


def _get_h2_setup():
    """Get H2/STO-3G integrals and determinant strings.

    Returns:
        (alpha_strings, beta_strings, h1e, h2e, e_fci)
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    h2e = ao2mo.restore(1, h2e, 2)

    e_fci, _ = fci.direct_spin1.kernel(h1e, h2e, 2, 2)

    alpha_strings = generate_strings(2, 1)
    beta_strings = generate_strings(2, 1)

    # Flatten into determinant-pair arrays (alpha x beta)
    all_alpha = []
    all_beta = []
    for a in alpha_strings:
        for b in beta_strings:
            all_alpha.append(a)
            all_beta.append(b)

    return (
        np.array(all_alpha, dtype=np.int64),
        np.array(all_beta, dtype=np.int64),
        h1e,
        h2e,
        e_fci,
    )


def test_h2_hamiltonian_eigenvalues():
    """Full H2 Hamiltonian eigenvalues match PySCF FCI ground state."""
    alpha, beta, h1e, h2e, e_fci = _get_h2_setup()
    h_ci = build_ci_hamiltonian(alpha, beta, h1e, h2e)

    eigenvalues = np.sort(np.linalg.eigvalsh(h_ci))
    assert abs(eigenvalues[0] - e_fci) < 1e-10, (
        f"Ground state mismatch: {eigenvalues[0]} vs {e_fci}"
    )


def test_symmetry():
    """Hamiltonian matrix produced by build_ci_hamiltonian is symmetric."""
    alpha, beta, h1e, h2e, _ = _get_h2_setup()
    h_ci = build_ci_hamiltonian(alpha, beta, h1e, h2e)

    if hasattr(h_ci, "toarray"):
        h_dense = h_ci.toarray()
    else:
        h_dense = h_ci

    assert np.allclose(h_dense, h_dense.T, atol=1e-14), (
        "Hamiltonian is not symmetric"
    )


def test_sparse_dense_agree():
    """Sparse and dense construction paths produce the same matrix.

    Force sparse by setting sparse_threshold=0, and force dense by setting
    sparse_threshold=999999.
    """
    alpha, beta, h1e, h2e, _ = _get_h2_setup()

    h_dense = build_ci_hamiltonian(
        alpha, beta, h1e, h2e, sparse_threshold=999999,
    )
    h_sparse = build_ci_hamiltonian(
        alpha, beta, h1e, h2e, sparse_threshold=0,
    )

    # Convert sparse to dense for comparison
    h_sparse_dense = h_sparse.toarray()

    assert np.allclose(h_dense, h_sparse_dense, atol=1e-14), (
        "Sparse and dense Hamiltonians disagree"
    )


def test_hamiltonian_dimensions():
    """Hamiltonian has shape (n_det, n_det)."""
    alpha, beta, h1e, h2e, _ = _get_h2_setup()
    h_ci = build_ci_hamiltonian(alpha, beta, h1e, h2e)

    n_det = len(alpha)
    if hasattr(h_ci, "shape"):
        assert h_ci.shape == (n_det, n_det), (
            f"Expected shape ({n_det}, {n_det}), got {h_ci.shape}"
        )


def test_sparse_output_type():
    """When sparse_threshold=0, output should be a sparse matrix."""
    alpha, beta, h1e, h2e, _ = _get_h2_setup()
    h_ci = build_ci_hamiltonian(
        alpha, beta, h1e, h2e, sparse_threshold=0,
    )

    import scipy.sparse as sp

    assert sp.issparse(h_ci), "Expected sparse matrix when sparse_threshold=0"
