"""Tests for ormas_ci.solver -- CI Hamiltonian diagonalization."""

import numpy as np
import scipy.sparse as sp

from ormas_ci.solver import solve_ci


def test_known_eigenvalues():
    """Diagonal matrix: eigenvalues are the diagonal elements."""
    h_mat = np.diag([1.0, 2.0, 3.0])
    energies, vectors = solve_ci(h_mat, n_roots=2)

    assert np.allclose(energies, [1.0, 2.0]), (
        f"Expected [1.0, 2.0], got {energies}"
    )


def test_eigenvectors_orthonormal():
    """Eigenvectors from a symmetric matrix should be orthonormal."""
    # Build a known symmetric matrix with distinct eigenvalues
    rng = np.random.default_rng(42)
    q_mat, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    eigenvalues = np.arange(1.0, 11.0)
    h_mat = q_mat @ np.diag(eigenvalues) @ q_mat.T

    n_roots = 5
    _, vectors = solve_ci(h_mat, n_roots=n_roots)

    # Check orthonormality: V^T V should be identity
    overlap = vectors.T @ vectors
    assert np.allclose(overlap, np.eye(n_roots), atol=1e-12), (
        f"Eigenvectors not orthonormal: max deviation {np.max(np.abs(overlap - np.eye(n_roots)))}"
    )


def test_sparse_solver():
    """Sparse input gives same eigenvalues as dense."""
    h_dense = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    h_sparse = sp.csr_matrix(h_dense)

    n_roots = 3
    e_dense, _ = solve_ci(h_dense, n_roots=n_roots)
    e_sparse, _ = solve_ci(h_sparse, n_roots=n_roots)

    assert np.allclose(e_dense, e_sparse, atol=1e-10), (
        f"Dense vs sparse mismatch: {e_dense} vs {e_sparse}"
    )


def test_single_root():
    """Requesting a single root returns the ground state."""
    h_mat = np.diag([5.0, 1.0, 3.0])
    energies, vectors = solve_ci(h_mat, n_roots=1)

    assert len(energies) == 1
    assert abs(energies[0] - 1.0) < 1e-12


def test_eigenvectors_shape():
    """Eigenvectors have shape (n_det, n_roots)."""
    n_det = 8
    h_mat = np.diag(np.arange(1.0, n_det + 1.0))
    n_roots = 3
    _, vectors = solve_ci(h_mat, n_roots=n_roots)

    assert vectors.shape == (n_det, n_roots), (
        f"Expected shape ({n_det}, {n_roots}), got {vectors.shape}"
    )


def test_eigenvalues_sorted_ascending():
    """Returned eigenvalues should be in ascending order."""
    rng = np.random.default_rng(123)
    q_mat, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    evals = np.array([10.0, 2.0, 8.0, 4.0, 6.0, 1.0])
    h_mat = q_mat @ np.diag(evals) @ q_mat.T

    energies, _ = solve_ci(h_mat, n_roots=4)

    for i in range(len(energies) - 1):
        assert energies[i] <= energies[i + 1] + 1e-14, (
            f"Eigenvalues not sorted: {energies}"
        )
