"""
CI Hamiltonian diagonalization.

Uses numpy.linalg.eigh for dense matrices and scipy.sparse.linalg.eigsh
(Lanczos) for sparse matrices. Returns eigenvalues sorted ascending.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["solve_ci"]


def solve_ci(
    hamiltonian: np.ndarray | sp.spmatrix,
    n_roots: int = 1,
    ci0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the CI Hamiltonian.

    Args:
        hamiltonian: CI matrix, dense or sparse.
        n_roots: Number of lowest eigenstates to compute.
        ci0: Initial guess for iterative solver, shape (n_det,) or (n_det, n_roots).
        tol: Convergence tolerance for the iterative solver.
        max_iter: Maximum iterations for the iterative solver.

    Returns:
        (energies, ci_vectors) where energies has shape (n_roots,)
        and ci_vectors has shape (n_det, n_roots).
    """
    if sp.issparse(hamiltonian):
        n_det = hamiltonian.shape[0]

        # scipy.sparse.linalg.eigsh (ARPACK Lanczos) requires k < n - 1
        # where n is the matrix dimension. For small CI spaces or when
        # requesting many roots, fall back to dense numpy.linalg.eigh
        # which has no such restriction.
        if n_roots >= n_det - 1:
            hamiltonian = hamiltonian.toarray()  # type: ignore[union-attr]
        else:
            v0 = None
            if ci0 is not None:
                v0 = ci0[:, 0] if ci0.ndim == 2 else ci0

            energies, vectors = spla.eigsh(
                hamiltonian,
                k=n_roots,
                which="SA",
                v0=v0,
                tol=tol,  # type: ignore[arg-type]
                maxiter=max_iter,
            )
            idx = np.argsort(energies)
            return energies[idx], vectors[:, idx]

    # Dense diagonalization
    energies, vectors = np.linalg.eigh(hamiltonian)  # type: ignore[arg-type]
    return energies[:n_roots], vectors[:, :n_roots]
