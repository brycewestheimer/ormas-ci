"""
Davidson-Liu iterative eigensolver for CI problems.

Pure-Python/NumPy implementation of the Davidson algorithm with diagonal
preconditioning. Designed as a drop-in replacement for scipy.sparse.linalg.eigsh
in the ORMAS-CI solver, with better convergence properties for CI Hamiltonians.

References:
    E.R. Davidson, J. Comput. Phys. 17, 87 (1975).
    G.L.G. Sleijpen & H.A. Van der Vorst, SIAM J. Matrix Anal. Appl. 17, 401 (1996).
"""

import warnings

import numpy as np

__all__ = ["davidson"]


def davidson(
    aop,
    x0,
    precond,
    tol=1e-12,
    max_cycle=100,
    max_space=12,
    nroots=1,
    lindep=1e-14,
    verbose=0,
):
    """Davidson-Liu eigensolver for the lowest eigenvalues of a symmetric operator.

    Args:
        aop: Callable that computes the matrix-vector product H @ x.
            Signature: aop(x) -> np.ndarray of same shape.
        x0: Initial guess vectors, shape (n,) for single root or (n, nroots)
            for multiple roots. Can also be a list of 1D arrays.
        precond: Callable for diagonal preconditioning.
            Signature: precond(residual, eigenvalue) -> np.ndarray.
        tol: Convergence tolerance on the residual norm.
        max_cycle: Maximum number of Davidson iterations.
        max_space: Maximum subspace dimension (per root) before restart.
            The subspace will grow up to max_space * nroots vectors.
        nroots: Number of lowest eigenvalues/vectors to compute.
        lindep: Threshold for detecting linear dependence when adding
            new vectors to the subspace.
        verbose: Print convergence info if > 0.

    Returns:
        (eigenvalues, eigenvectors, converged) where eigenvalues has shape
        (nroots,), eigenvectors has shape (n, nroots), and converged is a
        list of booleans indicating per-root convergence.
    """
    # Normalize initial guesses to a list of 1D vectors
    if isinstance(x0, np.ndarray):
        if x0.ndim == 1:
            x0 = [x0]
        else:
            x0 = [x0[:, i] for i in range(x0.shape[1])]
    x0 = list(x0)

    n = len(x0[0])

    # Ensure we have at least nroots initial guesses
    while len(x0) < nroots:
        # Generate additional random guess vectors
        v = np.random.RandomState(len(x0)).randn(n)
        v /= np.linalg.norm(v)
        x0.append(v)

    # Orthonormalize initial guesses (two CGS passes for numerical stability)
    vs = []
    for v in x0[:nroots]:
        v = v.copy().astype(np.float64)
        for u in vs:
            v -= u * np.dot(u, v)
        for u in vs:
            v -= u * np.dot(u, v)
        norm = np.linalg.norm(v)
        if norm > lindep:
            vs.append(v / norm)
    if len(vs) == 0:
        raise RuntimeError("Davidson: all initial guesses are linearly dependent")

    # Subspace vectors and sigma vectors (H @ v)
    subspace_v = list(vs)
    subspace_hv = [aop(v) for v in subspace_v]

    converged = [False] * nroots
    e_prev = np.zeros(nroots)

    for icycle in range(max_cycle):
        nv = len(subspace_v)

        # Build projected Hamiltonian: H_proj[i,j] = <v_i | H | v_j>
        h_proj = np.empty((nv, nv))
        for i in range(nv):
            for j in range(i + 1):
                h_proj[i, j] = h_proj[j, i] = np.dot(subspace_v[i], subspace_hv[j])

        # Diagonalize the small projected Hamiltonian
        e_sub, c_sub = np.linalg.eigh(h_proj)

        # Extract the nroots lowest Ritz values and vectors
        nroots_eff = min(nroots, nv)
        eigenvalues = e_sub[:nroots_eff]

        # Compute Ritz vectors and residuals in the full space
        new_vs = []
        all_converged = True
        for k in range(nroots_eff):
            # Ritz vector: x_k = sum_i c_sub[i,k] * v_i
            x_k = np.zeros(n)
            hx_k = np.zeros(n)
            for i in range(nv):
                x_k += c_sub[i, k] * subspace_v[i]
                hx_k += c_sub[i, k] * subspace_hv[i]

            # Residual: r_k = H x_k - e_k x_k
            r_k = hx_k - eigenvalues[k] * x_k
            r_norm = np.linalg.norm(r_k)

            if verbose > 0 and k == 0:
                de = eigenvalues[0] - e_prev[0] if icycle > 0 else 0.0
                print(
                    f"Davidson iter {icycle:3d}: E = {eigenvalues[0]:.12f}  "
                    f"dE = {de:+.2e}  |r| = {r_norm:.2e}  "
                    f"subspace = {nv}"
                )

            if r_norm < tol:
                converged[k] = True
            else:
                all_converged = False
                # Apply preconditioner to get correction vector
                delta = precond(r_k, eigenvalues[k])
                new_vs.append(delta)

        e_prev[:nroots_eff] = eigenvalues

        if all_converged:
            if verbose > 0:
                print(f"Davidson converged in {icycle + 1} iterations")
            break

        # Orthogonalize new vectors against existing subspace (two CGS passes)
        added = 0
        for delta in new_vs:
            for v in subspace_v:
                delta -= v * np.dot(v, delta)
            for v in subspace_v:
                delta -= v * np.dot(v, delta)
            norm = np.linalg.norm(delta)
            if norm > lindep:
                delta /= norm
                subspace_v.append(delta)
                subspace_hv.append(aop(delta))
                added += 1

        if added == 0:
            unconverged = [k for k in range(nroots) if not converged[k]]
            if unconverged:
                warnings.warn(
                    f"Davidson: subspace exhausted at iter {icycle} with "
                    f"{len(unconverged)} unconverged roots "
                    f"(indices {unconverged}). "
                    f"Results for these roots may be inaccurate.",
                    stacklevel=2,
                )
            if verbose > 0:
                print(
                    f"Davidson: no new vectors at iter {icycle}, "
                    f"{len(unconverged)} roots unconverged"
                )
            break

        # Restart if subspace gets too large
        if len(subspace_v) > max_space * nroots:
            if verbose > 0:
                print(f"Davidson: restarting, subspace {len(subspace_v)} > {max_space * nroots}")
            # Keep the nroots best Ritz vectors as the new subspace
            new_subspace_v = []
            new_subspace_hv = []
            for k in range(nroots_eff):
                x_k = np.zeros(n)
                hx_k = np.zeros(n)
                for i in range(nv):
                    x_k += c_sub[i, k] * subspace_v[i]
                    hx_k += c_sub[i, k] * subspace_hv[i]
                new_subspace_v.append(x_k)
                new_subspace_hv.append(hx_k)
            subspace_v = new_subspace_v
            subspace_hv = new_subspace_hv

    # Build final eigenvectors in the full space
    nv = len(subspace_v)
    h_proj = np.empty((nv, nv))
    for i in range(nv):
        for j in range(i + 1):
            h_proj[i, j] = h_proj[j, i] = np.dot(subspace_v[i], subspace_hv[j])
    e_sub, c_sub = np.linalg.eigh(h_proj)

    nroots_eff = min(nroots, nv)
    eigenvalues = e_sub[:nroots_eff]
    eigenvectors = np.zeros((n, nroots_eff))
    for k in range(nroots_eff):
        for i in range(nv):
            eigenvectors[:, k] += c_sub[i, k] * subspace_v[i]

    # Pad if fewer roots found than requested
    if nroots_eff < nroots:
        warnings.warn(
            f"Davidson found {nroots_eff} of {nroots} requested roots. "
            f"Missing roots padded with NaN.",
            stacklevel=2,
        )
        eigenvalues = np.concatenate([eigenvalues, np.full(nroots - nroots_eff, np.nan)])
        eigenvectors = np.concatenate(
            [eigenvectors, np.full((n, nroots - nroots_eff), np.nan)], axis=1
        )

    return eigenvalues, eigenvectors, converged
