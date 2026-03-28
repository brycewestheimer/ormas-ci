"""
CI Hamiltonian matrix construction from a determinant list and integrals.

Builds the matrix H[i,j] = <det_i|H|det_j> over all pairs of determinants
using the Slater-Condon rules. Supports both dense and sparse output.
"""

import numpy as np
import scipy.sparse as sp

from pyscf.ormas_ci.slater_condon import matrix_element

__all__ = ["build_ci_hamiltonian"]


# 16-bit popcount lookup table for vectorized excitation classification
_POPCOUNT_TABLE_16 = np.array(
    [bin(i).count("1") for i in range(65536)], dtype=np.int32
)


def _vectorized_popcount(arr: np.ndarray) -> np.ndarray:
    """Count set bits for an array of integers using a 16-bit lookup table."""
    result = np.zeros(arr.shape, dtype=np.int32)
    for shift in range(0, 64, 16):
        result += _POPCOUNT_TABLE_16[(arr >> shift) & 0xFFFF]
    return result


def _precompute_excitation_pairs(
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute which upper-triangle (i,j) pairs have <=2 total excitations.

    Returns arrays of (row_indices, col_indices, n_excitations) for pairs
    with 0, 1, or 2 total excitations only. Pairs with >2 excitations
    are guaranteed zero by the Slater-Condon rules and are excluded.
    """
    n_det = len(alpha_strings)

    # For small determinant spaces, precompute the full matrix.
    # For larger spaces, use a block-based approach to limit memory.
    block_size = min(n_det, 2000)

    all_rows = []
    all_cols = []
    all_nexc = []

    for i_start in range(0, n_det, block_size):
        i_end = min(i_start + block_size, n_det)
        a_i = alpha_strings[i_start:i_end].astype(np.int64)
        b_i = beta_strings[i_start:i_end].astype(np.int64)

        for j_start in range(i_start, n_det, block_size):
            j_end = min(j_start + block_size, n_det)
            a_j = alpha_strings[j_start:j_end].astype(np.int64)
            b_j = beta_strings[j_start:j_end].astype(np.int64)

            # Vectorized XOR and popcount
            alpha_xor = a_i[:, None] ^ a_j[None, :]
            beta_xor = b_i[:, None] ^ b_j[None, :]

            n_diff_a = _vectorized_popcount(alpha_xor) // 2
            n_diff_b = _vectorized_popcount(beta_xor) // 2
            n_exc = n_diff_a + n_diff_b

            # Only keep upper triangle (i <= j) and excitations <= 2
            local_i, local_j = np.where(n_exc <= 2)
            global_i = local_i + i_start
            global_j = local_j + j_start

            # Filter to upper triangle only
            mask = global_i <= global_j
            global_i = global_i[mask]
            global_j = global_j[mask]
            exc_vals = n_exc[local_i[mask], local_j[mask]]

            all_rows.append(global_i)
            all_cols.append(global_j)
            all_nexc.append(exc_vals)

    return (
        np.concatenate(all_rows),
        np.concatenate(all_cols),
        np.concatenate(all_nexc),
    )


def build_ci_hamiltonian(
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    sparse_threshold: int = 5000,
) -> np.ndarray | sp.csr_matrix:
    """Build the CI Hamiltonian matrix in the determinant basis.

    The matrix is real and symmetric. Only the upper triangle is computed
    explicitly; the lower triangle is filled by symmetry.

    Uses vectorized excitation precomputation to skip pairs with >2
    total excitations (which are zero by the Slater-Condon rules).

    Args:
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        h1e: One-electron integrals, shape (ncas, ncas).
        h2e: Two-electron integrals in chemists' notation, shape (ncas,ncas,ncas,ncas).
        sparse_threshold: Use sparse matrix if n_det exceeds this value.
            Below 5000 determinants, dense storage requires ~200 MB
            (float64); above this, the Hamiltonian is typically >95%
            zeros, making CSR format more memory-efficient.

    Returns:
        Hamiltonian matrix as dense ndarray or sparse CSR matrix.
    """
    n_det = len(alpha_strings)
    use_sparse = n_det > sparse_threshold

    # Precompute which pairs have <=2 excitations (non-zero by Slater-Condon)
    pair_rows, pair_cols, _ = _precompute_excitation_pairs(
        alpha_strings, beta_strings
    )

    if use_sparse:
        rows, cols, vals = [], [], []
        for idx in range(len(pair_rows)):
            i, j = int(pair_rows[idx]), int(pair_cols[idx])
            val = matrix_element(
                int(alpha_strings[i]), int(beta_strings[i]),
                int(alpha_strings[j]), int(beta_strings[j]),
                h1e, h2e,
            )
            if abs(val) > 1e-15:
                rows.append(i)
                cols.append(j)
                vals.append(val)
                if i != j:
                    rows.append(j)
                    cols.append(i)
                    vals.append(val)

        h_ci = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_det, n_det),
            dtype=np.float64,
        )
    else:
        h_ci = np.zeros((n_det, n_det), dtype=np.float64)
        for idx in range(len(pair_rows)):
            i, j = int(pair_rows[idx]), int(pair_cols[idx])
            val = matrix_element(
                int(alpha_strings[i]), int(beta_strings[i]),
                int(alpha_strings[j]), int(beta_strings[j]),
                h1e, h2e,
            )
            h_ci[i, j] = val
            h_ci[j, i] = val

    return h_ci
