"""
Determinant enumeration with ORMAS occupation restrictions.

The enumeration proceeds in three stages:

1. Enumerate valid per-subspace occupation distributions: how many alpha and
   beta electrons in each subspace, satisfying all min/max constraints.

2. For each valid distribution, generate alpha and beta occupation strings
   within each subspace, then map them to the full active space.

3. Form all valid (alpha, beta) determinant pairs by combining across
   subspaces and across distributions.
"""

from math import comb

import numpy as np

from pyscf.ormas_ci.subspaces import ORMASConfig
from pyscf.ormas_ci.utils import generate_strings

__all__ = [
    "build_determinant_list",
    "count_determinants",
    "enumerate_distributions",
    "casci_determinant_count",
]


def _enumerate_distributions(
    subspaces: list,
    n_alpha_remaining: int,
    n_beta_remaining: int,
    current_alpha: list[int],
    current_beta: list[int],
    idx: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Recursively enumerate valid (alpha, beta) occupation distributions.

    At each recursion level, tries all valid (n_alpha_k, n_beta_k) assignments
    for subspace idx, then recurses with updated remaining counts.

    Returns list of (alpha_dist, beta_dist) tuples where each tuple has
    one entry per subspace.
    """
    if idx == len(subspaces):
        if n_alpha_remaining == 0 and n_beta_remaining == 0:
            return [(tuple(current_alpha), tuple(current_beta))]
        return []

    sub = subspaces[idx]
    n_orb = sub.n_orbitals
    min_elec = sub.min_electrons
    max_elec = sub.max_electrons
    results = []

    max_alpha = min(n_orb, n_alpha_remaining)
    max_beta = min(n_orb, n_beta_remaining)

    for na in range(max_alpha + 1):
        for nb in range(max_beta + 1):
            total = na + nb
            if total < min_elec or total > max_elec:
                continue

            remaining_alpha = n_alpha_remaining - na
            remaining_beta = n_beta_remaining - nb
            remaining_subs = subspaces[idx + 1:]

            # Pruning: check feasibility for remaining subspaces
            if remaining_subs:
                max_remaining_a = sum(s.n_orbitals for s in remaining_subs)
                max_remaining_b = max_remaining_a
                min_remaining_total = sum(s.min_electrons for s in remaining_subs)
                max_remaining_total = sum(s.max_electrons for s in remaining_subs)

                if remaining_alpha > max_remaining_a:
                    continue
                if remaining_beta > max_remaining_b:
                    continue
                if remaining_alpha + remaining_beta < min_remaining_total:
                    continue
                if remaining_alpha + remaining_beta > max_remaining_total:
                    continue

            current_alpha.append(na)
            current_beta.append(nb)
            results.extend(_enumerate_distributions(
                subspaces, remaining_alpha, remaining_beta,
                current_alpha, current_beta, idx + 1,
            ))
            current_alpha.pop()
            current_beta.pop()

    return results


def enumerate_distributions(
    config: ORMASConfig,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate all valid occupation distributions for the given config.

    Args:
        config: Validated ORMASConfig.

    Returns:
        List of (alpha_dist, beta_dist) tuples. alpha_dist[k] is the number
        of alpha electrons in subspace k.
    """
    return _enumerate_distributions(
        config.subspaces, config.n_alpha, config.n_beta,
        [], [], 0,
    )


def _subspace_strings_to_full(
    subspace_string: int,
    orbital_indices: list[int],
) -> int:
    """Map a subspace-local occupation string to the full active space.

    The subspace string uses local orbital numbering (0 to n_orb_k - 1).
    This maps those bits to the actual orbital indices in the full active space.

    Args:
        subspace_string: Occupation string in subspace-local basis.
        orbital_indices: Global orbital indices for this subspace.

    Returns:
        Occupation string in the full active space basis.
    """
    full_string = 0
    for local_idx, global_idx in enumerate(orbital_indices):
        if subspace_string & (1 << local_idx):
            full_string |= (1 << global_idx)
    return full_string


def build_determinant_list(
    config: ORMASConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the complete list of allowed determinants.

    This is the main entry point for determinant enumeration. It returns
    paired arrays: determinant i is (alpha_strings[i], beta_strings[i]).

    Args:
        config: Validated ORMASConfig specifying the subspace partitioning.

    Returns:
        (alpha_strings, beta_strings) as 1D int64 numpy arrays of equal length.
    """
    config.validate()
    distributions = enumerate_distributions(config)

    all_alpha = []
    all_beta = []

    for alpha_dist, beta_dist in distributions:
        # For each subspace, generate local strings and map to full space
        subspace_alpha_lists = []
        subspace_beta_lists = []

        for k, sub in enumerate(config.subspaces):
            na_k = alpha_dist[k]
            nb_k = beta_dist[k]

            local_alpha = generate_strings(sub.n_orbitals, na_k)
            local_beta = generate_strings(sub.n_orbitals, nb_k)

            full_alpha = [
                _subspace_strings_to_full(s, sub.orbital_indices)
                for s in local_alpha
            ]
            full_beta = [
                _subspace_strings_to_full(s, sub.orbital_indices)
                for s in local_beta
            ]

            subspace_alpha_lists.append(full_alpha)
            subspace_beta_lists.append(full_beta)

        # Build full active-space strings by combining per-subspace strings
        # via bitwise OR. This produces the Cartesian product because
        # subspace orbital indices are DISJOINT (enforced by
        # ORMASConfig.validate()): if subspace A uses bits {0,1} and
        # subspace B uses bits {2,3}, their strings never share set bits,
        # so OR is equivalent to concatenation of occupation patterns.
        combined_alpha = [0]
        for sub_list in subspace_alpha_lists:
            combined_alpha = [a | s for a in combined_alpha for s in sub_list]

        combined_beta = [0]
        for sub_list in subspace_beta_lists:
            combined_beta = [b | s for b in combined_beta for s in sub_list]

        # All (alpha, beta) pairs for this distribution
        for a in combined_alpha:
            for b in combined_beta:
                all_alpha.append(a)
                all_beta.append(b)

    return np.array(all_alpha, dtype=np.int64), np.array(all_beta, dtype=np.int64)


def count_determinants(config: ORMASConfig) -> int:
    """Count determinants without fully enumerating them.

    Faster than len(build_determinant_list(...)[0]) for estimating space size.

    Args:
        config: Validated ORMASConfig.

    Returns:
        Total number of determinants satisfying the ORMAS constraints.
    """
    config.validate()
    distributions = enumerate_distributions(config)

    total = 0
    for alpha_dist, beta_dist in distributions:
        count = 1
        for k, sub in enumerate(config.subspaces):
            count *= comb(sub.n_orbitals, alpha_dist[k])
            count *= comb(sub.n_orbitals, beta_dist[k])
        total += count
    return total


def casci_determinant_count(ncas: int, nelecas: tuple[int, int]) -> int:
    """Count full CASCI determinants for comparison.

    Args:
        ncas: Number of active orbitals.
        nelecas: (n_alpha, n_beta) active electrons.

    Returns:
        Total CASCI determinant count.
    """
    return comb(ncas, nelecas[0]) * comb(ncas, nelecas[1])
