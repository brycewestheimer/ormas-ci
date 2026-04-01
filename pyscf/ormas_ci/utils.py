"""
Low-level bit manipulation utilities for occupation string representation.

Occupation strings represent Slater determinants as integers where bit k = 1
means orbital k is occupied. Alpha and beta strings are separate integers.
Orbital 0 is the least significant bit.

Example: 0b1010 (decimal 10) means orbitals 1 and 3 are occupied.
"""

import functools
from itertools import combinations

__all__ = [
    "popcount",
    "bits_to_indices",
    "indices_to_bits",
    "subspace_occupation",
    "generate_strings",
    "compute_phase",
]


def popcount(n: int) -> int:
    """Count the number of set bits (occupied orbitals) in an integer.

    Args:
        n: Non-negative integer representing an occupation bitstring.

    Returns:
        Number of set bits.
    """
    return n.bit_count()


@functools.cache
def bits_to_indices(n: int) -> tuple[int, ...]:
    """Convert a bitstring to a sorted tuple of set bit positions.

    Results are cached because the same occupation strings appear in many
    determinant pairs during Hamiltonian and RDM construction.

    Args:
        n: Non-negative integer representing an occupation bitstring.

    Returns:
        Sorted tuple of bit positions that are set to 1.
    """
    indices: list[int] = []
    pos = 0
    while n:
        if n & 1:
            indices.append(pos)
        n >>= 1
        pos += 1
    return tuple(indices)


def indices_to_bits(indices: list[int]) -> int:
    """Convert a list of orbital indices to a bitstring integer.

    Args:
        indices: List of orbital positions to set to 1.

    Returns:
        Integer with specified bit positions set.
    """
    n = 0
    for i in indices:
        n |= 1 << i
    return n


def subspace_occupation(bitstring: int, orbital_indices: list[int]) -> int:
    """Count how many of the specified orbitals are occupied in the bitstring.

    Args:
        bitstring: Occupation string as an integer.
        orbital_indices: List of orbital indices defining the subspace.

    Returns:
        Number of orbitals in the subspace that are occupied.
    """
    count = 0
    for idx in orbital_indices:
        if bitstring & (1 << idx):
            count += 1
    return count


def generate_strings(n_orbitals: int, n_electrons: int) -> list[int]:
    """Generate all occupation bitstrings with exactly n_electrons in n_orbitals.

    Args:
        n_orbitals: Total number of orbitals available.
        n_electrons: Number of electrons (set bits) required.

    Returns:
        Sorted list of integers, each with exactly n_electrons bits set
        within the lowest n_orbitals bits.
    """
    if n_electrons == 0:
        return [0]
    if n_electrons == n_orbitals:
        return [(1 << n_orbitals) - 1]
    if n_electrons > n_orbitals:
        return []

    strings = []
    for combo in combinations(range(n_orbitals), n_electrons):
        strings.append(indices_to_bits(list(combo)))
    return sorted(strings)


def compute_phase(string: int, p: int, q: int) -> int:
    """Compute the fermionic sign for moving an electron from orbital q to orbital p.

    The sign is (-1)^n where n is the number of occupied orbitals strictly
    between positions min(p,q) and max(p,q) in the occupation string.

    This accounts for the anticommutation of fermionic creation/annihilation
    operators: moving an electron past each occupied orbital picks up a factor of -1.

    Args:
        string: Occupation bitstring (before the excitation).
        p: One orbital index involved in the excitation.
        q: Other orbital index involved in the excitation.

    Returns:
        +1 or -1.
    """
    if p == q:
        return 1

    lo, hi = min(p, q), max(p, q)

    # Build a mask for bits strictly between lo and hi
    # (1 << hi) - 1 gives bits 0 through hi-1
    # (1 << (lo + 1)) - 1 gives bits 0 through lo
    # The difference masks bits lo+1 through hi-1
    mask = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1)
    n_between = popcount(string & mask)

    return 1 if n_between % 2 == 0 else -1
