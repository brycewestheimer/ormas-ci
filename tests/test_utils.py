"""Tests for ormas_ci.utils -- bit manipulation utilities."""

from math import comb

from pyscf.ormas_ci.utils import (
    bits_to_indices,
    compute_phase,
    generate_strings,
    indices_to_bits,
    popcount,
    subspace_occupation,
)


def test_popcount_zero():
    """Popcount of zero returns zero."""
    assert popcount(0) == 0


def test_popcount_powers_of_two():
    """Each power of two has exactly one set bit."""
    for i in range(10):
        assert popcount(1 << i) == 1


def test_popcount_all_bits():
    """Popcount of contiguous bit patterns matches expected count."""
    assert popcount(0b1111) == 4
    assert popcount(0b11111111) == 8


def test_bits_to_indices_roundtrip():
    """bits_to_indices and indices_to_bits are inverses."""
    for indices in [[0], [1, 3], [0, 2, 4], [0, 1, 2, 3]]:
        assert bits_to_indices(indices_to_bits(indices)) == tuple(sorted(indices))


def test_generate_strings_count():
    """generate_strings produces C(n,k) strings for all valid n, k."""
    for n in range(1, 8):
        for k in range(n + 1):
            strings = generate_strings(n, k)
            assert len(strings) == comb(n, k)


def test_generate_strings_popcount():
    """All generated strings have exactly k set bits."""
    for s in generate_strings(6, 3):
        assert popcount(s) == 3


def test_generate_strings_range():
    """All generated strings fit within n bits."""
    for s in generate_strings(6, 3):
        assert s < (1 << 6)


def test_compute_phase_adjacent():
    """Phase is -1 when one occupied orbital lies between p and q."""
    string = 0b0011  # orbitals 0,1 occupied
    assert compute_phase(string, 2, 0) == -1  # one occupied (1) between 0 and 2


def test_compute_phase_no_between():
    """Phase is +1 when no occupied orbital lies between p and q."""
    string = 0b0011
    assert compute_phase(string, 1, 0) == 1


def test_compute_phase_same_orbital():
    """Phase is +1 for zero excitation (same orbital)."""
    assert compute_phase(0b1111, 2, 2) == 1


def test_compute_phase_large_indices():
    """Phase computation works for orbital indices > 32."""
    # Orbitals 0, 33, 63 occupied
    string = (1 << 0) | (1 << 33) | (1 << 63)
    # Excite from 0 to 63: one occupied orbital (33) between them -> phase = -1
    assert compute_phase(string, 63, 0) == -1
    # Excite from 33 to 63: no occupied between -> phase = +1
    assert compute_phase(string, 63, 33) == 1


def test_compute_phase_adjacent_large():
    """Phase for adjacent orbitals at high index."""
    string = (1 << 50) | (1 << 51) | (1 << 52)
    # Excite from 50 to 52: one occupied (51) between -> phase = -1
    assert compute_phase(string, 50, 52) == -1
    # Excite from 50 to 51: no occupied between -> phase = +1
    assert compute_phase(string, 50, 51) == 1


def test_subspace_occupation():
    """Subspace occupation counts occupied orbitals within specified indices."""
    string = 0b1011  # orbitals 0, 1, 3 occupied
    assert subspace_occupation(string, [0, 1]) == 2
    assert subspace_occupation(string, [2, 3]) == 1
    assert subspace_occupation(string, [0, 1, 2, 3]) == 3
