"""Tests for ormas_ci.determinants -- determinant enumeration."""

from math import comb

from ormas_ci.determinants import (
    build_determinant_list,
    casci_determinant_count,
    count_determinants,
)
from ormas_ci.subspaces import ORMASConfig, RASConfig, Subspace
from ormas_ci.utils import popcount, subspace_occupation


def test_single_subspace_matches_casci():
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1, 2, 3], 0, 8)],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    alpha, beta = build_determinant_list(config)
    expected = comb(4, 2) * comb(4, 2)  # 36
    assert len(alpha) == expected


def test_count_matches_build():
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 1, 3),
            Subspace("B", [2, 3], 1, 3),
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    alpha, beta = build_determinant_list(config)
    assert count_determinants(config) == len(alpha)


def test_restriction_reduces_count():
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 1, 3),
            Subspace("B", [2, 3], 1, 3),
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    n_ormas = count_determinants(config)
    n_casci = casci_determinant_count(4, (2, 2))
    assert n_ormas < n_casci


def test_correct_popcount():
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1, 2, 3], 0, 8)],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    alpha, beta = build_determinant_list(config)
    for a in alpha:
        assert popcount(int(a)) == 2
    for b in beta:
        assert popcount(int(b)) == 2


def test_subspace_constraints_satisfied():
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 1, 3),
            Subspace("B", [2, 3], 1, 3),
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    alpha, beta = build_determinant_list(config)
    for i in range(len(alpha)):
        for sub in config.subspaces:
            occ = (
                subspace_occupation(int(alpha[i]), sub.orbital_indices)
                + subspace_occupation(int(beta[i]), sub.orbital_indices)
            )
            assert sub.min_electrons <= occ <= sub.max_electrons


def test_no_duplicate_determinants():
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 1, 3),
            Subspace("B", [2, 3], 1, 3),
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    alpha, beta = build_determinant_list(config)
    pairs = set()
    for i in range(len(alpha)):
        pair = (int(alpha[i]), int(beta[i]))
        assert pair not in pairs, f"Duplicate determinant: {pair}"
        pairs.add(pair)


def test_ras_determinant_count():
    ras = RASConfig(
        ras1_orbitals=[0, 1],
        ras2_orbitals=[2, 3],
        ras3_orbitals=[4, 5],
        max_holes_ras1=1,
        max_particles_ras3=1,
        nelecas=(3, 3),
    )
    config = ras.to_ormas_config()
    n = count_determinants(config)
    assert n > 0
    assert n < casci_determinant_count(6, (3, 3))
