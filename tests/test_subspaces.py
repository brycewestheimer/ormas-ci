"""Tests for ormas_ci.subspaces -- subspace definitions and validation."""

import pytest

from pyscf.ormas_ci.subspaces import ORMASConfig, RASConfig, Subspace


def test_valid_config():
    """Valid two-subspace config passes validation without error."""
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 1, 3),
            Subspace("B", [2, 3], 1, 3),
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    config.validate()  # Should not raise


def test_overlapping_orbitals():
    """Overlapping orbital indices across subspaces are rejected."""
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 0, 4),
            Subspace("B", [1, 2], 0, 4),  # Overlap at orbital 1
        ],
        n_active_orbitals=3,
        nelecas=(1, 1),
    )
    with pytest.raises(ValueError, match="multiple subspaces"):
        config.validate()


def test_incomplete_coverage():
    """Config with missing orbitals is rejected."""
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 0, 4),
        ],
        n_active_orbitals=4,  # Missing orbitals 2, 3
        nelecas=(1, 1),
    )
    with pytest.raises(ValueError, match="do not cover"):
        config.validate()


def test_impossible_electron_count():
    """Config where min electrons exceed available electrons is rejected."""
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], 3, 4),  # min 3
            Subspace("B", [2, 3], 3, 4),  # min 3, total min=6
        ],
        n_active_orbitals=4,
        nelecas=(2, 2),  # only 4 electrons, can't satisfy min=6
    )
    with pytest.raises(ValueError, match="less than"):
        config.validate()


def test_min_greater_than_max():
    """Subspace with min_electrons > max_electrons is rejected."""
    with pytest.raises(ValueError, match="min_electrons"):
        Subspace("bad", [0, 1], 3, 1).validate()


def test_ras_to_ormas_conversion():
    """RASConfig converts to a valid 3-subspace ORMASConfig."""
    ras = RASConfig(
        ras1_orbitals=[0, 1],
        ras2_orbitals=[2, 3],
        ras3_orbitals=[4, 5],
        max_holes_ras1=1,
        max_particles_ras3=1,
        nelecas=(3, 3),
    )
    config = ras.to_ormas_config()
    config.validate()  # Must be valid
    assert config.n_active_orbitals == 6
    assert len(config.subspaces) == 3


def test_single_subspace_unrestricted():
    """Single unrestricted subspace is a valid ORMAS config."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1, 2, 3], 0, 8)],
        n_active_orbitals=4,
        nelecas=(2, 2),
    )
    config.validate()
    assert config.n_subspaces == 1
