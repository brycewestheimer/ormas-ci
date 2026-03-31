"""Tests for ormas_ci.spinflip -- spin-flip ORMAS-CI configuration and enumeration."""

from math import comb

import numpy as np
import pytest

from pyscf.ormas_ci.determinants import build_determinant_list
from pyscf.ormas_ci.spinflip import (
    build_reference_determinant,
    count_sf_determinants,
    generate_sf_determinants,
    validate_reference_consistency,
)
from pyscf.ormas_ci.subspaces import (
    ORMASConfig,
    SFORMASConfig,
    SFRASConfig,
    Subspace,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def triplet_cas22() -> SFORMASConfig:
    """Single SF, triplet->singlet, CAS(2,2)."""
    return SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[
            Subspace("sf_cas", [0, 1], min_electrons=0, max_electrons=4),
        ],
    )


@pytest.fixture()
def quintet_cas44() -> SFORMASConfig:
    """Double SF, quintet->singlet, CAS(4,4)."""
    return SFORMASConfig(
        ref_spin=4,
        target_spin=0,
        n_spin_flips=2,
        n_active_orbitals=4,
        n_active_electrons=4,
        subspaces=[
            Subspace("sf_cas", [0, 1, 2, 3], min_electrons=0, max_electrons=8),
        ],
    )


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestSFORMASConfig:
    """Tests for SFORMASConfig validation and properties."""

    def test_sf_config_validation(self, triplet_cas22: SFORMASConfig) -> None:
        """Valid config (triplet->singlet, CAS(2,2)) passes without error."""
        # __post_init__ ran without raising; verify key fields
        assert triplet_cas22.ref_spin == 2
        assert triplet_cas22.target_spin == 0
        assert triplet_cas22.n_spin_flips == 1

    def test_sf_config_nelecas_reference(self) -> None:
        """Reference (alpha, beta) derived correctly from ref_spin."""
        # Triplet, 2 electrons -> (2, 0)
        cfg_t2 = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=2, n_active_electrons=2,
            subspaces=[Subspace("cas", [0, 1], 0, 4)],
        )
        assert cfg_t2.nelecas_reference == (2, 0)

        # Quintet, 4 electrons -> (4, 0)
        cfg_q4 = SFORMASConfig(
            ref_spin=4, target_spin=0, n_spin_flips=2,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[Subspace("cas", [0, 1, 2, 3], 0, 8)],
        )
        assert cfg_q4.nelecas_reference == (4, 0)

        # Triplet, 4 electrons -> (3, 1)
        cfg_t4 = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[Subspace("cas", [0, 1, 2, 3], 0, 8)],
        )
        assert cfg_t4.nelecas_reference == (3, 1)

    def test_sf_config_nelecas_target(self) -> None:
        """Target (alpha, beta) after spin flips."""
        # Single SF: (2,0) -> (1,1)
        cfg_1sf = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=2, n_active_electrons=2,
            subspaces=[Subspace("cas", [0, 1], 0, 4)],
        )
        assert cfg_1sf.nelecas_target == (1, 1)

        # Double SF: (4,0) -> (2,2)
        cfg_2sf = SFORMASConfig(
            ref_spin=4, target_spin=0, n_spin_flips=2,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[Subspace("cas", [0, 1, 2, 3], 0, 8)],
        )
        assert cfg_2sf.nelecas_target == (2, 2)

    def test_sf_config_to_ormas(self, triplet_cas22: SFORMASConfig) -> None:
        """ORMASConfig has target nelecas and same subspaces."""
        ormas = triplet_cas22.to_ormas_config()
        assert ormas.nelecas == triplet_cas22.nelecas_target
        assert ormas.nelecas == (1, 1)
        assert ormas.subspaces is triplet_cas22.subspaces
        assert ormas.n_active_orbitals == triplet_cas22.n_active_orbitals

    def test_sf_config_parity_check(self) -> None:
        """Odd ref_spin + even n_electrons raises ValueError with 'parity'."""
        with pytest.raises(ValueError, match="parity"):
            SFORMASConfig(
                ref_spin=1, target_spin=1, n_spin_flips=0,
                n_active_orbitals=2, n_active_electrons=2,
                subspaces=[Subspace("cas", [0, 1], 0, 4)],
            )

    def test_sf_config_too_many_flips(self) -> None:
        """ref_spin exceeds n_active_electrons raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            SFORMASConfig(
                ref_spin=4, target_spin=0, n_spin_flips=2,
                n_active_orbitals=2, n_active_electrons=2,
                subspaces=[Subspace("cas", [0, 1], 0, 4)],
            )

    def test_sf_config_negative_ref_spin(self) -> None:
        """Negative ref_spin raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SFORMASConfig(
                ref_spin=-2, target_spin=0, n_spin_flips=-1,
                n_active_orbitals=2, n_active_electrons=2,
                subspaces=[Subspace("cas", [0, 1], 0, 4)],
            )

    def test_sf_config_ref_less_than_target(self) -> None:
        """ref_spin < target_spin raises ValueError."""
        with pytest.raises(ValueError, match=">="):
            SFORMASConfig(
                ref_spin=0, target_spin=2, n_spin_flips=-1,
                n_active_orbitals=2, n_active_electrons=2,
                subspaces=[Subspace("cas", [0, 1], 0, 4)],
            )


class TestSFRASConfig:
    """Tests for SFRASConfig conversion."""

    def test_sfras_config_conversion(self) -> None:
        """SFRASConfig -> SFORMASConfig: correct subspaces, bounds, n_spin_flips."""
        sfras = SFRASConfig(
            ras1_orbitals=[0, 1],
            ras2_orbitals=[2, 3],
            ras3_orbitals=[4, 5],
            max_holes_ras1=1,
            max_particles_ras3=1,
            n_active_electrons=6,
            ref_spin=2,
            target_spin=0,
        )
        sf_config = sfras.to_sf_ormas_config()
        assert sf_config.n_spin_flips == 1
        assert sf_config.n_active_orbitals == 6
        assert sf_config.n_active_electrons == 6
        assert len(sf_config.subspaces) == 3

        ras1 = sf_config.subspaces[0]
        assert ras1.name == "RAS1"
        assert ras1.orbital_indices == [0, 1]
        assert ras1.min_electrons == 3  # 2*2 - 1
        assert ras1.max_electrons == 4  # 2*2

        ras2 = sf_config.subspaces[1]
        assert ras2.name == "RAS2_SF"
        assert ras2.min_electrons == 0
        assert ras2.max_electrons == 4

        ras3 = sf_config.subspaces[2]
        assert ras3.name == "RAS3"
        assert ras3.min_electrons == 0
        assert ras3.max_electrons == 1


class TestSingleSFDiradical:
    """Tests for the convenience constructor."""

    def test_single_sf_diradical_convenience(self) -> None:
        """Convenience constructor: ref_spin=2, target_spin=0, correct bounds."""
        cfg = SFORMASConfig.single_sf_diradical(
            n_active_orbitals=4,
            n_active_electrons=4,
            sf_cas_orbitals=[1, 2],
            hole_orbitals=[0],
            particle_orbitals=[3],
            max_holes=1,
            max_particles=1,
        )
        assert cfg.ref_spin == 2
        assert cfg.target_spin == 0
        assert cfg.n_spin_flips == 1
        assert len(cfg.subspaces) == 3

        hole = cfg.subspaces[0]
        assert hole.name == "hole"
        assert hole.min_electrons == 1  # 2 - 1
        assert hole.max_electrons == 2

        cas = cfg.subspaces[1]
        assert cas.name == "sf_cas"
        assert cas.min_electrons == 0
        assert cas.max_electrons == 4

        particle = cfg.subspaces[2]
        assert particle.name == "particle"
        assert particle.min_electrons == 0
        assert particle.max_electrons == 1


# ---------------------------------------------------------------------------
# Determinant Enumeration Tests
# ---------------------------------------------------------------------------

class TestSFDeterminants:
    """Tests for SF determinant enumeration."""

    def test_sf_det_count_h2_singlet(self, triplet_cas22: SFORMASConfig) -> None:
        """CAS(2,2), single SF, full bounds: 4 determinants."""
        n_det = count_sf_determinants(triplet_cas22)
        # M_s=0 sector of CAS(2,2): C(2,1)*C(2,1) = 4
        assert n_det == 4

    def test_sf_det_count_ethylene(self) -> None:
        """CAS(2,2), single SF: 4 determinants (same physics as H2)."""
        cfg = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=2, n_active_electrons=2,
            subspaces=[Subspace("pi", [0, 1], 0, 4)],
        )
        assert count_sf_determinants(cfg) == 4

    def test_sf_det_count_double_sf(
        self, quintet_cas44: SFORMASConfig,
    ) -> None:
        """CAS(4,4) double SF quintet->singlet matches full CAS(4,4) M_s=0."""
        n_sf = count_sf_determinants(quintet_cas44)
        n_cas = comb(4, 2) * comb(4, 2)  # 36
        assert n_sf == n_cas

    def test_sf_dets_equal_ormas_dets(
        self, triplet_cas22: SFORMASConfig,
    ) -> None:
        """SF determinants == standard ORMAS determinants with equivalent nelecas."""
        sf_alpha, sf_beta = generate_sf_determinants(triplet_cas22)

        # Build equivalent ORMAS config directly
        ormas_config = ORMASConfig(
            subspaces=triplet_cas22.subspaces,
            n_active_orbitals=triplet_cas22.n_active_orbitals,
            nelecas=triplet_cas22.nelecas_target,
        )
        ormas_alpha, ormas_beta = build_determinant_list(ormas_config)

        # Compare as sets of pairs
        sf_pairs = {
            (int(a), int(b)) for a, b in zip(sf_alpha, sf_beta)
        }
        ormas_pairs = {
            (int(a), int(b))
            for a, b in zip(ormas_alpha, ormas_beta)
        }
        assert sf_pairs == ormas_pairs

    def test_sf_det_count_with_restrictions(self) -> None:
        """SF with tight ORMAS bounds produces fewer dets than unrestricted."""
        # Unrestricted CAS(2,2)
        cfg_full = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[
                Subspace("cas", [0, 1, 2, 3], 0, 8),
            ],
        )
        n_full = count_sf_determinants(cfg_full)

        # Restricted: 2-subspace with tight bounds
        cfg_restricted = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[
                Subspace("hole", [0, 1], 2, 4),
                Subspace("sf_cas", [2, 3], 0, 4),
            ],
        )
        n_restricted = count_sf_determinants(cfg_restricted)
        assert n_restricted < n_full


# ---------------------------------------------------------------------------
# Reference Analysis Tests
# ---------------------------------------------------------------------------

class TestReferenceAnalysis:
    """Tests for reference determinant construction and validation."""

    def test_build_reference_determinant_aufbau(
        self, triplet_cas22: SFORMASConfig,
    ) -> None:
        """Triplet CAS(2,2): alpha_str=0b11, beta_str=0b00."""
        alpha, beta = build_reference_determinant(triplet_cas22)
        assert alpha == 0b11
        assert beta == 0b00

    def test_build_reference_determinant_from_occ(self) -> None:
        """Occupation [2,1,1,0]: alpha_str=0b0111, beta_str=0b0001."""
        cfg = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[Subspace("cas", [0, 1, 2, 3], 0, 8)],
        )
        occ = np.array([2.0, 1.0, 1.0, 0.0])
        alpha, beta = build_reference_determinant(cfg, active_occ=occ)
        assert alpha == 0b0111
        assert beta == 0b0001

    def test_validate_reference_consistency_clean(
        self, triplet_cas22: SFORMASConfig,
    ) -> None:
        """Well-configured system returns no warnings."""
        mo_occ = np.array([1.0, 1.0])  # Both singly occupied, in sf_cas
        result = validate_reference_consistency(triplet_cas22, mo_occ=mo_occ)
        assert result["warnings"] == []
        assert result["n_det"] == 4
        assert result["nelecas_ref"] == (2, 0)
        assert result["nelecas_target"] == (1, 1)

    def test_validate_reference_consistency_warnings(self) -> None:
        """Singly-occupied orbitals in 'hole' subspace -> warning."""
        cfg = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=4, n_active_electrons=4,
            subspaces=[
                Subspace("hole", [0, 1], 2, 4),
                Subspace("sf_cas", [2, 3], 0, 4),
            ],
        )
        # Orbital 0 is singly occupied but assigned to "hole" space
        mo_occ = np.array([1.0, 2.0, 1.0, 0.0])
        result = validate_reference_consistency(cfg, mo_occ=mo_occ)
        assert len(result["warnings"]) == 1
        assert "not in the SF-CAS" in result["warnings"][0]

    def test_validate_reference_negative_target_alpha(self) -> None:
        """Too many spin flips raises ValueError with 'negative'."""
        # 2 electrons, ref_spin=2 -> ref (2,0), but 2 flips -> target (-0, 2)
        # Actually need ref_spin=2 and n_spin_flips=2 which is inconsistent
        # with target_spin. Let's build a scenario that triggers it.
        # Use 2 electrons with quintet ref_spin=4 -> that would fail parity.
        # Actually, we need a config that passes __post_init__ but fails
        # in validate_reference_consistency. The check is on tgt_a < 0.
        # With ref (2,0) and 1 SF -> (1,1): fine.
        # We need many flips. Let's do 4 electrons, ref_spin=4 -> (4,0),
        # n_spin_flips=2, target_spin=0 -> (2,2): fine.
        # The check tgt_a < 0 is a runtime check that would fail if someone
        # manually constructed a bad config. Let's test via a subclass trick:
        # Actually the simplest: 4e, ref_spin=4, target_spin=0, flips=2 -> (2,2)
        # That's valid. The negative target alpha scenario requires more
        # flips than alpha electrons, which __post_init__ prevents via the
        # n_spin_flips consistency check. So we test the validate function
        # directly by creating a config object that bypasses __post_init__.
        # Use object.__setattr__ after creation:
        cfg = SFORMASConfig(
            ref_spin=2, target_spin=0, n_spin_flips=1,
            n_active_orbitals=2, n_active_electrons=2,
            subspaces=[Subspace("cas", [0, 1], 0, 4)],
        )
        # Manually override to create an inconsistent state for testing
        object.__setattr__(cfg, "n_spin_flips", 3)
        with pytest.raises(ValueError, match="negative"):
            validate_reference_consistency(cfg)
