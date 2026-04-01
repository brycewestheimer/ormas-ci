"""Validation tests for SF-ORMAS against PySCF CASCI reference calculations.

These tests validate SF-ORMAS on molecular systems beyond the basic H2 tests
in test_sf_integration.py, covering dissociation curves, excited states,
and larger active spaces.
"""

import pytest

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_h2(distance: float) -> tuple:
    """Build H2 molecule at given bond length and return (mol, mf_rohf).

    Args:
        distance: H-H distance in Angstrom.

    Returns:
        Tuple of (mol, mf) where mf is a converged triplet ROHF.
    """
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf


def _h2_sf_config() -> SFORMASConfig:
    """Return SFORMASConfig for H2 CAS(2,2) single SF."""
    return SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    )


def _build_twisted_ethylene() -> tuple:
    """Build 90-degree twisted ethylene and return (mol, mf_rohf).

    Returns:
        Tuple of (mol, mf) where mf is a converged triplet ROHF.
    """
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.340  0.000  0.000
        H -0.500  0.930  0.000
        H -0.500 -0.930  0.000
        H  1.840  0.000  0.930
        H  1.840  0.000 -0.930
        """,
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf


def _ethylene_sf_config() -> SFORMASConfig:
    """Return SFORMASConfig for ethylene CAS(2,2) single SF."""
    return SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    )


def _build_tmm() -> tuple:
    """Build trimethylenemethane (D3h) and return (mol, mf_rohf).

    Returns:
        Tuple of (mol, mf) where mf is a converged triplet ROHF.
    """
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.350  0.000  0.000
        C -0.675  1.169  0.000
        C -0.675 -1.169  0.000
        H  1.944  0.930  0.000
        H  1.944 -0.930  0.000
        H -1.269  2.099  0.000
        H -0.081  2.099  0.000
        H -1.269 -2.099  0.000
        H -0.081 -2.099  0.000
        """,
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf


def _tmm_sf_config() -> SFORMASConfig:
    """Return SFORMASConfig for TMM CAS(4,4) single SF."""
    return SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=4,
        n_active_electrons=4,
        subspaces=[
            Subspace("all", [0, 1, 2, 3], min_electrons=0, max_electrons=8),
        ],
    )


def _spin_quantum_number(ss: float) -> float:
    """Compute S from <S^2>.

    Args:
        ss: Expectation value <S^2>.

    Returns:
        Spin quantum number S.
    """
    return (-1 + (1 + 4 * ss) ** 0.5) / 2


# ---------------------------------------------------------------------------
# H2 Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestH2Dissociation:
    """Tests for SF-ORMAS on H2 dissociation curve."""

    def test_sf_h2_curve_matches_casci(self) -> None:
        """SF-ORMAS matches CASCI at equilibrium, stretched, and dissociated H2."""
        distances = [0.74, 2.0, 5.0]
        sf_config = _h2_sf_config()

        for d in distances:
            _mol, mf = _build_h2(d)

            # PySCF CASCI reference
            mc_ref = mcscf.CASCI(mf, 2, (1, 1))
            mc_ref.verbose = 0
            e_ref = mc_ref.kernel()[0]

            # SF-ORMAS
            mc_sf = mcscf.CASCI(mf, 2, (1, 1))
            mc_sf.verbose = 0
            mc_sf.fcisolver = SFORMASFCISolver(sf_config)
            e_sf = mc_sf.kernel()[0]

            assert abs(e_sf - e_ref) < 1e-10, (
                f"H2 at {d} A: SF-ORMAS={e_sf}, CASCI={e_ref}, diff={abs(e_sf - e_ref)}"
            )

    def test_sf_h2_singlet_below_triplet_at_equilibrium(self) -> None:
        """At equilibrium H2, the singlet root is lower than the triplet root."""
        _mol, mf = _build_h2(0.74)
        sf_config = _h2_sf_config()

        mc = mcscf.CASCI(mf, 2, (1, 1))
        mc.verbose = 0
        mc.fcisolver = SFORMASFCISolver(sf_config)
        mc.fcisolver.nroots = 2
        mc.kernel()

        # Identify singlet and triplet roots
        singlet_energy = None
        triplet_energy = None
        for i in range(2):
            ss, _mult = mc.fcisolver.spin_square(mc.ci[i], 2, (1, 1))
            s_val = _spin_quantum_number(ss)
            if round(s_val) == 0:
                singlet_energy = mc.e_tot[i]
            elif round(s_val) == 1:
                triplet_energy = mc.e_tot[i]

        assert singlet_energy is not None, "No singlet root found"
        assert triplet_energy is not None, "No triplet root found"
        assert singlet_energy < triplet_energy, (
            f"Singlet ({singlet_energy}) should be below triplet ({triplet_energy}) at equilibrium"
        )


# ---------------------------------------------------------------------------
# Ethylene Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEthylene:
    """Tests for SF-ORMAS on twisted ethylene."""

    def test_sf_ethylene_spin_purity(self) -> None:
        """Each root of twisted ethylene has a clean spin quantum number."""
        _mol, mf = _build_twisted_ethylene()
        sf_config = _ethylene_sf_config()

        mc = mcscf.CASCI(mf, 2, (1, 1))
        mc.verbose = 0
        mc.fcisolver = SFORMASFCISolver(sf_config)
        mc.fcisolver.nroots = 2
        mc.kernel()

        s_values = []
        for i in range(2):
            ss, _mult = mc.fcisolver.spin_square(mc.ci[i], 2, (1, 1))
            s_val = _spin_quantum_number(ss)
            assert abs(s_val - round(s_val)) < 1e-6, (
                f"Root {i}: S={s_val:.6f} is not a spin eigenstate"
            )
            s_values.append(round(s_val))

        # One singlet and one triplet
        assert 0 in s_values, f"No singlet root found, S values: {s_values}"
        assert 1 in s_values, f"No triplet root found, S values: {s_values}"

    def test_sf_ethylene_matches_casci(self) -> None:
        """SF-ORMAS matches PySCF CASCI for both roots of twisted ethylene."""
        _mol, mf = _build_twisted_ethylene()
        sf_config = _ethylene_sf_config()

        # PySCF CASCI reference (2 roots)
        mc_ref = mcscf.CASCI(mf, 2, (1, 1))
        mc_ref.verbose = 0
        mc_ref.fcisolver.nroots = 2
        mc_ref.kernel()
        e_casci = mc_ref.e_tot

        # SF-ORMAS (2 roots)
        mc_sf = mcscf.CASCI(mf, 2, (1, 1))
        mc_sf.verbose = 0
        mc_sf.fcisolver = SFORMASFCISolver(sf_config)
        mc_sf.fcisolver.nroots = 2
        mc_sf.kernel()
        e_sf = mc_sf.e_tot

        for i in range(2):
            assert abs(e_sf[i] - e_casci[i]) < 1e-10, (
                f"Root {i}: SF-ORMAS={e_sf[i]}, CASCI={e_casci[i]}, "
                f"diff={abs(e_sf[i] - e_casci[i])}"
            )


# ---------------------------------------------------------------------------
# TMM Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTMM:
    """Tests for SF-ORMAS on trimethylenemethane."""

    def test_sf_tmm_matches_casci(self) -> None:
        """SF-ORMAS ground state matches PySCF CASCI for TMM."""
        _mol, mf = _build_tmm()
        sf_config = _tmm_sf_config()

        # PySCF CASCI reference
        mc_ref = mcscf.CASCI(mf, 4, (2, 2))
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        # SF-ORMAS
        mc_sf = mcscf.CASCI(mf, 4, (2, 2))
        mc_sf.verbose = 0
        mc_sf.fcisolver = SFORMASFCISolver(sf_config)
        e_sf = mc_sf.kernel()[0]

        assert abs(e_sf - e_ref) < 1e-10, (
            f"TMM: SF-ORMAS={e_sf}, CASCI={e_ref}, diff={abs(e_sf - e_ref)}"
        )

    def test_sf_tmm_spin_purity(self) -> None:
        """Each root of TMM has a clean spin quantum number."""
        _mol, mf = _build_tmm()
        sf_config = _tmm_sf_config()

        mc = mcscf.CASCI(mf, 4, (2, 2))
        mc.verbose = 0
        mc.fcisolver = SFORMASFCISolver(sf_config)
        mc.fcisolver.nroots = 3
        mc.kernel()

        for i in range(3):
            ss, _mult = mc.fcisolver.spin_square(mc.ci[i], 4, (2, 2))
            s_val = _spin_quantum_number(ss)
            assert abs(s_val - round(s_val)) < 1e-6, (
                f"Root {i}: S={s_val:.6f} is not a spin eigenstate (<S^2>={ss:.6f})"
            )
