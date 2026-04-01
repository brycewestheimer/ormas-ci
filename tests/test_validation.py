"""Tests for input validation and error handling."""

import numpy as np
import pytest

from pyscf.ormas_ci.fcisolver import ORMASFCISolver, _make_hdiag_vectorized, _transform_ci
from pyscf.ormas_ci.sigma import SigmaEinsum
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace


class TestSigmaEinsumValidation:
    """Shape validation in SigmaEinsum.__init__."""

    def test_invalid_h1e_1d(self):
        """1D h1e should raise ValueError."""
        h1e = np.zeros(4)
        eri = np.zeros((4, 4, 4, 4))
        with pytest.raises(ValueError, match="h1e must be a square 2D array"):
            SigmaEinsum(
                np.array([0b01], dtype=np.int64),
                np.array([0b01], dtype=np.int64),
                h1e,
                eri,
                (1, 1),
            )

    def test_invalid_h1e_nonsquare(self):
        """Non-square h1e should raise ValueError."""
        h1e = np.zeros((3, 4))
        eri = np.zeros((3, 3, 3, 3))
        with pytest.raises(ValueError, match="h1e must be a square 2D array"):
            SigmaEinsum(
                np.array([0b01], dtype=np.int64),
                np.array([0b01], dtype=np.int64),
                h1e,
                eri,
                (1, 1),
            )

    def test_invalid_eri_shape(self):
        """ERI with wrong shape should raise ValueError."""
        norb = 2
        h1e = np.zeros((norb, norb))
        eri = np.zeros((norb, norb, norb))  # 3D instead of 4D
        with pytest.raises(ValueError, match="eri must have shape"):
            SigmaEinsum(
                np.array([0b01], dtype=np.int64),
                np.array([0b01], dtype=np.int64),
                h1e,
                eri,
                (1, 1),
            )

    def test_eri_h1e_norb_mismatch(self):
        """ERI norb different from h1e norb should raise ValueError."""
        h1e = np.zeros((2, 2))
        eri = np.zeros((3, 3, 3, 3))
        with pytest.raises(ValueError, match="eri must have shape"):
            SigmaEinsum(
                np.array([0b01], dtype=np.int64),
                np.array([0b01], dtype=np.int64),
                h1e,
                eri,
                (1, 1),
            )


class TestHdiagValidation:
    """Shape validation in _make_hdiag_vectorized."""

    def test_invalid_h1e_shape(self):
        """1D h1e raises ValueError."""
        alpha = np.array([0b01], dtype=np.int64)
        beta = np.array([0b01], dtype=np.int64)
        h1e = np.zeros(2)
        h2e = np.zeros((2, 2, 2, 2))
        with pytest.raises(ValueError, match="h1e must be a square 2D"):
            _make_hdiag_vectorized(alpha, beta, h1e, h2e)

    def test_invalid_h2e_shape(self):
        """h2e with wrong shape raises ValueError."""
        alpha = np.array([0b01], dtype=np.int64)
        beta = np.array([0b01], dtype=np.int64)
        h1e = np.zeros((2, 2))
        h2e = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="h2e must have shape"):
            _make_hdiag_vectorized(alpha, beta, h1e, h2e)


class TestTransformCIValidation:
    """Shape validation in _transform_ci."""

    def test_invalid_u_shape(self):
        """Wrong-shape rotation matrix raises ValueError."""
        ci = np.array([1.0, 0.0])
        alpha = np.array([0b01, 0b10], dtype=np.int64)
        beta = np.array([0b01, 0b01], dtype=np.int64)
        ncas = 2
        u = np.eye(3)  # Wrong size
        with pytest.raises(ValueError, match="Alpha rotation matrix"):
            _transform_ci(ci, alpha, beta, ncas, u)

    def test_invalid_u_tuple_shape(self):
        """Wrong-shape rotation matrix tuple raises ValueError."""
        ci = np.array([1.0, 0.0])
        alpha = np.array([0b01, 0b10], dtype=np.int64)
        beta = np.array([0b01, 0b01], dtype=np.int64)
        ncas = 2
        ua = np.eye(2)
        ub = np.eye(3)  # Wrong size
        with pytest.raises(ValueError, match="Beta rotation matrix"):
            _transform_ci(ci, alpha, beta, ncas, (ua, ub))


class TestCI1dTo2dValidation:
    """Length validation in _ci_1d_to_2d."""

    def test_wrong_length_ci_vector(self):
        """CI vector with wrong length raises ValueError."""
        from pyscf import gto, mcscf, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.run()

        config = ORMASConfig(
            subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
            n_active_orbitals=2,
            nelecas=(1, 1),
        )
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        wrong_ci = np.zeros(999)
        with pytest.raises(ValueError, match="CI vector length"):
            mc.fcisolver._ci_1d_to_2d(wrong_ci)


class TestSFConfigValidation:
    """SFORMASConfig.to_ormas_config() now calls validate()."""

    def test_sf_config_to_ormas_validates(self):
        """Invalid SF config raises ValueError on to_ormas_config()."""
        from pyscf.ormas_ci.subspaces import SFORMASConfig

        # Overlapping orbital indices -> should fail validate()
        sf_config = SFORMASConfig(
            subspaces=[
                Subspace("A", [0, 1], min_electrons=0, max_electrons=4),
                Subspace("B", [1, 2], min_electrons=0, max_electrons=4),
            ],
            n_active_orbitals=3,
            n_active_electrons=2,
            ref_spin=2,
            target_spin=0,
            n_spin_flips=1,
        )
        with pytest.raises(ValueError, match="appears in multiple subspaces"):
            sf_config.to_ormas_config()
