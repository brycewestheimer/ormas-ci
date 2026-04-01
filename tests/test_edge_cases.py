"""Edge case tests for small and degenerate active spaces."""

import warnings

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci.determinants import build_determinant_list, count_determinants
from pyscf.ormas_ci.fcisolver import ORMASFCISolver
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace


def test_single_orbital_two_electrons():
    """CAS(2,1): single orbital, two electrons -> one determinant."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0], min_electrons=0, max_electrons=2)],
        n_active_orbitals=1,
        nelecas=(1, 1),
    )
    config.validate()
    assert count_determinants(config) == 1

    alpha, beta = build_determinant_list(config)
    assert len(alpha) == 1
    assert alpha[0] == 0b1
    assert beta[0] == 0b1


def test_single_orbital_one_electron_alpha():
    """CAS(1,1): single orbital, one alpha electron -> one determinant."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0], min_electrons=0, max_electrons=1)],
        n_active_orbitals=1,
        nelecas=(1, 0),
    )
    config.validate()
    assert count_determinants(config) == 1

    alpha, beta = build_determinant_list(config)
    assert len(alpha) == 1
    assert alpha[0] == 0b1
    assert beta[0] == 0b0


def test_single_orbital_one_electron_beta():
    """CAS(1,1): single orbital, one beta electron -> one determinant."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0], min_electrons=0, max_electrons=1)],
        n_active_orbitals=1,
        nelecas=(0, 1),
    )
    config.validate()
    assert count_determinants(config) == 1


def test_nroots_greater_than_ndet():
    """Requesting more roots than determinants warns and returns available roots."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    # CAS(2,2) with single subspace -> 4 determinants
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.fcisolver.nroots = 10  # More than 4 determinants

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mc.kernel()
        root_warnings = [x for x in w if "roots" in str(x.message).lower()]
        assert len(root_warnings) > 0, "Expected a warning about fewer roots"


def test_nroots_equals_ndet():
    """Requesting exactly n_det roots returns all eigenvalues."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    n_det = count_determinants(config)

    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.fcisolver.nroots = n_det

    result = mc.kernel()
    e_tot = result[0]
    assert len(e_tot) == n_det
    # Eigenvalues should be sorted ascending
    for i in range(n_det - 1):
        assert e_tot[i] <= e_tot[i + 1] + 1e-10


def test_empty_subspace_zero_electrons():
    """Subspace with min=max=0 electrons is valid when combined with active subspace."""
    config = ORMASConfig(
        subspaces=[
            Subspace("active", [0, 1], min_electrons=0, max_electrons=4),
            Subspace("frozen_virtual", [2], min_electrons=0, max_electrons=0),
        ],
        n_active_orbitals=3,
        nelecas=(1, 1),
    )
    config.validate()
    n_det = count_determinants(config)
    # Same as CAS(2,2) since the third orbital is always empty
    config_ref = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    assert n_det == count_determinants(config_ref)
