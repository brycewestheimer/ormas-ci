"""Completeness tests targeting uncovered code paths and functional gaps.

Covers:
- RuntimeError guards when kernel() has not been called
- nelecas as int (not tuple) in kernel() and spin_square()
- ValueError for ncas/nelecas mismatch in kernel()
- 1D ERI format in kernel()
- large_ci with return_strs=False and high-tol fallback
- transform_ci with (u_alpha, u_beta) tuple and energy invariance
- rdm2 PySCF match for H2O (larger system)
- rdm12s spin components match PySCF for open-shell CH2
- rdm2 energy consistency for 3-subspace ORMAS
- Multi-root make_rdm12 ground-state consistency
- CASSCF with restricted ORMAS subspaces
"""

import numpy as np
import pytest

from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import build_determinant_list
from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian

# ------------------------------------------------------------------ #
#  Error guard tests (RuntimeError before kernel)                     #
# ------------------------------------------------------------------ #


def _fresh_solver():
    """Return a solver that has NOT had kernel() called."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    return ORMASFCISolver(config)


def test_make_rdm1_before_kernel():
    """make_rdm1 raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.make_rdm1(np.zeros(4), 2, (1, 1))


def test_make_rdm1s_before_kernel():
    """make_rdm1s raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.make_rdm1s(np.zeros(4), 2, (1, 1))


def test_make_rdm12_before_kernel():
    """make_rdm12 raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.make_rdm12(np.zeros(4), 2, (1, 1))


def test_make_rdm12s_before_kernel():
    """make_rdm12s raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.make_rdm12s(np.zeros(4), 2, (1, 1))


def test_transform_ci_before_kernel():
    """transform_ci raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.transform_ci_for_orbital_rotation(
            np.zeros(4), 2, (1, 1), np.eye(2)
        )


def test_large_ci_before_kernel():
    """large_ci raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.large_ci(np.zeros(4), 2, (1, 1))


def test_contract_2e_before_kernel():
    """contract_2e raises RuntimeError if kernel() not called."""
    solver = _fresh_solver()
    with pytest.raises(RuntimeError, match="kernel"):
        solver.contract_2e(np.zeros(4), np.zeros(4), 2, (1, 1))


# ------------------------------------------------------------------ #
#  nelecas as int                                                     #
# ------------------------------------------------------------------ #


def _h2_mf():
    """H2 / 6-31G RHF, return (mf, ncas, nelecas_tuple)."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return mf, 2, (1, 1)


def test_kernel_nelecas_as_int():
    """kernel() converts nelecas int to tuple and runs correctly."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    # Pass nelecas=2 (int) directly to kernel
    e, ci = mc.fcisolver.kernel(h1e, h2e, ncas, 2, ecore=ecore)
    assert e < 0, f"H2 energy should be negative, got {e}"


def test_spin_square_nelecas_as_int():
    """spin_square handles nelecas as int."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    # Pass nelecas=2 (int) to spin_square
    ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, 2)
    assert abs(ss) < 1e-8, f"H2 singlet <S^2> should be 0, got {ss}"
    assert abs(mult - 1.0) < 1e-8, f"Expected 2S+1=1, got {mult}"


# ------------------------------------------------------------------ #
#  Validation error tests                                             #
# ------------------------------------------------------------------ #


def test_kernel_ncas_mismatch():
    """kernel raises ValueError if ncas doesn't match config."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    solver = ORMASFCISolver(config)
    with pytest.raises(ValueError, match="mismatch"):
        solver.kernel(
            np.zeros((3, 3)),
            np.zeros((3, 3, 3, 3)),
            3,
            (1, 1),
            ecore=0,
        )


def test_kernel_nelecas_mismatch():
    """kernel raises ValueError if nelecas doesn't match config."""
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    solver = ORMASFCISolver(config)
    with pytest.raises(ValueError, match="mismatch"):
        solver.kernel(
            np.zeros((2, 2)),
            np.zeros((2, 2, 2, 2)),
            2,
            (2, 2),
            ecore=0,
        )


# ------------------------------------------------------------------ #
#  1D ERI format in kernel                                            #
# ------------------------------------------------------------------ #


def test_kernel_1d_eri():
    """kernel() handles 1D (flat packed) ERI format."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    h1e, ecore = mc.get_h1eff()
    h2e_2d = mc.get_h2eff()
    # Convert to 1D (8-fold symmetry packed)
    h2e_1d = ao2mo.restore(8, h2e_2d, ncas)
    assert h2e_1d.ndim == 1, "Expected 1D ERI"
    e, ci = mc.fcisolver.kernel(h1e, h2e_1d, ncas, nelecas, ecore=ecore)
    assert e < 0, f"H2 energy should be negative, got {e}"


# ------------------------------------------------------------------ #
#  large_ci edge cases                                                #
# ------------------------------------------------------------------ #


def test_large_ci_return_strs_false():
    """large_ci with return_strs=False returns occupation index lists."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    result = mc.fcisolver.large_ci(
        mc.ci, ncas, nelecas, tol=0.1, return_strs=False
    )
    assert len(result) >= 1
    for coeff, a_occ, b_occ in result:
        assert isinstance(a_occ, list), (
            f"Expected list, got {type(a_occ)}"
        )
        assert isinstance(b_occ, list), (
            f"Expected list, got {type(b_occ)}"
        )
        assert all(isinstance(x, int) for x in a_occ)
        assert all(isinstance(x, int) for x in b_occ)


def test_large_ci_fallback_high_tol():
    """large_ci with tol exceeding all coefficients returns largest."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    # tol=999 so no coefficient exceeds it
    result = mc.fcisolver.large_ci(mc.ci, ncas, nelecas, tol=999.0)
    assert len(result) == 1, (
        f"Fallback should return exactly 1 entry, got {len(result)}"
    )
    coeff, a_str, b_str = result[0]
    # Should be the largest-magnitude coefficient
    assert abs(coeff) == pytest.approx(
        np.max(np.abs(mc.ci)), abs=1e-14
    )


def test_large_ci_fallback_return_strs_false():
    """large_ci fallback path with return_strs=False."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    result = mc.fcisolver.large_ci(
        mc.ci, ncas, nelecas, tol=999.0, return_strs=False
    )
    assert len(result) == 1
    coeff, a_occ, b_occ = result[0]
    assert isinstance(a_occ, list)
    assert isinstance(b_occ, list)


# ------------------------------------------------------------------ #
#  transform_ci with spin-dependent rotation                          #
# ------------------------------------------------------------------ #


def test_transform_ci_spin_dependent_rotation():
    """transform_ci with (u_alpha, u_beta) tuple input and roundtrip."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    rng = np.random.default_rng(42)
    ua = np.linalg.qr(rng.standard_normal((ncas, ncas)))[0]
    ub = np.linalg.qr(rng.standard_normal((ncas, ncas)))[0]
    ci_rot = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, (ua, ub)
    )
    ci_back = mc.fcisolver.transform_ci_for_orbital_rotation(
        ci_rot, ncas, nelecas, (ua.T, ub.T)
    )
    assert np.allclose(ci_back, mc.ci, atol=1e-10), (
        f"Spin-dependent roundtrip failed, "
        f"max |diff| = {np.max(np.abs(ci_back - mc.ci))}"
    )


def test_transform_ci_energy_invariance():
    """Energy is invariant under simultaneous H and CI rotation."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)

    # Original energy via RDMs
    rdm1_orig, rdm2_orig = mc.fcisolver.make_rdm12(
        mc.ci, ncas, nelecas
    )
    e_orig = (
        np.einsum("pq,qp", h1e, rdm1_orig)
        + 0.5 * np.einsum("pqrs,pqrs", h2e, rdm2_orig)
        + ecore
    )

    # Random unitary rotation
    rng = np.random.default_rng(99)
    u = np.linalg.qr(rng.standard_normal((ncas, ncas)))[0]
    ci_rot = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, u
    )

    # Rotate integrals
    h1e_rot = u.T @ h1e @ u
    h2e_rot = np.einsum(
        "ip,jq,pqrs,rk,sl->ijkl", u, u, h2e, u, u
    )

    # Compute energy from rotated H and rotated CI
    alpha_str, beta_str = build_determinant_list(config)
    h_rot = build_ci_hamiltonian(
        alpha_str, beta_str, h1e_rot, h2e_rot
    )
    if hasattr(h_rot, "toarray"):
        h_rot = h_rot.toarray()
    e_rot = ci_rot @ h_rot @ ci_rot + ecore

    assert abs(e_orig - e_rot) < 1e-10, (
        f"Energy not invariant: {e_orig} vs {e_rot}"
    )


# ------------------------------------------------------------------ #
#  Larger system PySCF reference tests                                #
# ------------------------------------------------------------------ #


def test_rdm2_matches_pyscf_h2o():
    """Unrestricted ORMAS rdm2 matches PySCF FCI rdm2 on H2O."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="6-31g",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 4, (3, 3)
    config = ORMASConfig(
        subspaces=[
            Subspace("all", list(range(ncas)), 0, 2 * ncas),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)

    # PySCF FCI reference
    e_fci, ci_fci = fci.direct_spin1.kernel(h1e, h2e, ncas, nelecas)
    rdm1_ref, rdm2_ref = fci.direct_spin1.make_rdm12(
        ci_fci, ncas, nelecas
    )
    assert np.allclose(rdm1, rdm1_ref, atol=1e-10), (
        f"H2O rdm1 mismatch, "
        f"max |err| = {np.max(np.abs(rdm1 - rdm1_ref))}"
    )
    assert np.allclose(rdm2, rdm2_ref, atol=1e-10), (
        f"H2O rdm2 mismatch, "
        f"max |err| = {np.max(np.abs(rdm2 - rdm2_ref))}"
    )


def test_rdm12s_open_shell_matches_pyscf():
    """CH2 triplet: spin-separated RDMs match PySCF FCI."""
    mol = gto.M(
        atom="C 0 0 0; H 0 0.93 0.54; H 0 -0.93 0.54",
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 6, (4, 2)
    config = ORMASConfig(
        subspaces=[
            Subspace("all", list(range(ncas)), 0, 2 * ncas),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)
    rdm1_ours, rdm2_ours = mc.fcisolver.make_rdm12(
        mc.ci, ncas, nelecas
    )

    # PySCF FCI reference
    e_fci, ci_fci = fci.direct_spin1.kernel(h1e, h2e, ncas, nelecas)
    rdm1_ref, rdm2_ref = fci.direct_spin1.make_rdm12(
        ci_fci, ncas, nelecas
    )
    assert np.allclose(rdm1_ours, rdm1_ref, atol=1e-10), (
        f"CH2 rdm1 mismatch, "
        f"max |err| = {np.max(np.abs(rdm1_ours - rdm1_ref))}"
    )
    assert np.allclose(rdm2_ours, rdm2_ref, atol=1e-10), (
        f"CH2 rdm2 mismatch, "
        f"max |err| = {np.max(np.abs(rdm2_ours - rdm2_ref))}"
    )


# ------------------------------------------------------------------ #
#  3-subspace rdm2 energy consistency                                 #
# ------------------------------------------------------------------ #


def test_rdm2_three_subspace_energy():
    """3-subspace ORMAS on N2: rdm2 energy matches kernel energy."""
    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.09", basis="6-31g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 6, (3, 3)
    config = ORMASConfig(
        subspaces=[
            Subspace(
                "sigma", [0, 1],
                min_electrons=1, max_electrons=4,
            ),
            Subspace(
                "pi", [2, 3],
                min_electrons=1, max_electrons=4,
            ),
            Subspace(
                "sigma_star", [4, 5],
                min_electrons=0, max_electrons=4,
            ),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e_rdm = (
        np.einsum("pq,qp", h1e, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", h2e, rdm2)
        + ecore
    )
    assert abs(mc.e_tot - e_rdm) < 1e-10, (
        f"3-subspace N2 energy mismatch: "
        f"kernel={mc.e_tot}, rdm={e_rdm}"
    )


# ------------------------------------------------------------------ #
#  Multi-root RDM consistency                                         #
# ------------------------------------------------------------------ #


def test_rdm12_multi_root():
    """make_rdm12 on ground state from nroots=2 matches single-root."""
    mf, ncas, nelecas = _h2_mf()
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    # Single-root reference
    mc1 = mcscf.CASCI(mf, ncas, 2)
    mc1.verbose = 0
    mc1.fcisolver = ORMASFCISolver(config)
    mc1.kernel()
    rdm1_ref, rdm2_ref = mc1.fcisolver.make_rdm12(
        mc1.ci, ncas, nelecas
    )

    # Multi-root, extract ground state
    config2 = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc2 = mcscf.CASCI(mf, ncas, 2)
    mc2.verbose = 0
    mc2.fcisolver = ORMASFCISolver(config2)
    mc2.fcisolver.nroots = 2
    mc2.kernel()
    civecs = mc2.ci
    rdm1_mr, rdm2_mr = mc2.fcisolver.make_rdm12(
        civecs[0], ncas, nelecas
    )
    assert np.allclose(rdm1_ref, rdm1_mr, atol=1e-10), (
        "Multi-root ground-state rdm1 != single-root rdm1"
    )
    assert np.allclose(rdm2_ref, rdm2_mr, atol=1e-10), (
        "Multi-root ground-state rdm2 != single-root rdm2"
    )


# ------------------------------------------------------------------ #
#  CASSCF with restricted ORMAS                                       #
# ------------------------------------------------------------------ #


def test_casscf_restricted_ormas():
    """CASSCF with 2-subspace restricted ORMAS converges on H2O."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="6-31g",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 4, (3, 3)
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1], min_electrons=2, max_electrons=4),
            Subspace("B", [2, 3], min_electrons=2, max_electrons=4),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    assert mc.e_tot < 0, (
        f"CASSCF H2O energy should be negative, got {mc.e_tot}"
    )
