"""Advanced tests for ORMAS-CI density matrices, CI transforms, and CASSCF.

Validates the two-particle RDM, spin-separated RDMs, CI vector
transformations under orbital rotations, contract_2e / absorb_h1e,
the large_ci / dump_flags convenience helpers, and end-to-end CASSCF
integration.

Reference values come from PySCF's ``fci.direct_spin1`` solver.
"""

import numpy as np
from scipy.stats import ortho_group

from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.ormas_ci.determinants import build_determinant_list
from pyscf.ormas_ci.fcisolver import ORMASFCISolver
from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace

# ------------------------------------------------------------------ #
#  Molecule fixtures                                                  #
# ------------------------------------------------------------------ #

def _h2_setup():
    """H2 / STO-3G, CAS(2,2), (1a, 1b).

    Returns (mf, ncas, nelecas, config).
    """
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 2, (1, 1)
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    return mf, ncas, nelecas, config


def _h2o_setup():
    """H2O / cc-pVTZ, CAS(4,4), (3a, 3b) -- unrestricted single subspace.

    Returns (mf, ncas, nelecas, config).
    """
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVTZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 4, (3, 3)
    config = ORMASConfig(
        subspaces=[Subspace("all", list(range(ncas)), 0, 2 * ncas)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    return mf, ncas, nelecas, config


def _ch2_triplet_setup():
    """CH2 triplet / STO-3G, CAS(6,6), (4a, 2b), ROHF, spin=2.

    Returns (mf, ncas, nelecas, config).
    """
    mol = gto.M(
        atom="C 0 0 0; H 0 0.935 0.535; H 0 -0.935 0.535",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 6, (4, 2)
    config = ORMASConfig(
        subspaces=[Subspace("all", list(range(ncas)), 0, 2 * ncas)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    return mf, ncas, nelecas, config


def _run_casci(mf, ncas, nelecas, config):
    """Run CASCI with our solver, return the CASCI object.

    Also stores h1e, ecore, h2e on the returned object as attributes
    ``_h1e``, ``_ecore``, ``_h2e`` for convenient reuse.
    """
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)
    mc._h1e = h1e
    mc._ecore = ecore
    mc._h2e = h2e
    return mc


# ------------------------------------------------------------------ #
#  2-RDM symmetry tests                                               #
# ------------------------------------------------------------------ #

def test_rdm2_symmetry_aa():
    """rdm2aa antisymmetry under creation-index swap.

    For rdm2aa[p,q,r,s] = <a+_p a+_r a_s a_q>, swapping the two
    creation indices (p <-> r) flips sign:
        rdm2aa[p,q,r,s] = -rdm2aa[r,q,p,s]
    """
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    (_, _), (rdm2aa, _, _) = mc.fcisolver.make_rdm12s(
        mc.ci, ncas, nelecas
    )
    diff = rdm2aa + rdm2aa.transpose(2, 1, 0, 3)
    assert np.allclose(diff, 0, atol=1e-12), (
        f"rdm2aa antisymmetry violated, "
        f"max |err| = {np.max(np.abs(diff))}"
    )


def test_rdm2_symmetry_bb():
    """rdm2bb antisymmetry under creation-index swap.

    For rdm2bb[p,q,r,s] = <a+_p a+_r a_s a_q>, swapping the two
    creation indices (p <-> r) flips sign:
        rdm2bb[p,q,r,s] = -rdm2bb[r,q,p,s]
    """
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    (_, _), (_, _, rdm2bb) = mc.fcisolver.make_rdm12s(
        mc.ci, ncas, nelecas
    )
    diff = rdm2bb + rdm2bb.transpose(2, 1, 0, 3)
    assert np.allclose(diff, 0, atol=1e-12), (
        f"rdm2bb antisymmetry violated, "
        f"max |err| = {np.max(np.abs(diff))}"
    )


def test_rdm2_trace():
    """Trace einsum('ppqq', rdm2) must equal N*(N-1).

    For rdm2[p,q,r,s] = <a+_p a+_r a_s a_q>, contracting p=q and r=s
    gives the pair-counting sum over all electron pairs.
    """
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    _, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    n_elec = sum(nelecas)
    trace = np.einsum("ppqq", rdm2)
    expected = n_elec * (n_elec - 1)
    assert abs(trace - expected) < 1e-10, (
        f"Tr(rdm2) = {trace}, expected {expected}"
    )


# ------------------------------------------------------------------ #
#  2-RDM energy consistency tests                                     #
# ------------------------------------------------------------------ #

def test_rdm2_energy_h2():
    """Energy from RDMs must match kernel energy for H2/STO-3G."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    e_kernel = mc.e_tot
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e_rdm = (
        np.einsum("pq,qp", mc._h1e, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", mc._h2e, rdm2)
        + mc._ecore
    )
    assert abs(e_kernel - e_rdm) < 1e-10, (
        f"H2 energy mismatch: kernel={e_kernel}, rdm={e_rdm}"
    )


def test_rdm2_energy_h2o():
    """Energy from RDMs must match kernel energy for H2O/cc-pVTZ."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    e_kernel = mc.e_tot
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e_rdm = (
        np.einsum("pq,qp", mc._h1e, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", mc._h2e, rdm2)
        + mc._ecore
    )
    assert abs(e_kernel - e_rdm) < 1e-10, (
        f"H2O energy mismatch: kernel={e_kernel}, rdm={e_rdm}"
    )


def test_rdm2_matches_pyscf_h2():
    """Unrestricted ORMAS rdm2 must match PySCF FCI rdm2 for H2."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)

    # PySCF FCI reference
    e_fci, ci_fci = fci.direct_spin1.kernel(
        mc._h1e, mc._h2e, ncas, nelecas
    )
    rdm1_ref, rdm2_ref = fci.direct_spin1.make_rdm12(
        ci_fci, ncas, nelecas
    )
    assert np.allclose(rdm1, rdm1_ref, atol=1e-10), (
        f"rdm1 mismatch, max |err| = {np.max(np.abs(rdm1 - rdm1_ref))}"
    )
    assert np.allclose(rdm2, rdm2_ref, atol=1e-10), (
        f"rdm2 mismatch, max |err| = {np.max(np.abs(rdm2 - rdm2_ref))}"
    )


# ------------------------------------------------------------------ #
#  Spin-separated RDM consistency tests                               #
# ------------------------------------------------------------------ #

def test_rdm12s_consistency():
    """rdm2 must equal rdm2aa + rdm2ab + rdm2ab.T(2301) + rdm2bb."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    _, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    (_, _), (rdm2aa, rdm2ab, rdm2bb) = mc.fcisolver.make_rdm12s(
        mc.ci, ncas, nelecas
    )
    rdm2_sum = (
        rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb
    )
    assert np.allclose(rdm2, rdm2_sum, atol=1e-12), (
        "rdm2 != rdm2aa + rdm2ab + rdm2ba + rdm2bb, "
        f"max |err| = {np.max(np.abs(rdm2 - rdm2_sum))}"
    )


def test_rdm1s_from_rdm12s():
    """rdm1s from make_rdm12s must match make_rdm1s."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rdm1a_direct, rdm1b_direct = mc.fcisolver.make_rdm1s(
        mc.ci, ncas, nelecas
    )
    (rdm1a_12, rdm1b_12), _ = mc.fcisolver.make_rdm12s(
        mc.ci, ncas, nelecas
    )
    assert np.allclose(rdm1a_direct, rdm1a_12, atol=1e-12), (
        "rdm1a mismatch between make_rdm1s and make_rdm12s"
    )
    assert np.allclose(rdm1b_direct, rdm1b_12, atol=1e-12), (
        "rdm1b mismatch between make_rdm1s and make_rdm12s"
    )


def test_rdm1s_sum_equals_rdm1():
    """rdm1a + rdm1b must equal the spin-traced make_rdm1."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
    rdm1a, rdm1b = mc.fcisolver.make_rdm1s(mc.ci, ncas, nelecas)
    assert np.allclose(rdm1, rdm1a + rdm1b, atol=1e-12), (
        "rdm1 != rdm1a + rdm1b"
    )


def test_rdm1s_trace():
    """Tr(rdm1a) = n_alpha, Tr(rdm1b) = n_beta."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rdm1a, rdm1b = mc.fcisolver.make_rdm1s(mc.ci, ncas, nelecas)
    na, nb = nelecas
    assert abs(np.trace(rdm1a) - na) < 1e-10, (
        f"Tr(rdm1a) = {np.trace(rdm1a)}, expected {na}"
    )
    assert abs(np.trace(rdm1b) - nb) < 1e-10, (
        f"Tr(rdm1b) = {np.trace(rdm1b)}, expected {nb}"
    )


def test_rdm2_open_shell():
    """RDM energy match for CH2 triplet (open-shell, 4a/2b)."""
    mf, ncas, nelecas, config = _ch2_triplet_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    e_kernel = mc.e_tot
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e_rdm = (
        np.einsum("pq,qp", mc._h1e, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", mc._h2e, rdm2)
        + mc._ecore
    )
    assert abs(e_kernel - e_rdm) < 1e-10, (
        f"CH2 triplet energy mismatch: kernel={e_kernel}, rdm={e_rdm}"
    )


def test_rdm2_restricted_ormas():
    """RDM energy consistency with a 2-subspace restriction on H2O."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
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
    mc = _run_casci(mf, ncas, nelecas, config)
    e_kernel = mc.e_tot
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e_rdm = (
        np.einsum("pq,qp", mc._h1e, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", mc._h2e, rdm2)
        + mc._ecore
    )
    assert abs(e_kernel - e_rdm) < 1e-10, (
        f"Restricted ORMAS energy mismatch: kernel={e_kernel}, "
        f"rdm={e_rdm}"
    )


# ------------------------------------------------------------------ #
#  transform_ci tests                                                 #
# ------------------------------------------------------------------ #

def test_transform_identity():
    """Identity rotation must leave CI coefficients unchanged."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    u = np.eye(ncas)
    ci_new = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, u
    )
    assert np.allclose(ci_new, mc.ci, atol=1e-12), (
        f"Identity transform changed CI, max |diff| = "
        f"{np.max(np.abs(ci_new - mc.ci))}"
    )


def test_transform_roundtrip():
    """Applying U then U.T must recover the original CI vector."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rng = np.random.default_rng(42)
    u = ortho_group.rvs(ncas, random_state=rng)
    ci_rot = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, u
    )
    ci_back = mc.fcisolver.transform_ci_for_orbital_rotation(
        ci_rot, ncas, nelecas, u.T
    )
    assert np.allclose(ci_back, mc.ci, atol=1e-10), (
        f"Round-trip failed, max |diff| = "
        f"{np.max(np.abs(ci_back - mc.ci))}"
    )


def test_transform_norm_preserved():
    """A random unitary must preserve the CI vector norm."""
    mf, ncas, nelecas, config = _h2o_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rng = np.random.default_rng(123)
    u = ortho_group.rvs(ncas, random_state=rng)
    ci_rot = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, u
    )
    norm_orig = np.linalg.norm(mc.ci)
    norm_rot = np.linalg.norm(ci_rot)
    assert abs(norm_orig - norm_rot) < 1e-10, (
        f"Norm not preserved: original={norm_orig}, rotated={norm_rot}"
    )


def test_transform_open_shell():
    """CI transform must work for open-shell CH2 (4a, 2b)."""
    mf, ncas, nelecas, config = _ch2_triplet_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    rng = np.random.default_rng(7)
    u = ortho_group.rvs(ncas, random_state=rng)
    ci_rot = mc.fcisolver.transform_ci_for_orbital_rotation(
        mc.ci, ncas, nelecas, u
    )
    ci_back = mc.fcisolver.transform_ci_for_orbital_rotation(
        ci_rot, ncas, nelecas, u.T
    )
    assert np.allclose(ci_back, mc.ci, atol=1e-10), (
        f"CH2 round-trip failed, max |diff| = "
        f"{np.max(np.abs(ci_back - mc.ci))}"
    )


# ------------------------------------------------------------------ #
#  contract_2e / absorb_h1e tests                                     #
# ------------------------------------------------------------------ #

def test_contract_matches_hamiltonian():
    """contract_2e(eri, c) must equal H @ c."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    solver = mc.fcisolver

    # Build the full Hamiltonian explicitly
    alpha_strings, beta_strings = build_determinant_list(config)
    h_ci = build_ci_hamiltonian(
        alpha_strings, beta_strings, mc._h1e, mc._h2e
    )
    sigma_ref = h_ci @ mc.ci

    # contract_2e through the solver interface
    # absorb_h1e output is ignored inside contract_2e, but we must
    # pass something for interface compatibility.
    h2e_absorbed = solver.absorb_h1e(
        mc._h1e, mc._h2e, ncas, nelecas, fac=1
    )
    sigma = solver.contract_2e(h2e_absorbed, mc.ci, ncas, nelecas)

    assert np.allclose(sigma, sigma_ref, atol=1e-10), (
        f"contract_2e mismatch, max |err| = "
        f"{np.max(np.abs(sigma - sigma_ref))}"
    )


def test_absorb_h1e_matches_pyscf():
    """Our absorb_h1e output must match pyscf.fci.direct_spin1."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    solver = mc.fcisolver

    h2e_ours = solver.absorb_h1e(
        mc._h1e, mc._h2e, ncas, nelecas, fac=0.5
    )

    h2e_ref = fci.direct_spin1.absorb_h1e(
        mc._h1e, mc._h2e, ncas, nelecas, fac=0.5
    )
    # PySCF returns a flat array in 4-fold symmetry; reshape both
    # for comparison.
    h2e_ours_full = ao2mo.restore(1, h2e_ours, ncas)
    h2e_ref_full = ao2mo.restore(1, h2e_ref, ncas)

    assert np.allclose(h2e_ours_full, h2e_ref_full, atol=1e-12), (
        f"absorb_h1e mismatch, max |err| = "
        f"{np.max(np.abs(h2e_ours_full - h2e_ref_full))}"
    )


# ------------------------------------------------------------------ #
#  Convenience method tests                                           #
# ------------------------------------------------------------------ #

def test_large_ci_returns_dominant():
    """large_ci with default tol must return configs with |c| > tol."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    result = mc.fcisolver.large_ci(mc.ci, ncas, nelecas, tol=0.1)
    assert len(result) >= 1, "large_ci returned empty list"
    for coeff, a_str, b_str in result:
        assert abs(coeff) > 0.1, (
            f"Coefficient {coeff} does not exceed tol=0.1"
        )


def test_large_ci_threshold():
    """large_ci with tol=0 must return all determinants."""
    mf, ncas, nelecas, config = _h2_setup()
    mc = _run_casci(mf, ncas, nelecas, config)
    result = mc.fcisolver.large_ci(mc.ci, ncas, nelecas, tol=0)
    n_det = len(mc.ci)
    assert len(result) == n_det, (
        f"tol=0 returned {len(result)} configs, expected {n_det}"
    )


def test_dump_flags_no_error():
    """dump_flags must run without raising."""
    _, _, _, config = _h2_setup()
    solver = ORMASFCISolver(config)
    solver.dump_flags()
    solver.dump_flags(verbose=0)


# ------------------------------------------------------------------ #
#  CASSCF integration tests                                           #
# ------------------------------------------------------------------ #

def test_casscf_h2_runs():
    """CASSCF must complete on H2/STO-3G with our solver."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 2, (1, 1)
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    assert mc.converged or mc.e_tot < 0, (
        "CASSCF on H2 did not converge and energy is non-negative"
    )


def test_casscf_h2o_runs():
    """CASSCF must complete on H2O/STO-3G with our solver."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
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
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()
    assert mc.converged or mc.e_tot < 0, (
        "CASSCF on H2O did not converge and energy is non-negative"
    )


def test_casscf_energy_leq_casci():
    """CASSCF energy must be <= CASCI energy (orbital optimization)."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    ncas, nelecas = 2, (1, 1)
    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )

    # CASCI energy (no orbital optimization)
    mc_ci = mcscf.CASCI(mf, ncas, nelecas)
    mc_ci.verbose = 0
    mc_ci.fcisolver = ORMASFCISolver(config)
    e_casci = mc_ci.kernel()[0]

    # CASSCF energy (with orbital optimization)
    mc_scf = mcscf.CASSCF(mf, ncas, nelecas)
    mc_scf.verbose = 0
    mc_scf.fcisolver = ORMASFCISolver(config)
    e_casscf = mc_scf.kernel()[0]

    assert e_casscf <= e_casci + 1e-8, (
        f"CASSCF energy ({e_casscf}) > CASCI energy ({e_casci})"
    )
