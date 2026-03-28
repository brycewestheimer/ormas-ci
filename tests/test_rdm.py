"""Tests for ormas_ci.rdm -- reduced density matrix construction.

Validates the one-particle RDM computed from CI eigenvectors against
known mathematical properties (trace, symmetry, eigenvalue bounds) and
against PySCF's FCI RDM1 as a reference.
"""

import numpy as np

from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.ormas_ci.determinants import build_determinant_list
from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
from pyscf.ormas_ci.rdm import make_rdm1
from pyscf.ormas_ci.solver import solve_ci
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace
from pyscf.ormas_ci.utils import generate_strings


def _h2_ci_solution():
    """Run full CI on H2/STO-3G through our ORMAS machinery.

    Returns:
        (ci_vector, alpha_strings, beta_strings, ncas, h1e, h2e, ecore)
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    h2e = ao2mo.restore(1, h2e, 2)

    config = ORMASConfig(
        subspaces=[Subspace("all", [0, 1], 0, 4)],
        n_active_orbitals=2,
        nelecas=(1, 1),
    )
    alpha_strings, beta_strings = build_determinant_list(config)
    h_ci = build_ci_hamiltonian(alpha_strings, beta_strings, h1e, h2e)
    _, ci_vectors = solve_ci(h_ci, n_roots=1)
    ci_vector = ci_vectors[:, 0]

    return ci_vector, alpha_strings, beta_strings, 2, h1e, h2e, ecore


def test_rdm1_trace():
    """Trace of RDM1 must equal total number of electrons."""
    ci_vector, alpha_strings, beta_strings, ncas, _, _, _ = (
        _h2_ci_solution()
    )
    rdm1 = make_rdm1(ci_vector, alpha_strings, beta_strings, ncas)
    assert abs(np.trace(rdm1) - 2.0) < 1e-10, (
        f"Trace of RDM1 = {np.trace(rdm1)}, expected 2.0"
    )


def test_rdm1_symmetry():
    """RDM1 must be symmetric: rdm1[p,q] == rdm1[q,p]."""
    ci_vector, alpha_strings, beta_strings, ncas, _, _, _ = (
        _h2_ci_solution()
    )
    rdm1 = make_rdm1(ci_vector, alpha_strings, beta_strings, ncas)
    assert np.allclose(rdm1, rdm1.T, atol=1e-12), (
        "RDM1 is not symmetric"
    )


def test_rdm1_eigenvalues_in_range():
    """All eigenvalues (occupation numbers) must be in [0, 2]."""
    ci_vector, alpha_strings, beta_strings, ncas, _, _, _ = (
        _h2_ci_solution()
    )
    rdm1 = make_rdm1(ci_vector, alpha_strings, beta_strings, ncas)
    eigenvalues = np.linalg.eigvalsh(rdm1)

    assert np.all(eigenvalues > -1e-10), (
        f"Negative eigenvalue found: {eigenvalues}"
    )
    assert np.all(eigenvalues < 2.0 + 1e-10), (
        f"Eigenvalue > 2 found: {eigenvalues}"
    )


def test_rdm1_matches_pyscf():
    """For unrestricted ORMAS (= full CI), RDM1 must match PySCF FCI RDM1."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    mc = mcscf.CASCI(mf, 2, 2)
    mc.verbose = 0
    h1e, ecore = mc.get_h1eff()
    h2e = mc.get_h2eff()
    h2e = ao2mo.restore(1, h2e, 2)

    # PySCF FCI reference RDM1
    _, ci_fci = fci.direct_spin1.kernel(h1e, h2e, 2, 2)
    rdm1_ref = fci.direct_spin1.make_rdm1(ci_fci, 2, 2)

    # Our RDM1 via ORMAS machinery
    # Build determinants in the same order as PySCF's FCI: all alpha x all beta
    alpha_strings = np.array(generate_strings(2, 1), dtype=np.int64)
    beta_strings_list = generate_strings(2, 1)

    # PySCF's ci vector is a (n_alpha_str, n_beta_str) matrix.
    # Our determinant list pairs each alpha with each beta.
    all_alpha = []
    all_beta = []
    for a in alpha_strings:
        for b in beta_strings_list:
            all_alpha.append(a)
            all_beta.append(b)
    all_alpha = np.array(all_alpha, dtype=np.int64)
    all_beta = np.array(all_beta, dtype=np.int64)

    # Build and solve through our code to get the CI vector in our basis
    h_ci = build_ci_hamiltonian(all_alpha, all_beta, h1e, h2e)
    _, ci_vectors = solve_ci(h_ci, n_roots=1)
    ci_vector = ci_vectors[:, 0]

    rdm1_ours = make_rdm1(ci_vector, all_alpha, all_beta, 2)

    assert np.allclose(rdm1_ours, rdm1_ref, atol=1e-10), (
        f"RDM1 mismatch:\nOurs:\n{rdm1_ours}\nPySCF:\n{rdm1_ref}"
    )


def test_rdm1_hf_state():
    """For a single-determinant state, RDM1 should be diagonal with 0/1/2.

    For H2 in the HF state: 1 alpha + 1 beta in orbital 0, giving
    rdm1[0,0] = 2.0 and rdm1[1,1] = 0.0.
    """
    # Single determinant: alpha=0b01 (orbital 0), beta=0b01 (orbital 0)
    alpha_strings = np.array([0b01], dtype=np.int64)
    beta_strings = np.array([0b01], dtype=np.int64)
    ci_vector = np.array([1.0])
    ncas = 2

    rdm1 = make_rdm1(ci_vector, alpha_strings, beta_strings, ncas)

    assert abs(rdm1[0, 0] - 2.0) < 1e-14, (
        f"rdm1[0,0] = {rdm1[0, 0]}, expected 2.0"
    )
    assert abs(rdm1[1, 1] - 0.0) < 1e-14, (
        f"rdm1[1,1] = {rdm1[1, 1]}, expected 0.0"
    )
    assert abs(rdm1[0, 1]) < 1e-14, (
        f"rdm1[0,1] = {rdm1[0, 1]}, expected 0.0"
    )
    assert abs(rdm1[1, 0]) < 1e-14, (
        f"rdm1[1,0] = {rdm1[1, 0]}, expected 0.0"
    )
