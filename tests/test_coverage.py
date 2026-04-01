"""Coverage-gap tests for ORMAS-CI: multi-root, open-shell, 3-subspace, spin, RDM.

These tests target areas not covered by test_fcisolver.py or test_molecules.py:
- Multi-root eigenvalue problems (nroots > 1)
- Open-shell / ROHF reference calculations (triplet CH2 and O2)
- 3-subspace ORMAS partitioning (BeH2 and N2)
- spin_square computation for singlet and triplet states
- RDM properties on open-shell systems and multi-root calculations
"""

import numpy as np
import pytest

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci.determinants import (
    build_determinant_list,
    casci_determinant_count,
    count_determinants,
)
from pyscf.ormas_ci.fcisolver import ORMASFCISolver
from pyscf.ormas_ci.subspaces import ORMASConfig, Subspace

# ---------------------------------------------------------------------------
# Shared molecule fixtures
# ---------------------------------------------------------------------------


def _h2_setup():
    """H2 6-31G, CAS(2,2)."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 2, (1, 1)


def _lih_setup():
    """LiH 6-31G, CAS(4,4)."""
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 4, (2, 2)


def _ch2_triplet_setup():
    """CH2 triplet 6-31G with ROHF, CAS(6, (4,2))."""
    mol = gto.M(
        atom="C 0 0 0; H 0 0.93 0.54; H 0 -0.93 0.54",
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 6, (4, 2)


def _o2_triplet_setup():
    """O2 triplet 6-31G with ROHF, CAS(4, (3,1)).

    O2/6-31G has 10 MOs and 16 electrons.  With spin=2 the alpha/beta
    counts are (9, 7).  CAS(4, (3,1)) keeps ncore=6 and nvir=0, which
    is the largest feasible active space in this basis.
    """
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.21",
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 4, (3, 1)


def _beh2_setup():
    """BeH2 6-31G, CAS(6,6)."""
    mol = gto.M(
        atom="Be 0 0 0; H 0 0 1.3; H 0 0 -1.3",
        basis="6-31g",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 6, (3, 3)


def _n2_setup():
    """N2 6-31G, CAS(6,6)."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.09", basis="6-31g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return mol, mf, 6, (3, 3)


def _unrestricted_config(ncas, nelecas):
    """Single-subspace unrestricted ORMAS config (equivalent to CASCI)."""
    return ORMASConfig(
        subspaces=[
            Subspace("all", list(range(ncas)), 0, 2 * ncas),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )


# ---------------------------------------------------------------------------
# Multi-root tests
# ---------------------------------------------------------------------------


class TestMultiRoot:
    """Multi-root eigenvalue tests (nroots > 1)."""

    def test_h2_multi_root_energies(self):
        """H2 with nroots=2: verify 2 energies returned, E[0] < E[1]."""
        _, mf, ncas, nelecas = _h2_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.nroots = 2
        mc.kernel()

        energies = mc.e_tot
        assert len(energies) == 2
        assert energies[0] < energies[1]

    def test_h2_multi_root_matches_pyscf(self):
        """H2 nroots=4: all eigenvalues match PySCF FCI."""
        _, mf, ncas, nelecas = _h2_setup()

        # PySCF reference
        mc_ref = mcscf.CASCI(mf, ncas, 2)
        mc_ref.verbose = 0
        mc_ref.fcisolver.nroots = 4
        mc_ref.kernel()
        e_ref = mc_ref.e_tot

        # ORMAS unrestricted
        config = _unrestricted_config(ncas, nelecas)
        mc = mcscf.CASCI(mf, ncas, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.nroots = 4
        mc.kernel()
        e_ormas = mc.e_tot

        for i in range(4):
            assert abs(e_ref[i] - e_ormas[i]) < 1e-10, (
                f"Root {i}: ref={e_ref[i]}, ormas={e_ormas[i]}"
            )

    def test_lih_multi_root_three_states(self):
        """LiH CAS(4,4) nroots=3: matches PySCF CASCI nroots=3."""
        _, mf, ncas, nelecas = _lih_setup()

        # PySCF reference
        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        mc_ref.fcisolver.nroots = 3
        mc_ref.kernel()
        e_ref = mc_ref.e_tot

        # ORMAS unrestricted
        config = _unrestricted_config(ncas, nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.nroots = 3
        mc.kernel()
        e_ormas = mc.e_tot

        for i in range(3):
            assert abs(e_ref[i] - e_ormas[i]) < 1e-10, (
                f"Root {i}: ref={e_ref[i]}, ormas={e_ormas[i]}"
            )


# ---------------------------------------------------------------------------
# Open-shell / ROHF tests
# ---------------------------------------------------------------------------


class TestOpenShell:
    """Open-shell calculations with ROHF reference."""

    def test_ch2_triplet_rohf_unrestricted(self):
        """CH2 triplet with ROHF: unrestricted ORMAS matches CASCI."""
        _, mf, ncas, nelecas = _ch2_triplet_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = _unrestricted_config(ncas, nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert abs(e_ref - e_ormas) < 1e-10, f"CH2 triplet: ref={e_ref}, ormas={e_ormas}"

    def test_o2_triplet_rohf_unrestricted(self):
        """O2 triplet with ROHF: unrestricted ORMAS matches CASCI."""
        _, mf, ncas, nelecas = _o2_triplet_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = _unrestricted_config(ncas, nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert abs(e_ref - e_ormas) < 1e-10, f"O2 triplet: ref={e_ref}, ormas={e_ormas}"

    def test_ch2_triplet_restricted_variational(self):
        """CH2 triplet with 2 subspaces: energy >= CASCI."""
        _, mf, ncas, nelecas = _ch2_triplet_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = ORMASConfig(
            subspaces=[
                Subspace(
                    "core",
                    [0, 1, 2],
                    min_electrons=2,
                    max_electrons=6,
                ),
                Subspace(
                    "virt",
                    [3, 4, 5],
                    min_electrons=0,
                    max_electrons=4,
                ),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert e_ormas >= e_ref - 1e-10, (
            f"Variational violation: restricted={e_ormas} < CASCI={e_ref}"
        )

    def test_o2_triplet_restricted_variational(self):
        """O2 triplet with 2 restricted subspaces: energy >= CASCI."""
        _, mf, ncas, nelecas = _o2_triplet_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = ORMASConfig(
            subspaces=[
                Subspace(
                    "core",
                    [0, 1],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "virt",
                    [2, 3],
                    min_electrons=0,
                    max_electrons=3,
                ),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert e_ormas >= e_ref - 1e-10, (
            f"Variational violation: restricted={e_ormas} < CASCI={e_ref}"
        )


# ---------------------------------------------------------------------------
# 3-subspace ORMAS tests
# ---------------------------------------------------------------------------


class TestThreeSubspaces:
    """Tests with 3 explicit ORMAS subspaces."""

    def test_beh2_three_subspaces(self):
        """BeH2 with 3 ORMAS subspaces: variational bound + det reduction."""
        _, mf, ncas, nelecas = _beh2_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = ORMASConfig(
            subspaces=[
                Subspace(
                    "sigma_g",
                    [0, 1],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "sigma_u",
                    [2, 3],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "delta",
                    [4, 5],
                    min_electrons=0,
                    max_electrons=4,
                ),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert e_ormas >= e_ref - 1e-10, (
            f"Variational violation: restricted={e_ormas} < CASCI={e_ref}"
        )

        n_ormas = count_determinants(config)
        n_casci = casci_determinant_count(ncas, nelecas)
        assert n_ormas < n_casci, f"Expected det reduction: ORMAS={n_ormas}, CASCI={n_casci}"

    def test_n2_three_subspace_ormas(self):
        """N2 with 3 explicit ORMAS subspaces: variational bound."""
        _, mf, ncas, nelecas = _n2_setup()

        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]

        config = ORMASConfig(
            subspaces=[
                Subspace(
                    "sigma",
                    [0, 1],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "pi",
                    [2, 3],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "sigma_star",
                    [4, 5],
                    min_electrons=0,
                    max_electrons=4,
                ),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        e_ormas = mc.kernel()[0]

        assert e_ormas >= e_ref - 1e-10, (
            f"Variational violation: restricted={e_ormas} < CASCI={e_ref}"
        )

    def test_three_subspace_constraints_satisfied(self):
        """All determinants satisfy per-subspace occupation bounds."""
        _, _, ncas, nelecas = _beh2_setup()

        config = ORMASConfig(
            subspaces=[
                Subspace(
                    "sigma_g",
                    [0, 1],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "sigma_u",
                    [2, 3],
                    min_electrons=1,
                    max_electrons=4,
                ),
                Subspace(
                    "delta",
                    [4, 5],
                    min_electrons=0,
                    max_electrons=4,
                ),
            ],
            n_active_orbitals=ncas,
            nelecas=nelecas,
        )
        alpha_strings, beta_strings = build_determinant_list(config)

        for idx in range(len(alpha_strings)):
            a_str = int(alpha_strings[idx])
            b_str = int(beta_strings[idx])

            for sub in config.subspaces:
                occ = 0
                for orb in sub.orbital_indices:
                    if a_str & (1 << orb):
                        occ += 1
                    if b_str & (1 << orb):
                        occ += 1
                assert sub.min_electrons <= occ <= sub.max_electrons, (
                    f"Det {idx}: subspace '{sub.name}' has "
                    f"{occ} electrons, bounds="
                    f"[{sub.min_electrons}, {sub.max_electrons}]"
                )


# ---------------------------------------------------------------------------
# spin_square tests
# ---------------------------------------------------------------------------


class TestSpinSquare:
    """Tests for spin_square (<S^2> expectation value)."""

    def test_spin_square_h2_singlet(self):
        """H2 singlet ground state: <S^2>=0, 2S+1=1."""
        _, mf, ncas, nelecas = _h2_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, nelecas)
        assert abs(ss) < 1e-10, f"Expected <S^2>=0 for singlet, got {ss}"
        assert abs(mult - 1.0) < 1e-10, f"Expected 2S+1=1 for singlet, got {mult}"

    def test_spin_square_h2_triplet(self):
        """H2 excited state (triplet ms=0): <S^2>=2, 2S+1=3."""
        _, mf, ncas, nelecas = _h2_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, 2)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.fcisolver.nroots = 4
        mc.kernel()

        civecs = mc.ci
        energies = mc.e_tot

        # Find the root with <S^2> closest to 2.0 (triplet)
        found_triplet = False
        for i in range(len(energies)):
            ss, mult = mc.fcisolver.spin_square(civecs[i], ncas, nelecas)
            if abs(ss - 2.0) < 0.1:
                found_triplet = True
                assert abs(ss - 2.0) < 1e-6, f"Triplet root {i}: expected <S^2>=2, got {ss}"
                assert abs(mult - 3.0) < 1e-6, f"Triplet root {i}: expected 2S+1=3, got {mult}"
                break
        assert found_triplet, "No triplet state found among excited roots"

    def test_spin_square_ch2_triplet(self):
        """CH2 triplet: <S^2>=2, 2S+1=3."""
        _, mf, ncas, nelecas = _ch2_triplet_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, nelecas)
        assert abs(ss - 2.0) < 1e-6, f"CH2 triplet: expected <S^2>=2, got {ss}"
        assert abs(mult - 3.0) < 1e-6, f"CH2 triplet: expected 2S+1=3, got {mult}"

    def test_spin_square_o2_triplet(self):
        """O2 triplet: <S^2>=2, 2S+1=3."""
        _, mf, ncas, nelecas = _o2_triplet_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, nelecas)
        assert abs(ss - 2.0) < 1e-6, f"O2 triplet: expected <S^2>=2, got {ss}"
        assert abs(mult - 3.0) < 1e-6, f"O2 triplet: expected 2S+1=3, got {mult}"

    def test_spin_square_raises_before_kernel(self):
        """spin_square raises RuntimeError if kernel() not called."""
        config = _unrestricted_config(2, (1, 1))
        solver = ORMASFCISolver(config)

        with pytest.raises(RuntimeError, match="kernel"):
            solver.spin_square(None, 2, (1, 1))


# ---------------------------------------------------------------------------
# Open-shell RDM tests
# ---------------------------------------------------------------------------


class TestOpenShellRDM:
    """RDM tests on open-shell systems and multi-root calculations."""

    def test_rdm1_trace_open_shell(self):
        """CH2 triplet: trace(rdm1) = n_alpha + n_beta = 6."""
        _, mf, ncas, nelecas = _ch2_triplet_setup()
        config = _unrestricted_config(ncas, nelecas)

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

        rdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
        n_elec = nelecas[0] + nelecas[1]
        assert abs(np.trace(rdm1) - n_elec) < 1e-10, (
            f"RDM1 trace = {np.trace(rdm1)}, expected {n_elec}"
        )

    def test_rdm1_open_shell_matches_pyscf(self):
        """CH2 triplet: unrestricted ORMAS rdm1 matches PySCF."""
        _, mf, ncas, nelecas = _ch2_triplet_setup()

        # PySCF reference
        mc_ref = mcscf.CASCI(mf, ncas, nelecas)
        mc_ref.verbose = 0
        mc_ref.kernel()
        rdm1_ref = mc_ref.fcisolver.make_rdm1(mc_ref.ci, ncas, nelecas)

        # ORMAS unrestricted
        config = _unrestricted_config(ncas, nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()
        rdm1_ormas = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)

        np.testing.assert_allclose(
            rdm1_ormas,
            rdm1_ref,
            atol=1e-8,
            err_msg="CH2 triplet: ORMAS rdm1 != PySCF rdm1",
        )

    def test_rdm1_multi_root_ground_state(self):
        """H2 nroots=2: ground-state rdm1 matches single-root rdm1."""
        _, mf, ncas, nelecas = _h2_setup()
        config = _unrestricted_config(ncas, nelecas)

        # Single-root reference
        mc1 = mcscf.CASCI(mf, ncas, 2)
        mc1.verbose = 0
        mc1.fcisolver = ORMASFCISolver(config)
        mc1.kernel()
        rdm1_single = mc1.fcisolver.make_rdm1(mc1.ci, ncas, nelecas)

        # Multi-root, extract ground state
        config2 = _unrestricted_config(ncas, nelecas)
        mc2 = mcscf.CASCI(mf, ncas, 2)
        mc2.verbose = 0
        mc2.fcisolver = ORMASFCISolver(config2)
        mc2.fcisolver.nroots = 2
        mc2.kernel()
        civecs = mc2.ci
        rdm1_multi = mc2.fcisolver.make_rdm1(civecs[0], ncas, nelecas)

        np.testing.assert_allclose(
            rdm1_multi,
            rdm1_single,
            atol=1e-10,
            err_msg="Multi-root ground-state rdm1 != single-root rdm1",
        )
