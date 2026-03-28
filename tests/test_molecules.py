"""End-to-end molecular tests for the ORMAS-CI solver.

Runs the solver on real molecules to verify chemical correctness:
- H2 dissociation curve (smooth, matches CASCI at each point)
- N2 with RASCI sigma/pi separation (variational bound)
- H2O with ORMAS subspaces (variational bound, determinant reduction)
"""

from pyscf import gto, mcscf, scf

from ormas_ci.determinants import casci_determinant_count, count_determinants
from ormas_ci.fcisolver import ORMASFCISolver
from ormas_ci.subspaces import ORMASConfig, RASConfig, Subspace


def test_h2_dissociation():
    """ORMAS-CI on H2 at multiple bond lengths matches CASCI at each point.

    Runs unrestricted ORMAS-CI (= full CASCI) at five bond lengths along
    the H2 dissociation curve. Verifies:
    1. Energy matches PySCF CASCI at each geometry.
    2. Energy curve is smooth (no discontinuous jumps).
    """
    bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0]
    energies_ref = []
    energies_ormas = []

    for r in bond_lengths:
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {r}",
            basis="sto-3g",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.run()

        # PySCF reference
        mc_ref = mcscf.CASCI(mf, 2, 2)
        mc_ref.verbose = 0
        e_ref = mc_ref.kernel()[0]
        energies_ref.append(e_ref)

        # Our solver (unrestricted = CASCI)
        config = ORMASConfig(
            subspaces=[
                Subspace("all", [0, 1], min_electrons=0, max_electrons=4),
            ],
            n_active_orbitals=2,
            nelecas=(1, 1),
        )
        mc_test = mcscf.CASCI(mf, 2, 2)
        mc_test.verbose = 0
        mc_test.fcisolver = ORMASFCISolver(config)
        e_test = mc_test.kernel()[0]
        energies_ormas.append(e_test)

        assert abs(e_ref - e_test) < 1e-10, (
            f"Energy mismatch at R={r}: PySCF={e_ref}, ORMAS={e_test}"
        )

    # Smoothness check: energy differences between adjacent points should
    # be moderate (no jumps larger than 1 Hartree for these geometries)
    for i in range(len(energies_ormas) - 1):
        delta = abs(energies_ormas[i + 1] - energies_ormas[i])
        assert delta < 1.0, (
            f"Suspicious energy jump between R={bond_lengths[i]} and "
            f"R={bond_lengths[i+1]}: delta={delta} Hartree"
        )


def test_n2_ras():
    """RASCI on N2/STO-3G with sigma/pi separation.

    Uses RASConfig with max_holes=1, max_particles=1 to define a
    restricted active space. Verifies:
    1. Energy is above (or equal to) full CASCI (variational bound).
    2. Determinant count is reduced relative to CASCI.
    """
    mol = gto.M(atom="N 0 0 0; N 0 0 1.09", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    # Active space: 6 orbitals, 6 electrons (3 alpha, 3 beta)
    ncas, nelecas = 6, (3, 3)

    # PySCF full CASCI reference
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # RAS: sigma (RAS1: 2 orbs), pi (RAS2: 2 orbs), sigma* (RAS3: 2 orbs)
    ras = RASConfig(
        ras1_orbitals=[0, 1],
        ras2_orbitals=[2, 3],
        ras3_orbitals=[4, 5],
        max_holes_ras1=1,
        max_particles_ras3=1,
        nelecas=nelecas,
    )
    config = ras.to_ormas_config()

    mc_test = mcscf.CASCI(mf, ncas, nelecas)
    mc_test.verbose = 0
    mc_test.fcisolver = ORMASFCISolver(config)
    e_test = mc_test.kernel()[0]

    # Variational bound: restricted energy >= CASCI energy
    assert e_test >= e_ref - 1e-10, (
        f"Variational violation: RAS={e_test} < CASCI={e_ref}"
    )

    # Determinant reduction
    n_ras = count_determinants(config)
    n_casci = casci_determinant_count(ncas, nelecas)
    assert n_ras < n_casci, (
        f"Expected fewer determinants: RAS={n_ras}, CASCI={n_casci}"
    )
    assert n_ras > 0, "RAS determinant count should be positive"


def test_h2o_ormas():
    """ORMAS-CI on H2O/cc-pVTZ with two subspaces.

    Uses cc-pVTZ basis for s/p/d/f function coverage. Partitions the
    active space into two subspaces with occupation restrictions. Verifies:
    1. Energy is above (or equal to) full CASCI (variational bound).
    2. Determinant count is reduced relative to CASCI.
    """
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVTZ",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 5, (3, 3)

    # PySCF full CASCI reference
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]

    # ORMAS: two subspaces with restricted occupation
    # 6 electrons in 5 orbitals, constrain to exclude extreme distributions
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1, 2], min_electrons=2, max_electrons=5),
            Subspace("B", [3, 4], min_electrons=1, max_electrons=3),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc_test = mcscf.CASCI(mf, ncas, nelecas)
    mc_test.verbose = 0
    mc_test.fcisolver = ORMASFCISolver(config)
    e_test = mc_test.kernel()[0]

    # Variational bound: restricted energy >= CASCI energy
    assert e_test >= e_ref - 1e-10, (
        f"Variational violation: ORMAS={e_test} < CASCI={e_ref}"
    )

    # Determinant reduction
    n_ormas = count_determinants(config)
    n_casci = casci_determinant_count(ncas, nelecas)
    assert n_ormas < n_casci, (
        f"Expected fewer determinants: ORMAS={n_ormas}, CASCI={n_casci}"
    )
    assert n_ormas > 0, "ORMAS determinant count should be positive"
