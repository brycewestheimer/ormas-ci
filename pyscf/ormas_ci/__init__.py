"""
ORMAS-CI: Occupation-Restricted Multiple Active Space Configuration Interaction

A PySCF-compatible fcisolver implementing RASCI and ORMAS-CI with arbitrary
subspace partitioning and occupation constraints. Designed for quantum computing
workflows where restricted CI expansions reduce qubit requirements and circuit depth.

Usage:
    from pyscf import gto, scf, mcscf
    from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g')
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas=6, nelecas=(4, 4))

    config = ORMASConfig(
        subspaces=[
            Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
            Subspace("pi", [3, 4, 5], min_electrons=2, max_electrons=6),
        ],
        n_active_orbitals=6,
        nelecas=(4, 4),
    )

    mc.fcisolver = ORMASFCISolver(config)
    mc.kernel()

No modifications to PySCF or QDK/Chemistry are required. The subspace configuration
is carried entirely by the ORMASFCISolver object. PySCF's CASCI handles orbital
setup and integral transformation; our solver handles the restricted CI internally.
"""

from pyscf.ormas_ci.determinants import (
    build_determinant_list,
    casci_determinant_count,
    count_determinants,
)
from pyscf.ormas_ci.fcisolver import ORMASFCISolver, SFORMASFCISolver
from pyscf.ormas_ci.spinflip import (
    count_sf_determinants,
    generate_sf_determinants,
    validate_reference_consistency,
)
from pyscf.ormas_ci.subspaces import ORMASConfig, RASConfig, SFORMASConfig, SFRASConfig, Subspace

__version__ = "0.3.0"

__all__ = [
    "Subspace",
    "ORMASConfig",
    "RASConfig",
    "SFORMASConfig",
    "SFRASConfig",
    "ORMASFCISolver",
    "SFORMASFCISolver",
    "build_determinant_list",
    "casci_determinant_count",
    "count_determinants",
    "generate_sf_determinants",
    "count_sf_determinants",
    "validate_reference_consistency",
]
