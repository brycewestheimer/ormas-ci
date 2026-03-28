# QDK/Chemistry Pipeline

## Overview

QDK/Chemistry (github.com/microsoft/qdk-chemistry) is a modular toolkit
for quantum chemistry workflows targeting quantum computers. It provides
molecular structure handling, SCF, active space selection, CI solvers,
qubit Hamiltonian construction, and resource estimation.

Our ORMAS-CI solver integrates through QDK/Chemistry's existing PySCF
plugin, requiring no modifications to QDK/Chemistry itself.

## Prerequisites

```bash
pip install qdk-chemistry pyscf-ormas-ci pyscf
```

Note: QDK/Chemistry is new (January 2026) and the installation process
may evolve. Check the QDK/Chemistry README for current instructions.

## Integration Architecture

```
QDK/Chemistry
    |
    +-> Molecular structure + basis
    +-> SCF (via PySCF plugin or native)
    +-> Active space selection (AVAS, occupation-based, etc.)
    +-> Integral transformation
    |
    +-> PySCF CASCI <--- ORMASFCISolver (our code)
    |
    +-> RDMs, CI vector
    +-> Qubit Hamiltonian (Jordan-Wigner mapping)
    +-> Resource estimation (qubit count, circuit depth)
    +-> Circuit compilation (via QDK)
```

The key point: QDK/Chemistry handles everything before and after the
CI solve. We handle only the CI solve itself, plugging in through PySCF.

## Workflow

The `qdk_chemistry` 1.0.2 API is plugin-oriented.  The key classes are:

- `qdk_chemistry.data.Structure` -- molecular geometry
- `qdk_chemistry.plugins.pyscf.scf_solver.PyscfScfSolver` -- SCF via PySCF
- `qdk_chemistry.plugins.pyscf.conversion.orbitals_to_scf` -- convert QDK
  orbitals to a PySCF SCF object
- `qdk_chemistry.plugins.pyscf.active_space_avas.PyscfAVAS` -- AVAS active
  space selection

The general pattern is:

1. Create a `Structure` and run SCF via `PyscfScfSolver`
2. Optionally use `PyscfAVAS` for active space selection
3. Convert QDK orbitals to a PySCF SCF object with `orbitals_to_scf`
4. Construct a PySCF CASCI and swap the FCI solver to `ORMASFCISolver`
5. Run the calculation
6. Feed results back to QDK/Chemistry for quantum pipeline stages

```python
from qdk_chemistry.data import Structure
from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf, SCFType
from pyscf import mcscf
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

# 1. Structure and SCF via QDK/Chemistry
xyz_str = "2\nH2\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74"
structure = Structure.from_xyz(xyz_str)
scf_solver = PyscfScfSolver()
scf_energy, wfn = scf_solver.run(structure, 0, 1, "sto-3g")

# 2. Convert QDK orbitals to PySCF
orbitals = wfn.get_container().get_orbitals()
n_mo = orbitals.get_num_molecular_orbitals()
occ_a = [1] + [0] * (n_mo - 1)
occ_b = [1] + [0] * (n_mo - 1)
pyscf_scf = orbitals_to_scf(orbitals, occ_alpha=occ_a, occ_beta=occ_b,
                             scf_type=SCFType.RESTRICTED)

# 3. CASCI with ORMASFCISolver
config = ORMASConfig(
    subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    n_active_orbitals=2,
    nelecas=(1, 1),
)
mc = mcscf.CASCI(pyscf_scf, 2, 2)
mc.fcisolver = ORMASFCISolver(config)
e_ormas = mc.kernel()[0]
```

See the QDK/Chemistry integration notebook (notebooks/04) for a
more detailed working example with resource estimation.

## Mapping Active Space Selection to ORMAS Subspaces

QDK/Chemistry's active space selection tools (AVAS, occupation-based,
etc.) choose which orbitals go into the active space. ORMAS further
partitions those active orbitals into subspaces.

A natural workflow:
1. Use AVAS to select active orbitals based on atomic character
   (e.g., Fe 3d and ligand pi/sigma)
2. AVAS reports which active orbitals have which atomic character
3. Use that information to define ORMAS subspaces (metal orbitals
   in one subspace, ligand orbitals in another)

This is currently manual. Automated subspace assignment based on
AVAS orbital character is a future direction.

## Resource Estimation

The main quantum computing payoff is in resource estimation. After
running ORMAS-CI:

- The number of qubits needed per subspace is smaller than for full CAS
- The circuit depth is reduced because fewer excitation operators are
  included
- If subspace factorization is used, the maximum qubit count is the
  size of the largest subspace, not the full active space

Comparing resource estimates between full CASCI and ORMAS-restricted
calculations quantifies the hardware savings.

## Known Limitations

- QDK/Chemistry's API is new and may change. The integration notebook
  should be treated as a snapshot, not a stable interface.
- The PySCF plugin in QDK/Chemistry may not expose all of PySCF's CASCI
  internals directly. Accessing the underlying PySCF objects may require
  reaching into implementation details.
- Resource estimation assumes a specific ansatz mapping. The connection
  between ORMAS determinant space and VQE ansatz structure is conceptually
  clear but not yet automated in QDK/Chemistry.

## Open-Shell Support

QDK/Chemistry v1.0.2 supports open-shell SCF (ROHF/UHF) through
its PySCF plugin. However, the active space selectors (`qdk_valence`,
`PyscfAVAS`) currently only work with restricted orbitals.

For open-shell systems (e.g., FeO quintet), the benchmark uses a
`ModelOrbitals` bridge: active-space integrals are extracted from
PySCF's CASCI, then packaged into a QDK
`CanonicalFourCenterHamiltonianContainer` with `ModelOrbitals(ncas, True)`.
This produces a valid qubit Hamiltonian from the target system's
integrals without requiring QDK's active space selector.

| QDK Layer | Closed-Shell | Open-Shell |
|-----------|-------------|------------|
| PySCF SCF (RHF/ROHF/UHF) | Supported | Supported |
| Active space selectors | Supported | Not yet supported |
| HamiltonianConstructor | Supported | Supported (full MOs) |
| Qubit mapping (JW) | Supported | Supported |
| Active-space bridge | `qdk_valence` | `ModelOrbitals` + PySCF integrals |
