# QDK/Chemistry Integration Path

## Current QDK/Chemistry Architecture

QDK/Chemistry (github.com/microsoft/qdk-chemistry) uses a plugin
architecture for external quantum chemistry backends. Its capabilities
include:

- **Native implementations:** CASCI, ASCI (via MACIS), active space
  selection (AVAS, occupation-based, valence-based, MP2 natural orbitals)
- **PySCF plugin:** Exposes PySCF's SCF, CASCI, and CASSCF through
  QDK/Chemistry's uniform interface
- **Core:** C++/pybind11 with Python bindings

ORMAS-CI extends this ecosystem with RAS/ORMAS/GAS-style constrained CI,
plugging in through the PySCF backend.

## Integration Without Modifying QDK/Chemistry

Because QDK/Chemistry's PySCF plugin delegates to PySCF's CASCI, and
PySCF's CASCI supports swappable FCI solvers, our ORMAS-CI solver can
be used within a QDK/Chemistry workflow without any changes to
QDK/Chemistry itself.

The chain is:

```
QDK/Chemistry
    -> PySCF plugin
        -> PySCF CASCI
            -> ORMASFCISolver (our code)
                -> restricted CI
            <- energy, CI vector
        <- energy, CI vector
    <- energy, CI vector, RDMs
-> Hamiltonian construction, qubit mapping, resource estimation
```

The user sets up the molecule and active space through QDK/Chemistry's
tools, then swaps the FCI solver before running the calculation.

## Workflow Example

The `qdk_chemistry` 1.0.2 API is plugin-oriented.  SCF is provided by
`PyscfScfSolver`, and orbital conversion by `orbitals_to_scf`.  After
obtaining a PySCF SCF object we construct a standard PySCF CASCI and
swap the FCI solver.

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
scf_energy, wfn = scf_solver.run(structure, 0, 1, "6-31g")

# 2. Convert QDK orbitals to PySCF SCF object
orbitals = wfn.get_container().get_orbitals()
n_mo = orbitals.get_num_molecular_orbitals()
occ_a = [1] + [0] * (n_mo - 1)
occ_b = [1] + [0] * (n_mo - 1)
pyscf_scf = orbitals_to_scf(orbitals, occ_alpha=occ_a, occ_beta=occ_b,
                             scf_type=SCFType.RESTRICTED)

# 3. Set up CASCI with ORMASFCISolver
config = ORMASConfig(
    subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    n_active_orbitals=2,
    nelecas=(1, 1),
)
mc = mcscf.CASCI(pyscf_scf, 2, 2)
mc.fcisolver = ORMASFCISolver(config)
e_ormas = mc.kernel()[0]
```

This workflow is tested in ``tests/test_qdk_integration.py``.

## Future: Direct QDK/Chemistry Integration

Three potential paths for deeper integration:

### Option A: External PySCF FCI Solver Plugin

The current approach. No QDK/Chemistry changes needed. The solver is
installed alongside qdk-chemistry and the user swaps it in manually.
This is what the proof-of-concept demonstrates.

### Option B: QDK/Chemistry Plugin Registration

QDK/Chemistry's plugin architecture supports registering new algorithm
implementations. A dedicated plugin could register ORMAS-CI as an
available CI method, making it discoverable through QDK/Chemistry's
factory system without manual solver swapping.

This would require a small addition to QDK/Chemistry's plugin registry
and a wrapper that maps QDK/Chemistry's data model to our ORMASConfig.

### Option C: Native C++ Implementation

For production use, the performance-critical inner loops (Hamiltonian
construction, determinant enumeration) could be reimplemented in C++
within QDK/Chemistry's existing C++/pybind11 framework. The Python
proof-of-concept would serve as the reference implementation and test
oracle.

## What QDK/Chemistry Provides That We Use

- Molecular structure handling and basis set setup
- SCF computation (HF orbitals)
- Active space selection algorithms (AVAS, etc.)
- Integral transformation (AO to active-space MO)
- Qubit Hamiltonian construction from RDMs/integrals
- Resource estimation for quantum circuits
- Quantum circuit compilation and simulation (via QDK)

## What ORMAS-CI Adds

- Restricted determinant enumeration (ORMAS/RAS constraints)
- Structured CI expansion that maps to quantum circuit ansatze
- Subspace factorization for qubit reduction
- User control over which excitation classes to include
