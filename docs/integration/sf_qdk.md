# SF-ORMAS + QDK/Chemistry Integration

## Overview

SF-ORMAS enables excited-state and diradical calculations through
spin-flip CI within the ORMAS framework. By starting from a
high-spin ROHF reference and flipping one or more spins, it accesses
singlet and low-spin states that are difficult for standard methods.
This page describes integrating SF-ORMAS with QDK/Chemistry for
quantum resource estimation of excited-state problems.

## Why SF-ORMAS for Quantum Computing

Excited states and near-degenerate electronic structures are exactly
where quantum computers are expected to offer practical advantage
over classical methods. SF-ORMAS is relevant to quantum computing
workflows for several reasons:

- **Classically-validated benchmarks**: SF-ORMAS provides excited-state
  wavefunctions that serve as reference benchmarks for quantum
  algorithms (VQE, IQPE). Comparing quantum results against SF-ORMAS
  energies validates quantum hardware and algorithm performance.
- **Reduced determinant count**: ORMAS occupation restrictions
  eliminate determinants that violate subspace constraints. Fewer
  determinants may translate to simpler state preparation circuits
  on quantum hardware.
- **Targeting quantum-hard problems**: Diradicals, conical
  intersections, and spin-state gaps are challenging for classical
  CI but are natural targets for quantum algorithms. SF-ORMAS
  provides the classical baseline for these systems.
- **The pipeline**: classical ROHF reference → SF-ORMAS CI solve →
  quantum resource estimation via QDK/Chemistry.

## Workflow

The SF-ORMAS + QDK integration follows these steps:

1. **Build ROHF reference**: Start from a high-spin state (e.g.,
   triplet for single spin-flip to singlet). PySCF's `scf.ROHF`
   provides the orbitals and integrals.
2. **Configure SF-ORMAS**: Define an `SFORMASConfig` specifying
   the reference spin, target spin, number of spin flips, and
   ORMAS subspaces.
3. **Run PySCF CASCI**: Attach `SFORMASFCISolver` to a PySCF
   `CASCI` object and run the kernel. This yields energies and
   CI vectors in the target spin sector.
4. **Extract active-space integrals**: Retrieve the one- and
   two-electron integrals over the active space from PySCF.
5. **Construct qubit Hamiltonian**: Use the `ModelOrbitals` bridge
   to package integrals into a QDK Hamiltonian container, then
   map to a qubit Hamiltonian via Jordan-Wigner.
6. **Report resource metrics**: Extract qubit count, Pauli term
   count, and other resource metrics from the qubit Hamiltonian.

## Code Example

### SF-ORMAS Calculation

Complete working example for twisted ethylene CAS(2,2):

```python
from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count
from pyscf.ormas_ci.spinflip import count_sf_determinants

# 1. Build molecule with ROHF (triplet reference for SF)
mol = gto.M(
    atom="""
    C  0.000  0.000  0.000
    C  1.340  0.000  0.000
    H -0.500  0.930  0.000
    H -0.500 -0.930  0.000
    H  1.840  0.000  0.930
    H  1.840  0.000 -0.930
    """,
    basis="sto-3g", spin=2, verbose=0,
)
mf = scf.ROHF(mol)
mf.run()

# 2. Configure SF-ORMAS
sf_config = SFORMASConfig(
    ref_spin=2,        # triplet reference (2S = 2)
    target_spin=0,     # singlet target (2S = 0)
    n_spin_flips=1,
    n_active_orbitals=2,
    n_active_electrons=2,
    subspaces=[Subspace("pi", [0, 1], 0, 4)],
)

# 3. Run SF-ORMAS CASCI
n_a, n_b = sf_config.nelecas_target  # (1, 1) for singlet M_s=0
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
mc.fcisolver.nroots = 2  # singlet + triplet M_s=0
mc.kernel()

print(f"SF-ORMAS energies: {mc.e_tot}")
print(f"SF determinants: {count_sf_determinants(sf_config)}")
print(f"Full CAS determinants: {casci_determinant_count(2, (1, 1))}")
```

### QDK Qubit Hamiltonian

The QDK bridge constructs a qubit Hamiltonian from the active-space
integrals. This requires `qdk-chemistry`.

```python
# 4. QDK qubit Hamiltonian (requires qdk-chemistry)
import numpy as np
from pyscf import ao2mo
from qdk_chemistry._core.data import (
    CanonicalFourCenterHamiltonianContainer,
    ModelOrbitals,
)
from qdk_chemistry._core.data import Hamiltonian as QdkHamiltonian
from qdk_chemistry.algorithms import QdkQubitMapper

# Extract active-space integrals
mc_tmp = mcscf.CASCI(mf, 2, 2)
mc_tmp.verbose = 0
mc_tmp.kernel()
h1e, ecore = mc_tmp.get_h1eff()
eri = mc_tmp.get_h2eff()
if eri.ndim != 4:
    eri = ao2mo.restore(1, eri, 2)

# Build qubit Hamiltonian via ModelOrbitals bridge
model_orbs = ModelOrbitals(2, True)
container = CanonicalFourCenterHamiltonianContainer(
    h1e, eri.reshape(-1), model_orbs, ecore, np.zeros((2, 2))
)
hamiltonian = QdkHamiltonian(container)
qubit_ham = QdkQubitMapper().run(hamiltonian)

print(f"Qubits: {qubit_ham.num_qubits}")
print(f"Pauli terms: {len(qubit_ham.pauli_strings)}")
```

```{note}
The `ModelOrbitals` bridge is necessary because QDK's active-space
selectors do not support open-shell (ROHF/UHF) systems directly.
By packaging PySCF's active-space integrals into a
`CanonicalFourCenterHamiltonianContainer` with `ModelOrbitals`,
we bypass the active-space selection layer and feed integrals
directly into the qubit mapping stage.
See {doc}`qdk_chemistry` for more details on the bridge.
```

## Resource Comparison

Comparing SF-ORMAS restricted determinant spaces against full CAS
quantifies the potential quantum resource savings:

- **Determinant count**: `count_sf_determinants()` vs
  `casci_determinant_count()`. SF-ORMAS restricts which
  determinants are included based on spin-flip and subspace
  occupation constraints.
- **Qubit count**: For the same active space, the qubit count
  under Jordan-Wigner mapping is identical (2 × n_active_orbitals).
  ORMAS restrictions do not change the number of qubits.
- **State preparation**: Fewer determinants translate to simpler
  state preparation circuits with fewer CNOT gates. A CI vector
  over 4 determinants requires fewer unitary rotations than one
  over 16.
- **Wavefunction-aware filtering**: ORMAS-restricted wavefunctions
  may allow more aggressive Pauli term filtering, since terms
  coupling excluded determinants can be dropped without affecting
  accuracy.

```python
n_det_sf = count_sf_determinants(sf_config)
n_det_cas = casci_determinant_count(ncas, nelecas)
reduction = (1 - n_det_sf / n_det_cas) * 100
print(f"Determinant reduction: {reduction:.1f}%")
```

## Multi-root Resource Estimation

SF-ORMAS naturally produces multiple roots (e.g., the lowest
singlet and the M_s=0 component of the triplet). Each root shares
the same qubit Hamiltonian but has a different CI vector:

```python
mc.fcisolver.nroots = 3
mc.kernel()

# Each root has the same qubit Hamiltonian (same active-space integrals)
# but different CI vectors for state preparation
for i, e in enumerate(mc.e_tot):
    print(f"Root {i}: E = {e:.6f} Ha")
    # mc.ci[i] is the CI vector for root i
```

The qubit Hamiltonian is identical for all roots because it depends
only on the active-space integrals, which are fixed by the orbital
choice. Each root's CI vector provides a different state preparation
target for VQE or IQPE. This means a single qubit Hamiltonian
construction serves all excited-state resource estimates.

## FCIDUMP Generation

FCIDUMP files are a standard format for exchanging active-space
integrals between quantum chemistry codes. PySCF can generate
FCIDUMP files directly from an MCSCF object:

```python
from pyscf.tools import fcidump

# FCIDUMP works unchanged for SF-ORMAS
fcidump.from_mcscf(mc, "sf_ormas_ethylene.fcidump")
```

The FCIDUMP file contains active-space integrals in the target M_s
sector. For provenance, the following metadata can be added to the
header:

- `MS2`: Target M_s value
- `SF_REF_SPIN`: Reference 2S value
- `SF_NFLIPS`: Number of spin flips

See {doc}`qdk_chemistry` for background on the QDK/Chemistry bridge
and FCIDUMP usage.
