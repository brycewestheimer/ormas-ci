# Examples

Standalone, runnable scripts demonstrating ORMAS-CI usage with PySCF
and the QDK/Chemistry quantum computing pipeline.

## Prerequisites

Examples 01-04 require only PySCF (core dependency):
```bash
pip install ormas-ci
```

Examples 05-06 additionally require QDK/Chemistry:
```bash
pip install ormas-ci[qdk]
```

## PySCF Examples

| # | Script | System | What It Demonstrates |
|---|--------|--------|---------------------|
| 01 | `01_basic_ormas.py` | H2O/cc-pVDZ CAS(6,6) | Basic ORMAS-CI workflow, energy comparison, determinant reduction |
| 02 | `02_rasci.py` | Formaldehyde/cc-pVDZ CAS(4,3) | RASCI partitioning (RAS1/RAS2/RAS3), hole/particle control |
| 03 | `03_casscf_optimization.py` | H2O/STO-3G CAS(4,4) | ORMAS-CI as drop-in fcisolver in PySCF CASSCF |
| 04 | `04_convergence_study.py` | N2/cc-pVDZ CAS(6,6) | Systematic constraint relaxation, accuracy/cost tradeoff |

## QDK Pipeline Examples

| # | Script | System | What It Demonstrates |
|---|--------|--------|---------------------|
| 05 | `05_qdk_pipeline.py` | H2O/STO-3G | QDK SCF -> ORMAS-CI -> qubit Hamiltonian -> classical solve |
| 06 | `06_qdk_quantum_simulation.py` | Formaldehyde/cc-pVDZ CAS(4,3) | Real IQPE + VQE with method-specific state prep |

## Running

```bash
python examples/01_basic_ormas.py
python examples/02_rasci.py
python examples/03_casscf_optimization.py
python examples/04_convergence_study.py

# Requires qdk-chemistry
python examples/05_qdk_pipeline.py
python examples/06_qdk_quantum_simulation.py
```
