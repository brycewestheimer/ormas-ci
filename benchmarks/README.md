# Benchmarks

Performance benchmarks comparing ORMAS-CI against PySCF's native FCI solver.

## Running Benchmarks

```bash
# Default systems (H2, H2O, N2/STO-3G, N2/cc-pVDZ)
python benchmarks/bench_vs_pyscf.py

# Specific systems
python benchmarks/bench_vs_pyscf.py --systems h2 n2_sto

# Include large CAS(10,10) benchmark (may take several minutes)
python benchmarks/bench_vs_pyscf.py --include-large

# More repeats for stable timing
python benchmarks/bench_vs_pyscf.py --repeats 5
```

## What Is Measured

### Timing
Each computational phase is timed separately:
- **Determinant enumeration**: Building the restricted determinant list
- **Hamiltonian construction**: Matrix element evaluation via Slater-Condon rules (dominant cost, O(n_det^2))
- **Diagonalization**: Eigenvalue solve (dense eigh, Davidson+einsum, or ARPACK Lanczos)

### Determinant Space Reduction
The key value proposition of ORMAS-CI for quantum computing: restricting the active space reduces the number of determinants and excitation operators.

- **n_det**: Number of Slater determinants in the CI expansion
- **n_excitation_ops**: Number of non-zero off-diagonal Hamiltonian elements (upper triangle). Each corresponds to an excitation operator in the quantum computing Hamiltonian representation.
- **Reduction factor**: Ratio of full CAS to restricted space counts

### Configurations Compared
For each molecular system, four configurations are benchmarked:
1. **PySCF FCI**: Native direct CI solver (reference energy and timing)
2. **ORMAS unrestricted**: Single subspace with no restrictions (= full CAS). Should match PySCF energy exactly. Measures the overhead of the explicit Hamiltonian approach.
3. **RASCI**: Classic 3-space restricted active space with hole/particle limits
4. **ORMAS restricted**: General multi-subspace partitioning with occupation bounds

## Interpreting Results

- **Energy error (dE)**: RASCI/ORMAS restricted energies are always >= full CAS energy (variational principle). The error in mHa quantifies the cost of truncation.
- **Timing overhead**: ORMAS unrestricted vs PySCF FCI shows the cost of explicit Hamiltonian construction vs PySCF's optimized direct CI. This overhead is expected and acceptable for small systems.
- **Reduction factor**: The practical benefit. A 10x reduction in determinants translates directly to a smaller quantum circuit representation.

## Output

Results are printed as formatted tables and saved as JSON to `benchmarks/results/latest.json`.

---

## Quantum Resource Estimation Benchmarks

Compares PySCF CASCI + QDK vs RASCI + QDK vs ORMAS + QDK on chemically
motivated model systems. All quantum metrics (Pauli terms, gate counts,
IQPE energies, VQE energies) are computed from real QDK operations.

Note: Open-shell systems (FeO) use QDK's `ModelOrbitals` bridge for
the active-space Hamiltonian, which packages PySCF active-space integrals
into a QDK Hamiltonian container. The integrals are from the target
system; only the orbital metadata object is a model container.

Requires QDK/Chemistry: `pip install pyscf-ormas-ci[qdk]`

### Model Systems

**RAS systems** (clear occupied/frontier/virtual separation):
- Ethylene (C2H4) / cc-pVDZ: CAS(2,2) pi/pi*
- Ethylene (C2H4) / cc-pVTZ: CAS(4,4) pi+sigma
- Ozone (O3) / cc-pVDZ: CAS(6,5) diradical
- Formaldehyde (CH2O) / cc-pVDZ: CAS(4,3) n/pi/pi*

**ORMAS systems** (each demonstrates a different partitioning strategy):
- N2 / cc-pVDZ: CAS(6,6) sigma/pi symmetry partitioning (2.3x reduction)
- FeO / cc-pVDZ: CAS(8,8) quintet, Fe 3d / O 2p metal/ligand split (3.5x reduction)
- Butadiene (C4H6) / cc-pVDZ: CAS(4,4) spatial locality (localized C=C pi pairs)

### Running

```bash
python benchmarks/bench_qdk_quantum.py                          # All systems, classical metrics
python benchmarks/bench_qdk_quantum.py --systems ethylene_dz    # Specific system
python benchmarks/bench_qdk_quantum.py --iqpe                   # + real IQPE simulation
python benchmarks/bench_qdk_quantum.py --vqe                    # + VQE energy estimation
python benchmarks/bench_qdk_quantum.py --resource-estimate       # + fault-tolerant estimates
python benchmarks/bench_qdk_quantum.py --wfn-filter              # + wfn-aware filtering
python benchmarks/bench_qdk_quantum.py --all                     # Everything
python benchmarks/bench_qdk_quantum.py --all --quick             # Fast mode (6-bit, 1K shots)
```

### What Is Measured

- **Pauli terms**: Real count from `QdkQubitMapper.run()` Jordan-Wigner mapping
- **Hamiltonian 1-norm (lambda)**: `qubit_ham.schatten_norm` — sum of |Pauli
  coefficients|. Determines QPE iteration count: `n_iter ~ lambda / epsilon`
- **Gate counts**: Real CNOTs and rotations from QASM parsing of `PauliSequenceMapper` circuits
- **Qubit solve**: Classical diagonalization via QDK's Davidson eigensolver
- **IQPE energy**: Real iterative QPE simulation via `qdk_full_state_simulator`
- **VQE energy**: Shot-based expectation value via `qdk_base_simulator`
- **Determinant reduction**: RASCI/ORMAS vs full CASCI determinant count
- **Fault-tolerant resource estimates** (`--resource-estimate`): Physical qubit
  count, runtime, code distance, T-state count from Azure Quantum Resource
  Estimator via `qsharp.estimator.LogicalCounts.estimate()`. Uses Majorana
  qubit model (ns gate time, 1e-6 error rate) with Floquet QEC code
- **Wavefunction-aware filtering** (`--wfn-filter`): Pauli term reduction from
  evaluating expectation values against the method-specific CI wavefunction.
  Terms with 0 expectation removed; +/-1 terms treated classically. ORMAS
  wavefunctions enable more aggressive filtering

### Output

Results are printed as formatted tables and saved as JSON to
`benchmarks/results/quantum_latest.json`.
