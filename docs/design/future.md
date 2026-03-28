# Future Directions

## Near-Term Improvements

### Two-Particle Reduced Density Matrix (2-RDM)

**Status: Implemented.** Both 1-RDM and 2-RDM construction are available
via `make_rdm1()` and `make_rdm12()`. CASSCF orbital optimization through
PySCF's mc1step/mc2step is supported.

### GAS-CI (Cumulative Constraints)

Adding GAS support requires modifying the feasibility check in the
recursive distribution enumeration. Instead of checking per-subspace
bounds independently, the check would track cumulative electron counts
and compare against cumulative bounds. The rest of the pipeline
(string generation, Hamiltonian construction, solver, RDMs) is
identical to ORMAS.

### State-Averaged CI

Multi-root support (computing multiple eigenstates simultaneously)
is partially implemented (the solver supports nroots > 1). Full
state-averaged CASCI requires constructing state-averaged RDMs, which
is a straightforward extension of the current make_rdm1.

### Spin-Adapted CI (CSF Basis)

The current implementation uses Slater determinants as the basis.
A configuration state function (CSF) basis would exploit spin symmetry
to reduce the CI matrix size. This is a more invasive change requiring
new basis construction and modified Hamiltonian rules, but would
improve both performance and wavefunction interpretability.

## Medium-Term Extensions

### C++ Core with pybind11

Reimplement the performance-critical inner loops (Hamiltonian
construction, determinant enumeration) in C++. Expose through pybind11,
matching QDK/Chemistry's own architecture. The Python implementation
serves as the reference and test oracle.

### Direct QDK/Chemistry Plugin

Register ORMAS-CI as a QDK/Chemistry plugin through its factory system,
making it available as a first-class CI method option alongside CASCI
and ASCI. This requires a wrapper mapping QDK/Chemistry's data model
to ORMASConfig and registering the solver in the plugin registry.

### Automated Subspace Selection

Use information from a preliminary ASCI or CASCI calculation to
automatically partition orbitals into subspaces. For example:
- Cluster orbitals by natural orbital occupation (strongly correlated
  orbitals with occupations far from 0 or 2 go into the "core" subspace,
  weakly correlated ones into peripheral subspaces)
- Use orbital localization to identify spatially separated groups
- Use entanglement measures from a small CI calculation to group
  orbitals by correlation strength

### ORMAS + Selected CI Hybrid

Within the ORMAS-restricted determinant space, apply selected CI to
further prune unimportant determinants. This combines ORMAS's structural
benefits (ansatz compatibility, subspace factorization) with selected
CI's numerical efficiency (keeping only important determinants within
each subspace). The Hamiltonian construction would switch from explicit
matrix building to an iterative grow-and-prune algorithm operating
within the ORMAS-allowed space.

## Long-Term Research Directions

### Subspace Factorization for QPU Distribution

Formalize the inter-subspace coupling treatment: develop and benchmark
mean-field embedding, perturbative corrections, and dressed Hamiltonian
approaches for combining subspace calculations run on separate QPUs.
Quantify the accuracy/qubit tradeoff for realistic chemical systems.

### Ansatz Construction from ORMAS Constraints

Build VQE ansatze directly from the ORMAS subspace structure. Map
occupation constraints to circuit construction rules (which excitation
operators to include/exclude). Benchmark circuit depth reduction against
unrestricted UCCSD for transition metal complexes.

### Rust Implementation

A Rust implementation with Python bindings (via PyO3 or similar) would
offer performance comparable to C++ with better memory safety guarantees.
This could eventually become part of a broader Rust-based quantum
chemistry toolkit.
