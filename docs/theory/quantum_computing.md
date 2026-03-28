# Quantum Computing Relevance of Partitioned Active Space Methods

## Why Quantum Computers for Chemistry

A quantum computer with N qubits can natively represent a 2^N dimensional
Hilbert space. In quantum chemistry, the electronic wavefunction of a
molecule with m active orbitals lives in a Hilbert space of dimension
C(m, n_alpha) * C(m, n_beta), which grows exponentially with m. Classical
computers approximate this (HF, MP2, CCSD(T), CASCI); quantum computers
can represent it directly by mapping orbitals to qubits.

The dominant algorithm for near-term quantum chemistry is VQE (Variational
Quantum Eigensolver): a hybrid classical-quantum method where the quantum
computer prepares a parameterized trial state (ansatz), measures the energy,
and a classical optimizer adjusts the parameters to minimize the energy.

## How RAS/ORMAS Helps Quantum Computing

Partitioned active space methods address two bottlenecks in quantum
chemistry on quantum hardware:

### 1. Circuit Depth Reduction

The VQE ansatz (typically UCCSD or similar) includes parameterized gate
sequences for each allowed excitation operator. In a full CAS ansatz,
every single and double excitation among all active orbitals is included.
For CAS(n,m), this means O(n^2 * m^2) excitation operators, each becoming
a multi-gate sequence in the circuit.

RAS/ORMAS constraints directly reduce the number of excitation operators.
An excitation that would violate an occupation constraint (e.g., creating
a third hole in RAS1) is simply not included in the ansatz. Fewer operators
means fewer gates means a shallower circuit. On noisy hardware, shallower
circuits accumulate less error.

This reduction is structural, not heuristic: the constraints are baked into
the circuit design before any quantum computation runs. The circuit is
correct by construction.

### 2. Qubit Reduction Through Subspace Factorization (ORMAS)

ORMAS's local per-subspace constraints open the possibility of solving each
subspace as a separate, smaller quantum calculation. If a molecule's active
space has 30 orbitals (requiring 30 qubits for full CAS), but ORMAS
partitions it into three subspaces of 10 orbitals each, each subspace
can be solved on a 10-qubit quantum computer.

The subspace calculations can be run sequentially on the same QPU (reusing
the same physical qubits for each subspace) or in parallel on multiple QPUs
when available. Inter-subspace coupling is recovered through classical
post-processing: mean-field embedding, perturbative corrections, or
selected inter-subspace excitations.

This approach trades exact inter-subspace correlation for a dramatic
reduction in qubit requirements, making large molecular systems tractable
on near-term hardware.

### Why Not Selected CI?

Selected CI (ASCI, CIPSI, etc.) also reduces the determinant count, but
its determinant space has no regular structure. The included determinants
depend on the wavefunction being computed, which isn't known beforehand.

This creates problems for quantum circuit construction:
- No systematic way to build an ansatz that explores exactly the selected
  determinants
- The determinant space can't be factored into independent subspace
  calculations
- The space size isn't known until the (classical) selection converges

RAS/ORMAS constraints are defined by orbital structure, which is fixed
before any calculation. They map directly to circuit construction rules
and enable hardware-aware resource planning.

Selected CI and RAS/ORMAS are complementary: use selected CI classically
to understand the wavefunction structure, then use that understanding to
define ORMAS subspaces for the quantum calculation.

## Connection to QMPI

For systems too large for a single QPU even after ORMAS partitioning,
Microsoft's QMPI (Quantum MPI) framework enables distributing a single
quantum calculation across multiple interconnected quantum processors.
ORMAS fragmentation operates at the algorithmic level (no quantum
communication needed between subspaces), while QMPI operates at the
hardware level (entanglement-based communication between QPUs). The
two approaches are complementary and address the qubit scaling problem
from different directions.

## Jordan-Wigner Mapping

The standard mapping from molecular orbitals to qubits is Jordan-Wigner:
each spin-orbital maps to one qubit, with |0> = unoccupied and |1> =
occupied. A Slater determinant maps to a computational basis state.

In this mapping, ORMAS constraints translate to restrictions on which
computational basis states the ansatz can explore. The constraints are
defined in terms of Hamming weight (popcount) of subsets of the qubit
register, which can be enforced through circuit structure.

ORMAS's determinant-basis representation (which this package produces)
maps directly to qubit computational basis states under Jordan-Wigner.
This is one reason why ORMAS (local constraints, determinant basis) is
preferred over GAS (cumulative constraints, sometimes CSF basis) for
quantum computing applications.
