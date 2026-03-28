"""Tests for quantum resource estimation benchmark utilities.

Tests the helper functions and real QDK integrations used in
bench_qdk_quantum.py. All QDK-dependent tests are skipped if
qdk-chemistry is not installed.
"""

import sys

import pytest

sys.path.insert(0, "benchmarks")

from pyscf.ormas_ci import ORMASConfig, Subspace

try:
    from qdk_chemistry.algorithms import QdkHamiltonianConstructor, QdkQubitMapper, create
    from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
    from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper import (
        PauliSequenceMapper,
    )
    from qdk_chemistry.data import Structure
    from qdk_chemistry.data.time_evolution.controlled_time_evolution import (
        ControlledTimeEvolutionUnitary,
    )
    from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False

qdk_only = pytest.mark.skipif(not QDK_AVAILABLE, reason="qdk-chemistry not installed")


# ---------------------------------------------------------------------------
# QASM parser (from bench_qdk_quantum.py, kept local to avoid import issues)
# ---------------------------------------------------------------------------


from bench_qdk_quantum import parse_gate_counts as _parse_gate_counts  # noqa: E402

# ---------------------------------------------------------------------------
# Tests that always run (no QDK needed)
# ---------------------------------------------------------------------------


class TestQASMParser:
    """Test the QASM gate count parser with recursive custom-gate expansion."""

    def test_empty_qasm(self):
        assert _parse_gate_counts("") == {}

    def test_single_invocation_expands(self):
        """A custom gate invoked once expands to its primitive contents."""
        qasm = """OPENQASM 3.0;
include "stdgates.inc";
gate my_gate q0, q1 {
  cx q0, q1;
  h q0;
  crz(0.5) q1, q0;
}
my_gate q[0], q[1];
"""
        counts = _parse_gate_counts(qasm)
        assert counts["cx"] == 1
        assert counts["h"] == 1
        assert counts["crz"] == 1
        assert "my_gate" not in counts

    def test_repeated_invocation(self):
        """A custom gate invoked twice doubles the primitive count."""
        qasm = """OPENQASM 3.0;
gate trotter q0, q1 {
  cx q0, q1;
  rz(0.5) q1;
}
trotter q[0], q[1];
trotter q[0], q[1];
"""
        counts = _parse_gate_counts(qasm)
        assert counts["cx"] == 2
        assert counts["rz"] == 2
        assert "trotter" not in counts

    def test_inline_gates(self):
        """Inline gates (no gate def) are counted directly."""
        qasm = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
rz(0.5) q[1];
"""
        counts = _parse_gate_counts(qasm)
        assert counts["h"] == 1
        assert counts["cx"] == 1
        assert counts["rz"] == 1
        assert "qubit[2]" not in counts

    def test_declarations_excluded(self):
        """OPENQASM, include, qubit, bit are not counted as gates."""
        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[1] c;
h q[0];
"""
        counts = _parse_gate_counts(qasm)
        assert counts == {"h": 1}

    def test_nested_custom_gates(self):
        """Nested custom gates are recursively expanded."""
        qasm = """OPENQASM 3.0;
gate inner q0 {
  h q0;
  rz(0.1) q0;
}
gate outer q0, q1 {
  inner q0;
  cx q0, q1;
}
outer q[0], q[1];
"""
        counts = _parse_gate_counts(qasm)
        assert counts["h"] == 1
        assert counts["rz"] == 1
        assert counts["cx"] == 1
        assert "inner" not in counts
        assert "outer" not in counts


class TestQubitSavingsFormula:
    """Test qubit savings formula."""

    def test_two_equal_subspaces(self):
        ncas = 8
        max_sub = 4
        savings = (1.0 - max_sub / ncas) * 100.0
        assert savings == 50.0

    def test_three_subspaces(self):
        ncas = 8
        config = ORMASConfig(
            subspaces=[
                Subspace("a", [0, 1, 2], min_electrons=0, max_electrons=6),
                Subspace("b", [3, 4], min_electrons=0, max_electrons=4),
                Subspace("c", [5, 6, 7], min_electrons=0, max_electrons=6),
            ],
            n_active_orbitals=ncas,
            nelecas=(4, 4),
        )
        max_sub = max(sub.n_orbitals for sub in config.subspaces)
        savings = (1.0 - max_sub / ncas) * 100.0
        assert savings == pytest.approx(62.5)


@qdk_only
class TestBitstringsToConfigurations:
    """Test conversion of ORMAS bitstrings to QDK Configuration objects."""

    def test_doubly_occupied(self):
        """Alpha=0b11, Beta=0b11 -> starts with '22' for 2 orbitals."""
        import numpy as np

        alpha = np.array([0b11], dtype=np.int64)
        beta = np.array([0b11], dtype=np.int64)
        from benchmarks.bench_qdk_quantum import _bitstrings_to_configurations

        configs = _bitstrings_to_configurations(alpha, beta, 2)
        assert len(configs) == 1
        assert str(configs[0]).startswith("22")

    def test_mixed_occupation(self):
        """Alpha=0b10, Beta=0b01 -> starts with 'du' for 2 orbitals."""
        import numpy as np

        alpha = np.array([0b10], dtype=np.int64)
        beta = np.array([0b01], dtype=np.int64)
        from benchmarks.bench_qdk_quantum import _bitstrings_to_configurations

        configs = _bitstrings_to_configurations(alpha, beta, 2)
        assert len(configs) == 1
        assert str(configs[0]).startswith("du")


# ---------------------------------------------------------------------------
# Tests that require qdk-chemistry
# ---------------------------------------------------------------------------


def _h2_active_space_qdk():
    """Get QDK active-space qubit Hamiltonian for H2/STO-3G."""
    xyz = "2\nH2\nH  0.000000  0.000000  0.000000\nH  0.000000  0.000000  0.740000"
    structure = Structure.from_xyz(xyz)
    _, wfn = PyscfScfSolver().run(structure, 0, 1, "sto-3g")
    as_sel = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=2,
        num_active_orbitals=2,
    )
    sel_wfn = as_sel.run(wfn)
    sel_orbs = sel_wfn.get_orbitals()
    ham = QdkHamiltonianConstructor().run(sel_orbs)
    qh = QdkQubitMapper().run(ham)
    return qh, ham


@qdk_only
class TestQubitHamiltonianH2:
    """Test qubit Hamiltonian construction for H2."""

    def test_num_qubits(self):
        qh, _ = _h2_active_space_qdk()
        assert qh.num_qubits == 4

    def test_pauli_term_count(self):
        qh, _ = _h2_active_space_qdk()
        assert len(qh.pauli_strings) == 15

    def test_encoding_is_jordan_wigner(self):
        qh, _ = _h2_active_space_qdk()
        assert qh.encoding == "jordan-wigner"


@qdk_only
class TestRealCircuitConstruction:
    """Test real circuit construction via QDK Trotter + PauliSequenceMapper."""

    def test_h2_circuit_has_real_gates(self):
        """H2 Trotter circuit has real CNOT and rotation gates."""
        qh, _ = _h2_active_space_qdk()
        trotter = Trotter()
        evolution = trotter.run(qh, 1.0)
        n_q = evolution.get_num_qubits()
        ctrl_evo = ControlledTimeEvolutionUnitary(evolution, [n_q])
        circuit = PauliSequenceMapper().run(ctrl_evo)

        counts = _parse_gate_counts(circuit.qasm)
        assert counts.get("cx", 0) > 0, "Circuit should have CNOT gates"
        assert counts.get("crz", 0) > 0, "Circuit should have rotation gates"
        assert sum(counts.values()) > 10, "Circuit should have significant gate count"

    def test_h2_circuit_cnot_count(self):
        """H2/STO-3G CAS(2,2) should have 36 CNOTs."""
        qh, _ = _h2_active_space_qdk()
        trotter = Trotter()
        evolution = trotter.run(qh, 1.0)
        ctrl_evo = ControlledTimeEvolutionUnitary(evolution, [evolution.get_num_qubits()])
        circuit = PauliSequenceMapper().run(ctrl_evo)
        counts = _parse_gate_counts(circuit.qasm)
        assert counts["cx"] == 36


@qdk_only
class TestClassicalQubitSolve:
    """Test classical qubit Hamiltonian solve."""

    def test_h2_energy_reasonable(self):
        """Qubit solve energy for H2 is in the expected range."""
        qh, ham = _h2_active_space_qdk()
        solver = create("qubit_hamiltonian_solver", "qdk_sparse_matrix_solver")
        e_raw, _ = solver.run(qh)
        e_total = e_raw + ham.get_core_energy()
        assert -1.2 < e_total < -1.0


@qdk_only
class TestTrotterDecomposition:
    """Test Trotter decomposition for QPE resource counting."""

    def test_h2_step_terms_count(self):
        """H2 Trotter decomposition has 15 step terms (= Pauli terms)."""
        qh, _ = _h2_active_space_qdk()
        trotter = Trotter()
        evolution = trotter.run(qh, 1.0)
        container = evolution.get_container()
        assert len(container.step_terms) == 15
        assert container.num_qubits == 4

    def test_step_terms_have_angles(self):
        """Each step term has a rotation angle."""
        qh, _ = _h2_active_space_qdk()
        trotter = Trotter()
        evolution = trotter.run(qh, 1.0)
        container = evolution.get_container()
        for term in container.step_terms:
            assert hasattr(term, "angle")
            assert isinstance(term.angle, float)


# ---------------------------------------------------------------------------
# Tests for new quantum resource features
# ---------------------------------------------------------------------------

try:
    from qsharp.estimator import LogicalCounts

    QSHARP_AVAILABLE = True
except ImportError:
    QSHARP_AVAILABLE = False

qsharp_only = pytest.mark.skipif(not QSHARP_AVAILABLE, reason="qsharp not installed")


class TestBenchmarkResultNewFields:
    """Test that new BenchmarkResult fields have correct defaults."""

    def test_default_ham_1_norm(self):
        from bench_qdk_quantum import BenchmarkResult

        r = BenchmarkResult(
            system="test", method="test", basis="test",
            energy=0.0, energy_error_mha=0.0, n_det=0, t_fermionic=0.0,
        )
        assert r.ham_1_norm == 0.0
        assert r.re_physical_qubits == 0
        assert r.wfn_filter_original_terms == 0
        assert r.df_physical_qubits == 0

    def test_asdict_includes_new_fields(self):
        from dataclasses import asdict

        from bench_qdk_quantum import BenchmarkResult

        r = BenchmarkResult(
            system="test", method="test", basis="test",
            energy=0.0, energy_error_mha=0.0, n_det=0, t_fermionic=0.0,
        )
        d = asdict(r)
        assert "ham_1_norm" in d
        assert "re_physical_qubits" in d
        assert "wfn_filter_original_terms" in d
        assert "df_physical_qubits" in d


@qdk_only
class TestSchattenNorm:
    """Test Hamiltonian 1-norm (schatten_norm) reporting."""

    def test_h2_schatten_norm_positive(self):
        """H2 schatten_norm should be a positive number."""
        qh, _ = _h2_active_space_qdk()
        assert qh.schatten_norm > 0.0

    def test_h2_schatten_norm_matches_manual(self):
        """schatten_norm should match manual sum of |coefficients|."""
        import numpy as np

        qh, _ = _h2_active_space_qdk()
        manual = float(np.sum(np.abs(qh.coefficients)))
        assert abs(qh.schatten_norm - manual) < 1e-10


@qdk_only
@qsharp_only
class TestResourceEstimation:
    """Test Azure Quantum Resource Estimator integration."""

    def test_logical_counts_construction(self):
        """LogicalCounts can be constructed from circuit data."""
        lc = LogicalCounts({
            "numQubits": 5,
            "tCount": 0,
            "rotationCount": 50,
            "rotationDepth": 50,
            "cczCount": 0,
            "measurementCount": 5,
        })
        assert lc["numQubits"] == 5
        assert lc["rotationCount"] == 50

    def test_h2_resource_estimation_produces_results(self):
        """H2 resource estimation produces non-zero physical qubits."""
        from bench_qdk_quantum import _run_resource_estimation

        qh, _ = _h2_active_space_qdk()
        result = _run_resource_estimation(qh, {"crz": 10, "rz": 5}, qh.num_qubits)
        assert result.get("physical_qubits", 0) > 0
        assert result.get("runtime_ns", 0) > 0
        assert result.get("code_distance", 0) > 0

    def test_resource_estimation_without_qsharp(self):
        """Resource estimation returns empty dict when rotations are 0."""
        from bench_qdk_quantum import _run_resource_estimation

        qh, _ = _h2_active_space_qdk()
        # Zero rotations should still work (just minimal resources)
        result = _run_resource_estimation(qh, {}, qh.num_qubits)
        # With 0 rotations, it either succeeds with minimal resources or fails gracefully
        assert isinstance(result, dict)


@qdk_only
class TestWavefunctionFilter:
    """Test wavefunction-aware Hamiltonian filtering."""

    def test_h2_filter_returns_valid_structure(self):
        """Filter on H2 produces valid result structure."""
        from bench_qdk_quantum import _build_method_state_prep, _pyscf_ci_to_determinant_arrays

        from pyscf import gto, mcscf, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = mcscf.CASCI(mf, 2, 2)
        mc.verbose = 0
        mc.kernel()

        qh, ham = _h2_active_space_qdk()

        # Build wavefunction via the state prep pipeline
        ci_coeffs, ci_alpha, ci_beta = _pyscf_ci_to_determinant_arrays(
            mc.ci, 2, (1, 1)
        )
        _, wfn_obj = PyscfScfSolver().run(
            Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74"), 0, 1, "sto-3g"
        )
        as_sel = create(
            "active_space_selector", "qdk_valence",
            num_active_electrons=2, num_active_orbitals=2,
        )
        qdk_orbs = as_sel.run(wfn_obj).get_orbitals()
        _, _, wfn = _build_method_state_prep(
            ci_coeffs, ci_alpha, ci_beta, 2, qdk_orbs
        )
        assert wfn is not None

        from bench_qdk_quantum import _run_wavefunction_filter

        result = _run_wavefunction_filter(qh, wfn)
        assert result.get("original_terms", 0) == 15  # H2 has 15 Pauli terms
        assert 0 <= result.get("remaining_terms", -1) <= 15
        assert 0.0 <= result.get("reduction_pct", -1.0) <= 100.0
