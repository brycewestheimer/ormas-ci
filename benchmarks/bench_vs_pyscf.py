#!/usr/bin/env python3
"""Benchmark ORMAS-CI against PySCF's native FCI solver.

Compares performance and determinant space reduction across four configurations
for each molecular system:

1. PySCF native FCI (reference)
2. ORMAS unrestricted (single subspace = full CAS, quantifies overhead)
3. RASCI restricted (classic 3-space RAS partitioning)
4. ORMAS restricted (multi-subspace with occupation bounds)

Reports timing, determinant counts, excitation operator counts, and energy
errors. The determinant/operator reduction metrics are directly relevant to
quantum computing resource estimation: fewer determinants and excitation
operators translate to smaller quantum circuit representations.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from pyscf import gto, mcscf, scf

from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace, build_determinant_list
from pyscf.ormas_ci.determinants import casci_determinant_count
from pyscf.ormas_ci.hamiltonian import build_ci_hamiltonian
from pyscf.ormas_ci.solver import solve_ci
from pyscf.ormas_ci.subspaces import RASConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    method: str
    n_det: int
    n_excitation_ops: int
    energy: float
    t_enum: float
    t_hamiltonian: float
    t_diag: float
    t_total: float


@dataclass
class QubitHamiltonianMetrics:
    """Metrics from Jordan-Wigner qubit Hamiltonian construction."""

    n_qubits: int
    n_pauli_terms: int
    qubit_hamiltonian_sparsity: float
    t_hamiltonian_construct: float
    t_qubit_mapping: float


@dataclass
class BenchmarkCase:
    """Definition of a molecular benchmark system."""

    name: str
    atom: str
    basis: str
    ncas: int
    nelecas: tuple[int, int]
    ras_config: RASConfig | None = None
    ormas_config: ORMASConfig | None = None
    large: bool = False


def _count_excitation_ops(h_ci) -> int:
    """Count non-zero off-diagonal CI Hamiltonian elements (upper triangle).

    This is a classical proxy metric: the number of determinant pairs connected
    by non-zero matrix elements. It correlates with but is NOT the same as the
    Pauli term count in a Jordan-Wigner qubit Hamiltonian. For real Pauli term
    counts, see ``benchmarks/bench_qdk_quantum.py`` which uses QDK's
    ``QdkQubitMapper`` to construct actual qubit Hamiltonians.
    """
    if sp.issparse(h_ci):
        h_dense = h_ci.toarray()
    else:
        h_dense = np.asarray(h_ci)
    upper = np.triu(h_dense, k=1)
    return int(np.count_nonzero(np.abs(upper) > 1e-14))


def _build_qubit_hamiltonian(orbitals) -> tuple | None:
    """Build a qubit Hamiltonian from QDK Orbitals via Jordan-Wigner mapping.

    Returns (qubit_hamiltonian, QubitHamiltonianMetrics) or None if QDK is
    not available.
    """
    try:
        from qdk_chemistry.algorithms import QdkHamiltonianConstructor, QdkQubitMapper
    except ImportError:
        return None

    t0 = time.perf_counter()
    hamiltonian = QdkHamiltonianConstructor().run(orbitals)
    t1 = time.perf_counter()
    qubit_ham = QdkQubitMapper().run(hamiltonian)
    t2 = time.perf_counter()

    n_qubits = qubit_ham.num_qubits
    n_terms = len(qubit_ham.pauli_strings)
    sparsity = 1.0 - n_terms / (4**n_qubits) if n_qubits > 0 else 0.0

    metrics = QubitHamiltonianMetrics(
        n_qubits=n_qubits,
        n_pauli_terms=n_terms,
        qubit_hamiltonian_sparsity=sparsity,
        t_hamiltonian_construct=t1 - t0,
        t_qubit_mapping=t2 - t1,
    )
    return qubit_ham, metrics


def _make_unrestricted_config(ncas: int, nelecas: tuple[int, int]) -> ORMASConfig:
    """Create an unrestricted ORMAS config (single subspace = full CAS)."""
    return ORMASConfig(
        subspaces=[
            Subspace("all", list(range(ncas)), min_electrons=0, max_electrons=2 * ncas),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )


def _run_pyscf_reference(mf, ncas, nelecas, n_repeats):
    """Run PySCF native FCI and return timing + energy."""
    times = []
    energy = None
    for _ in range(n_repeats):
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        t0 = time.perf_counter()
        e = mc.kernel()[0]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        energy = e

    # Count determinants and excitation ops for comparison
    n_det = casci_determinant_count(ncas, nelecas)

    return BenchmarkResult(
        name="",
        method="PySCF FCI",
        n_det=n_det,
        n_excitation_ops=-1,  # Not directly comparable — PySCF uses direct CI
        energy=energy,
        t_enum=0.0,
        t_hamiltonian=0.0,
        t_diag=0.0,
        t_total=statistics.median(times),
    )


def _run_ormas(mf, ncas, nelecas, config, method_name, n_repeats):
    """Run ORMAS-CI with detailed phase timing."""
    # Pre-compute integrals (outside timing)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    # Run once to get integrals from PySCF
    h1e, ecore = mc.get_h1eff()
    eri = mc.get_h2eff()
    from pyscf import ao2mo
    if eri.ndim != 4:
        eri = ao2mo.restore(1, eri, ncas)

    times_enum = []
    times_ham = []
    times_diag = []
    times_total = []
    energy = None
    n_det = 0
    n_excitation_ops = 0

    for _ in range(n_repeats):
        t_start = time.perf_counter()

        # Phase 1: Determinant enumeration
        t0 = time.perf_counter()
        alpha_strings, beta_strings = build_determinant_list(config)
        t1 = time.perf_counter()
        times_enum.append(t1 - t0)

        # Phase 2: Hamiltonian construction
        t0 = time.perf_counter()
        h_ci = build_ci_hamiltonian(alpha_strings, beta_strings, h1e, eri)
        t1 = time.perf_counter()
        times_ham.append(t1 - t0)

        # Phase 3: Diagonalization
        t0 = time.perf_counter()
        energies, ci_vectors = solve_ci(h_ci, n_roots=1)
        t1 = time.perf_counter()
        times_diag.append(t1 - t0)

        t_end = time.perf_counter()
        times_total.append(t_end - t_start)

        energy = float(energies[0]) + ecore
        n_det = len(alpha_strings)
        n_excitation_ops = _count_excitation_ops(h_ci)

    return BenchmarkResult(
        name="",
        method=method_name,
        n_det=n_det,
        n_excitation_ops=n_excitation_ops,
        energy=energy,
        t_enum=statistics.median(times_enum),
        t_hamiltonian=statistics.median(times_ham),
        t_diag=statistics.median(times_diag),
        t_total=statistics.median(times_total),
    )


# ---------------------------------------------------------------------------
# Benchmark case definitions
# ---------------------------------------------------------------------------

BENCHMARK_CASES = {
    "h2": BenchmarkCase(
        name="H2/STO-3G",
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        ncas=2,
        nelecas=(1, 1),
        ras_config=None,  # Too small for meaningful RAS
        ormas_config=None,  # Too small for meaningful ORMAS
    ),
    "h2o": BenchmarkCase(
        name="H2O/cc-pVTZ",
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pVTZ",
        ncas=5,
        nelecas=(3, 3),
        ras_config=RASConfig(
            ras1_orbitals=[0, 1],
            ras2_orbitals=[2],
            ras3_orbitals=[3, 4],
            max_holes_ras1=1,
            max_particles_ras3=1,
            nelecas=(3, 3),
        ),
        # Moderate ORMAS constraint: keep core-like orbitals partially occupied
        ormas_config=ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
                Subspace("pi", [3, 4], min_electrons=0, max_electrons=3),
            ],
            n_active_orbitals=5,
            nelecas=(3, 3),
        ),
    ),
    "n2_sto": BenchmarkCase(
        name="N2/STO-3G",
        atom="N 0 0 0; N 0 0 1.09",
        basis="sto-3g",
        ncas=6,
        nelecas=(3, 3),
        # Allow up to 2 holes/particles for better energy recovery
        ras_config=RASConfig(
            ras1_orbitals=[0, 1],
            ras2_orbitals=[2, 3],
            ras3_orbitals=[4, 5],
            max_holes_ras1=2,
            max_particles_ras3=2,
            nelecas=(3, 3),
        ),
        ormas_config=ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 1], min_electrons=1, max_electrons=4),
                Subspace("pi", [2, 3], min_electrons=1, max_electrons=4),
                Subspace("sigma_star", [4, 5], min_electrons=0, max_electrons=2),
            ],
            n_active_orbitals=6,
            nelecas=(3, 3),
        ),
    ),
    "n2_dz": BenchmarkCase(
        name="N2/cc-pVDZ",
        atom="N 0 0 0; N 0 0 1.09",
        basis="cc-pVDZ",
        ncas=8,
        nelecas=(4, 4),
        ras_config=RASConfig(
            ras1_orbitals=[0, 1, 2],
            ras2_orbitals=[3, 4],
            ras3_orbitals=[5, 6, 7],
            max_holes_ras1=2,
            max_particles_ras3=2,
            nelecas=(4, 4),
        ),
        ormas_config=ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
                Subspace("pi", [3, 4], min_electrons=1, max_electrons=4),
                Subspace("sigma_star", [5, 6, 7], min_electrons=0, max_electrons=3),
            ],
            n_active_orbitals=8,
            nelecas=(4, 4),
        ),
    ),
    "n2_large": BenchmarkCase(
        name="N2/cc-pVDZ(10,10)",
        atom="N 0 0 0; N 0 0 1.09",
        basis="cc-pVDZ",
        ncas=10,
        nelecas=(5, 5),
        ras_config=RASConfig(
            ras1_orbitals=[0, 1, 2],
            ras2_orbitals=[3, 4, 5, 6],
            ras3_orbitals=[7, 8, 9],
            max_holes_ras1=2,
            max_particles_ras3=2,
            nelecas=(5, 5),
        ),
        ormas_config=ORMASConfig(
            subspaces=[
                Subspace("core", [0, 1, 2], min_electrons=3, max_electrons=6),
                Subspace("valence", [3, 4, 5, 6], min_electrons=2, max_electrons=8),
                Subspace("virtual", [7, 8, 9], min_electrons=0, max_electrons=3),
            ],
            n_active_orbitals=10,
            nelecas=(5, 5),
        ),
        large=True,
    ),
}


def run_benchmark(case: BenchmarkCase, n_repeats: int) -> list[BenchmarkResult]:
    """Run all configurations for a single benchmark case."""
    print(f"\n{'='*70}")
    print(f"  {case.name}: CAS({sum(case.nelecas)},{case.ncas})")
    print(f"{'='*70}")

    mol = gto.M(atom=case.atom, basis=case.basis, verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    results = []

    # 1. PySCF native FCI
    print("  Running PySCF native FCI...", end=" ", flush=True)
    ref = _run_pyscf_reference(mf, case.ncas, case.nelecas, n_repeats)
    ref.name = case.name
    results.append(ref)
    print(f"E = {ref.energy:.8f} Ha, t = {ref.t_total:.4f}s")

    # 2. ORMAS unrestricted (= full CAS)
    print("  Running ORMAS unrestricted...", end=" ", flush=True)
    unr_config = _make_unrestricted_config(case.ncas, case.nelecas)
    unr = _run_ormas(mf, case.ncas, case.nelecas, unr_config, "ORMAS (full CAS)", n_repeats)
    unr.name = case.name
    results.append(unr)
    de = (unr.energy - ref.energy) * 1000
    print(f"E = {unr.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {unr.t_total:.4f}s")

    # 3. RASCI restricted
    if case.ras_config is not None:
        print("  Running RASCI restricted...", end=" ", flush=True)
        ras_ormas = case.ras_config.to_ormas_config()
        ras = _run_ormas(mf, case.ncas, case.nelecas, ras_ormas, "RASCI", n_repeats)
        ras.name = case.name
        results.append(ras)
        de = (ras.energy - ref.energy) * 1000
        print(f"E = {ras.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {ras.t_total:.4f}s")

    # 4. ORMAS restricted
    if case.ormas_config is not None:
        print("  Running ORMAS restricted...", end=" ", flush=True)
        ormas = _run_ormas(
            mf, case.ncas, case.nelecas, case.ormas_config, "ORMAS restricted", n_repeats
        )
        ormas.name = case.name
        results.append(ormas)
        de = (ormas.energy - ref.energy) * 1000
        print(f"E = {ormas.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {ormas.t_total:.4f}s")

    return results


def print_summary_table(all_results: list[BenchmarkResult]):
    """Print a formatted summary table of all benchmark results."""
    print(f"\n{'='*120}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*120}")

    header = (
        f"{'System':<22} {'Method':<18} {'n_det':>8} {'n_exc_ops':>10} "
        f"{'Reduction':>10} {'dE(mHa)':>10} "
        f"{'t_enum(s)':>10} {'t_ham(s)':>10} {'t_diag(s)':>10} {'t_total(s)':>10}"
    )
    print(header)
    print("-" * 120)

    # Group by system
    systems = {}
    for r in all_results:
        systems.setdefault(r.name, []).append(r)

    for name, results in systems.items():
        ref_n_det = None
        ref_energy = None
        for r in results:
            if r.method == "PySCF FCI":
                ref_n_det = r.n_det
                ref_energy = r.energy
                break

        for r in results:
            if ref_n_det and r.n_det > 0 and r.method != "PySCF FCI":
                reduction = f"{ref_n_det / r.n_det:.1f}x"
            else:
                reduction = "-"

            if ref_energy is not None and r.method != "PySCF FCI":
                de = f"{(r.energy - ref_energy) * 1000:+.4f}"
            else:
                de = "-"

            exc_ops = f"{r.n_excitation_ops}" if r.n_excitation_ops >= 0 else "n/a"

            t_enum = f"{r.t_enum:.4f}" if r.t_enum > 0 else "-"
            t_ham = f"{r.t_hamiltonian:.4f}" if r.t_hamiltonian > 0 else "-"
            t_diag = f"{r.t_diag:.4f}" if r.t_diag > 0 else "-"

            print(
                f"{r.name:<22} {r.method:<18} {r.n_det:>8} {exc_ops:>10} "
                f"{reduction:>10} {de:>10} "
                f"{t_enum:>10} {t_ham:>10} {t_diag:>10} {r.t_total:>10.4f}"
            )
        print()

    # Quantum computing relevance summary
    print(f"{'='*120}")
    print("  QUANTUM COMPUTING RELEVANCE: Determinant & Operator Reduction")
    print(f"{'='*120}")
    print(
        f"{'System':<22} {'Method':<18} {'n_det_full':>12} {'n_det_restr':>12} "
        f"{'Det. Reduction':>15} {'Exc. Ops Full':>14} {'Exc. Ops Restr':>15} "
        f"{'Op. Reduction':>14}"
    )
    print("-" * 120)

    for name, results in systems.items():
        ref_result = None
        full_cas_result = None
        for r in results:
            if r.method == "PySCF FCI":
                ref_result = r
            elif r.method == "ORMAS (full CAS)":
                full_cas_result = r

        if ref_result is None or full_cas_result is None:
            continue

        for r in results:
            if r.method in ("PySCF FCI", "ORMAS (full CAS)"):
                continue

            det_reduction = f"{ref_result.n_det / r.n_det:.1f}x" if r.n_det > 0 else "-"
            if full_cas_result.n_excitation_ops > 0 and r.n_excitation_ops > 0:
                op_reduction = (
                    f"{full_cas_result.n_excitation_ops / r.n_excitation_ops:.1f}x"
                )
            else:
                op_reduction = "-"

            print(
                f"{r.name:<22} {r.method:<18} {ref_result.n_det:>12} {r.n_det:>12} "
                f"{det_reduction:>15} {full_cas_result.n_excitation_ops:>14} "
                f"{r.n_excitation_ops:>15} {op_reduction:>14}"
            )
    print()


def _run_qdk_benchmarks(
    n_repeats: int, all_results: list[BenchmarkResult]
) -> QubitHamiltonianMetrics | None:
    """Run QDK/Chemistry pipeline benchmarks if qdk-chemistry is installed.

    Compares:
    - QDK/Chemistry -> PySCF CASCI (native FCI solver)
    - QDK/Chemistry -> PySCF CASCI + ORMASFCISolver (unrestricted = same energy)
    - QDK/Chemistry -> PySCF CASCI + ORMASFCISolver (restricted = fewer dets)

    Returns QubitHamiltonianMetrics if qubit Hamiltonian was built, else None.
    """
    try:
        from qdk_chemistry.data import Structure
        from qdk_chemistry.plugins.pyscf.conversion import SCFType, orbitals_to_scf
        from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver
    except ImportError:
        print("\n  [QDK/Chemistry not installed — skipping QDK benchmarks]")
        print("  Install with: pip install pyscf-ormas-ci[qdk]")
        return None

    from pyscf.ormas_ci import ORMASConfig, Subspace
    from pyscf.ormas_ci.subspaces import RASConfig

    print(f"\n{'='*70}")
    print("  QDK/Chemistry Integration Benchmarks")
    print(f"{'='*70}")

    # ---------- H2O / STO-3G via QDK pipeline ----------
    print("\n  Setting up H2O via QDK/Chemistry SCF pipeline...")
    xyz_str = (
        "3\nWater\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.000000  0.757000  0.587000\n"
        "H  0.000000 -0.757000  0.587000"
    )
    structure = Structure.from_xyz(xyz_str)
    scf_solver = PyscfScfSolver()
    _scf_energy, wfn = scf_solver.run(structure, 0, 1, "sto-3g")

    container = wfn.get_container()
    orbitals = container.get_orbitals()
    n_mo = orbitals.get_num_molecular_orbitals()
    n_occ = 5
    occ_a = [1] * n_occ + [0] * (n_mo - n_occ)
    occ_b = [1] * n_occ + [0] * (n_mo - n_occ)
    mf = orbitals_to_scf(orbitals, occ_alpha=occ_a, occ_beta=occ_b, scf_type=SCFType.RESTRICTED)

    ncas = 4
    nelecas = (2, 2)
    system_name = "H2O/STO-3G (QDK)"

    # 1. QDK -> PySCF CASCI (native FCI)
    print("  Running QDK -> PySCF CASCI (native FCI)...", end=" ", flush=True)
    ref = _run_pyscf_reference(mf, ncas, nelecas, n_repeats)
    ref.name = system_name
    all_results.append(ref)
    print(f"E = {ref.energy:.8f} Ha, t = {ref.t_total:.4f}s")

    # 2. QDK -> PySCF CASCI + ORMAS (unrestricted = full CAS)
    print("  Running QDK -> CASCI + ORMAS (full CAS)...", end=" ", flush=True)
    unr_config = _make_unrestricted_config(ncas, nelecas)
    unr = _run_ormas(mf, ncas, nelecas, unr_config, "QDK+ORMAS full", n_repeats)
    unr.name = system_name
    all_results.append(unr)
    de = (unr.energy - ref.energy) * 1000
    print(f"E = {unr.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {unr.t_total:.4f}s")

    # 3. QDK -> PySCF CASCI + ORMAS (restricted: bond/lone-pair subspaces)
    print("  Running QDK -> CASCI + ORMAS (restricted)...", end=" ", flush=True)
    restricted_config = ORMASConfig(
        subspaces=[
            Subspace("bond", [0, 1], min_electrons=1, max_electrons=4),
            Subspace("lone_pair", [2, 3], min_electrons=0, max_electrons=3),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    restr = _run_ormas(mf, ncas, nelecas, restricted_config, "QDK+ORMAS restr", n_repeats)
    restr.name = system_name
    all_results.append(restr)
    de = (restr.energy - ref.energy) * 1000
    print(f"E = {restr.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {restr.t_total:.4f}s")

    # 4. QDK -> PySCF CASCI + RASCI
    print("  Running QDK -> CASCI + RASCI...", end=" ", flush=True)
    ras = RASConfig(
        ras1_orbitals=[0],
        ras2_orbitals=[1, 2],
        ras3_orbitals=[3],
        max_holes_ras1=1,
        max_particles_ras3=1,
        nelecas=nelecas,
    )
    ras_result = _run_ormas(mf, ncas, nelecas, ras.to_ormas_config(), "QDK+RASCI", n_repeats)
    ras_result.name = system_name
    all_results.append(ras_result)
    de = (ras_result.energy - ref.energy) * 1000
    print(f"E = {ras_result.energy:.8f} Ha, dE = {de:+.4f} mHa, t = {ras_result.t_total:.4f}s")

    # Summary for QDK section
    print(f"\n  QDK Integration Summary ({system_name}, CAS({sum(nelecas)},{ncas})):")
    print(f"  {'Method':<25} {'n_det':>6} {'n_exc_ops':>10} {'dE(mHa)':>10} {'t_total':>10}")
    print(f"  {'-'*65}")
    for r in [ref, unr, restr, ras_result]:
        de_str = "-" if r.method == "PySCF FCI" else f"{(r.energy - ref.energy)*1000:+.4f}"
        ops_str = "n/a" if r.n_excitation_ops < 0 else str(r.n_excitation_ops)
        print(f"  {r.method:<25} {r.n_det:>6} {ops_str:>10} {de_str:>10} {r.t_total:>10.4f}s")
    print()

    # --- Qubit Hamiltonian metrics (Jordan-Wigner) ---
    qh_result = _build_qubit_hamiltonian(orbitals)
    if qh_result is not None:
        _qubit_ham, qh_metrics = qh_result
        print(f"  Qubit Hamiltonian (Jordan-Wigner, {system_name}):")
        print(f"  {'Qubits:':<25} {qh_metrics.n_qubits}")
        print(f"  {'Pauli terms:':<25} {qh_metrics.n_pauli_terms}")
        print(f"  {'Sparsity:':<25} {qh_metrics.qubit_hamiltonian_sparsity:.6%}")
        print(f"  {'Ham construct time:':<25} {qh_metrics.t_hamiltonian_construct:.4f}s")
        print(f"  {'JW mapping time:':<25} {qh_metrics.t_qubit_mapping:.4f}s")
        print()
        return qh_metrics

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ORMAS-CI against PySCF native FCI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bench_vs_pyscf.py                          # Run default systems
  python bench_vs_pyscf.py --systems h2 n2_sto      # Run specific systems
  python bench_vs_pyscf.py --include-large           # Include large CAS(10,10) benchmark
  python bench_vs_pyscf.py --repeats 5 --output-json # 5 repeats, save JSON
        """,
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(BENCHMARK_CASES.keys()),
        default=None,
        help="Specific systems to benchmark (default: all non-large)",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large benchmark cases (CAS(10,10), may take minutes)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per benchmark (default: 3, reports median)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results as JSON (default: benchmarks/results/latest.json)",
    )
    args = parser.parse_args()

    # Select cases
    if args.systems:
        cases = {k: BENCHMARK_CASES[k] for k in args.systems}
    else:
        cases = {
            k: v
            for k, v in BENCHMARK_CASES.items()
            if not v.large or args.include_large
        }

    print("ORMAS-CI Benchmark Suite")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Systems: {', '.join(cases.keys())}")
    print(f"Repeats: {args.repeats} (reporting median)")

    all_results = []
    for case in cases.values():
        results = run_benchmark(case, args.repeats)
        all_results.extend(results)

    print_summary_table(all_results)

    # QDK/Chemistry benchmarks (if available)
    qh_metrics = None
    if not args.systems or "qdk" in (args.systems or []):
        qh_metrics = _run_qdk_benchmarks(args.repeats, all_results)

    # Save JSON output
    json_path = args.output_json or "benchmarks/results/latest.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "repeats": args.repeats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if qh_metrics is not None:
        metadata["qubit_hamiltonian"] = {
            "n_qubits": qh_metrics.n_qubits,
            "n_pauli_terms": qh_metrics.n_pauli_terms,
            "sparsity": qh_metrics.qubit_hamiltonian_sparsity,
            "t_hamiltonian_construct_s": qh_metrics.t_hamiltonian_construct,
            "t_qubit_mapping_s": qh_metrics.t_qubit_mapping,
        }
    json_data = {
        "metadata": metadata,
        "results": [
            {
                "system": r.name,
                "method": r.method,
                "n_det": r.n_det,
                "n_excitation_ops": r.n_excitation_ops,
                "energy_hartree": r.energy,
                "t_enum_s": r.t_enum,
                "t_hamiltonian_s": r.t_hamiltonian,
                "t_diag_s": r.t_diag,
                "t_total_s": r.t_total,
            }
            for r in all_results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
