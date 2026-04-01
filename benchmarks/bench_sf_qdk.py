#!/usr/bin/env python3
"""SF-ORMAS QDK resource estimation benchmarks.

Compares SF-ORMAS quantum resource estimates against full CAS for twisted
ethylene CAS(2,2) and TMM CAS(4,4). When qdk_chemistry is available, constructs
qubit Hamiltonians and reports qubit/Pauli metrics. Otherwise, prints classical
results only.

Usage:
  python bench_sf_qdk.py                         # Run all systems
  python bench_sf_qdk.py --systems ethylene       # Run specific system
  python bench_sf_qdk.py --systems ethylene,tmm   # Run multiple systems
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count
from pyscf.ormas_ci.spinflip import count_sf_determinants

try:
    from qdk_chemistry._core.data import (
        CanonicalFourCenterHamiltonianContainer,
        ModelOrbitals,
    )
    from qdk_chemistry._core.data import Hamiltonian as QdkHamiltonian
    from qdk_chemistry.algorithms import QdkQubitMapper

    QDK_AVAILABLE = True
except ImportError:
    QDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SFQDKBenchmarkResult:
    """Results from a single SF-ORMAS QDK benchmark."""

    name: str
    method: str  # "SF-ORMAS" or "CASCI"
    n_det: int
    energy: float
    time_s: float
    s2: float
    n_qubits: int = 0
    n_pauli_terms: int = 0


# ---------------------------------------------------------------------------
# QDK bridge
# ---------------------------------------------------------------------------


def _build_qubit_hamiltonian(
    mf: scf.rohf.ROHF, ncas: int, nelecas: tuple[int, int]
) -> object | None:
    """Build QDK qubit Hamiltonian from PySCF ROHF via ModelOrbitals bridge.

    Since SF-ORMAS always uses ROHF (open-shell), QDK's active-space selectors
    don't work. We extract active-space integrals from PySCF CASCI and build
    the QDK Hamiltonian using CanonicalFourCenterHamiltonianContainer.

    Args:
        mf: Converged PySCF ROHF object.
        ncas: Number of active orbitals.
        nelecas: Tuple of (n_alpha, n_beta) active electrons.

    Returns:
        QDK qubit Hamiltonian, or None if qdk_chemistry is not available.
    """
    if not QDK_AVAILABLE:
        return None

    from pyscf import ao2mo as ao2mo_mod

    mc_tmp = mcscf.CASCI(mf, ncas, sum(nelecas))
    mc_tmp.verbose = 0
    mc_tmp.kernel()
    h1e, ecore = mc_tmp.get_h1eff()
    eri = mc_tmp.get_h2eff()
    if eri.ndim != 4:
        eri = ao2mo_mod.restore(1, eri, ncas)
    eri_flat = eri.reshape(-1)
    fock = np.zeros((ncas, ncas))

    model_orbs = ModelOrbitals(ncas, True)

    container = CanonicalFourCenterHamiltonianContainer(h1e, eri_flat, model_orbs, ecore, fock)
    hamiltonian = QdkHamiltonian(container)
    qubit_ham = QdkQubitMapper().run(hamiltonian)
    return qubit_ham


# ---------------------------------------------------------------------------
# System benchmarks
# ---------------------------------------------------------------------------


def _bench_ethylene() -> list[SFQDKBenchmarkResult]:
    """Twisted ethylene, 6-31G, CAS(2,2), single SF with nroots=2."""
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.340  0.000  0.000
        H -0.500  0.930  0.000
        H -0.500 -0.930  0.000
        H  1.840  0.000  0.930
        H  1.840  0.000 -0.930
        """,
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 2, (1, 1)

    sf_config = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=2,
        n_active_electrons=2,
        subspaces=[Subspace("pi", [0, 1], 0, 4)],
    )

    # QDK qubit Hamiltonian (shared by both methods)
    qubit_ham = _build_qubit_hamiltonian(mf, ncas, nelecas)
    n_qubits = 0
    n_pauli_terms = 0
    if qubit_ham is not None:
        n_qubits = qubit_ham.num_qubits
        n_pauli_terms = len(qubit_ham.pauli_strings)

    results: list[SFQDKBenchmarkResult] = []

    # PySCF native CASCI (nroots=2)
    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    mc_ref.fcisolver.nroots = 2
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    n_det_cas = casci_determinant_count(ncas, nelecas)
    for i, e in enumerate(e_ref):
        ss, _ = mc_ref.fcisolver.spin_square(mc_ref.ci[i], ncas, nelecas)
        results.append(
            SFQDKBenchmarkResult(
                name=f"ethylene_r{i}",
                method="CASCI",
                n_det=n_det_cas,
                energy=float(e),
                time_s=time_cas,
                s2=float(ss),
                n_qubits=n_qubits,
                n_pauli_terms=n_pauli_terms,
            )
        )

    # SF-ORMAS (nroots=2)
    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    mc_sf.fcisolver.nroots = 2
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    n_det_sf = count_sf_determinants(sf_config)
    ci_list = mc_sf.ci if isinstance(mc_sf.ci, list) else [mc_sf.ci]
    e_list = list(e_sf) if hasattr(e_sf, "__iter__") else [e_sf]
    for i, e in enumerate(e_list):
        ss, _ = mc_sf.fcisolver.spin_square(ci_list[i], ncas, nelecas)
        results.append(
            SFQDKBenchmarkResult(
                name=f"ethylene_r{i}",
                method="SF-ORMAS",
                n_det=n_det_sf,
                energy=float(e),
                time_s=time_sf,
                s2=float(ss),
                n_qubits=n_qubits,
                n_pauli_terms=n_pauli_terms,
            )
        )

    return results


def _bench_tmm() -> list[SFQDKBenchmarkResult]:
    """TMM, 6-31G, CAS(4,4), single SF from triplet."""
    mol = gto.M(
        atom="""
        C  0.000  0.000  0.000
        C  1.350  0.000  0.000
        C -0.675  1.169  0.000
        C -0.675 -1.169  0.000
        H  1.944  0.930  0.000
        H  1.944 -0.930  0.000
        H -1.269  2.099  0.000
        H -0.081  2.099  0.000
        H -1.269 -2.099  0.000
        H -0.081 -2.099  0.000
        """,
        basis="6-31g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 4, (2, 2)

    sf_config = SFORMASConfig(
        ref_spin=2,
        target_spin=0,
        n_spin_flips=1,
        n_active_orbitals=4,
        n_active_electrons=4,
        subspaces=[Subspace("pi", [0, 1, 2, 3], 0, 8)],
    )

    # QDK qubit Hamiltonian (shared by both methods)
    qubit_ham = _build_qubit_hamiltonian(mf, ncas, nelecas)
    n_qubits = 0
    n_pauli_terms = 0
    if qubit_ham is not None:
        n_qubits = qubit_ham.num_qubits
        n_pauli_terms = len(qubit_ham.pauli_strings)

    results: list[SFQDKBenchmarkResult] = []

    # PySCF native CASCI
    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    ss_ref, _ = mc_ref.fcisolver.spin_square(mc_ref.ci, ncas, nelecas)
    n_det_cas = casci_determinant_count(ncas, nelecas)
    results.append(
        SFQDKBenchmarkResult(
            name="tmm",
            method="CASCI",
            n_det=n_det_cas,
            energy=float(e_ref),
            time_s=time_cas,
            s2=float(ss_ref),
            n_qubits=n_qubits,
            n_pauli_terms=n_pauli_terms,
        )
    )

    # SF-ORMAS
    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    ss_sf, _ = mc_sf.fcisolver.spin_square(mc_sf.ci, ncas, nelecas)
    n_det_sf = count_sf_determinants(sf_config)
    results.append(
        SFQDKBenchmarkResult(
            name="tmm",
            method="SF-ORMAS",
            n_det=n_det_sf,
            energy=float(e_sf),
            time_s=time_sf,
            s2=float(ss_sf),
            n_qubits=n_qubits,
            n_pauli_terms=n_pauli_terms,
        )
    )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

BENCHMARK_SYSTEMS: dict[str, callable] = {
    "ethylene": _bench_ethylene,
    "tmm": _bench_tmm,
}


def print_results(results: list[SFQDKBenchmarkResult]) -> None:
    """Print a comparison table of benchmark results.

    Args:
        results: List of benchmark results to display.
    """
    has_qdk = any(r.n_qubits > 0 for r in results)

    print()
    print("SF-ORMAS QDK Resource Estimation Benchmarks")
    print("=" * 76)
    header = (
        f"{'System':<16} {'Method':<10} {'n_det':>5}  "
        f"{'Energy(Ha)':>12}  {'Time(s)':>7}  {'<S^2>':>5}"
    )
    if has_qdk:
        header += f"  {'Qubits':>6}  {'Pauli':>6}"
    else:
        header += f"  {'Qubits':>6}  {'Pauli':>6}"
    print(header)
    print("-" * 76)

    for r in results:
        qubits_str = str(r.n_qubits) if r.n_qubits > 0 else "N/A"
        pauli_str = str(r.n_pauli_terms) if r.n_pauli_terms > 0 else "N/A"
        print(
            f"{r.name:<16} {r.method:<10} {r.n_det:>5}  "
            f"{r.energy:>12.5f}  {r.time_s:>7.3f}  {r.s2:>5.2f}"
            f"  {qubits_str:>6}  {pauli_str:>6}"
        )

    print("=" * 76)
    if not has_qdk:
        print("Note: qdk_chemistry not installed. Qubits/Pauli columns show N/A.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run SF-ORMAS QDK resource estimation benchmarks."""
    parser = argparse.ArgumentParser(
        description="SF-ORMAS QDK resource estimation benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bench_sf_qdk.py                         # Run all systems
  python bench_sf_qdk.py --systems ethylene       # Run specific system
  python bench_sf_qdk.py --systems ethylene,tmm   # Run multiple systems
        """,
    )
    parser.add_argument(
        "--systems",
        type=str,
        default=None,
        help=(
            "Comma-separated list of systems to benchmark "
            f"(choices: {', '.join(BENCHMARK_SYSTEMS)}; default: all)"
        ),
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.systems:
        names = [s.strip() for s in args.systems.split(",")]
        for name in names:
            if name not in BENCHMARK_SYSTEMS:
                parser.error(f"Unknown system: {name}")
        systems = {k: BENCHMARK_SYSTEMS[k] for k in names}
    else:
        systems = BENCHMARK_SYSTEMS

    print("SF-ORMAS QDK Resource Estimation Benchmarks")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"QDK available: {QDK_AVAILABLE}")
    print(f"Systems: {', '.join(systems)}")
    print()

    all_results: list[SFQDKBenchmarkResult] = []
    for name, bench_fn in systems.items():
        print(f"Running {name}...", flush=True)
        all_results.extend(bench_fn())

    print_results(all_results)

    # Save JSON
    json_path = args.output_json or "benchmarks/results/sf_qdk_latest.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        "metadata": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "qdk_available": QDK_AVAILABLE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": [asdict(r) for r in all_results],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
