#!/usr/bin/env python3
"""SF-ORMAS-CI benchmarks: performance and determinant reduction metrics.

Compares SF-ORMAS against PySCF's native CASCI across several test systems,
measuring determinant reduction, wall time, energy accuracy, and spin purity.

Usage:
  python bench_sf_ormas.py                    # Run all systems
  python bench_sf_ormas.py --systems h2,tmm   # Run specific systems
"""

import argparse
import time
from dataclasses import dataclass

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace
from pyscf.ormas_ci.determinants import casci_determinant_count
from pyscf.ormas_ci.spinflip import count_sf_determinants


@dataclass
class SFBenchmarkResult:
    """Results from a single SF-ORMAS benchmark."""

    name: str
    n_det_sf: int
    n_det_cas: int
    time_sf: float
    time_cas: float
    delta_e: float
    s2: float


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------


def _bench_h2() -> SFBenchmarkResult:
    """H2 stretched, STO-3G, CAS(2,2), single SF."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 2, (1, 1)

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("sf_cas", [0, 1], 0, 4)],
    )

    # PySCF native CASCI
    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    # SF-ORMAS
    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    ss, _ = mc_sf.fcisolver.spin_square(mc_sf.ci, ncas, nelecas)

    return SFBenchmarkResult(
        name="h2",
        n_det_sf=count_sf_determinants(sf_config),
        n_det_cas=casci_determinant_count(ncas, nelecas),
        time_sf=time_sf,
        time_cas=time_cas,
        delta_e=abs(e_sf - e_ref),
        s2=ss,
    )


def _bench_ethylene() -> SFBenchmarkResult:
    """Twisted ethylene, STO-3G, CAS(2,2), single SF."""
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
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 2, (1, 1)

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=2, n_active_electrons=2,
        subspaces=[Subspace("pi", [0, 1], 0, 4)],
    )

    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    ss, _ = mc_sf.fcisolver.spin_square(mc_sf.ci, ncas, nelecas)

    return SFBenchmarkResult(
        name="ethylene",
        n_det_sf=count_sf_determinants(sf_config),
        n_det_cas=casci_determinant_count(ncas, nelecas),
        time_sf=time_sf,
        time_cas=time_cas,
        delta_e=abs(e_sf - e_ref),
        s2=ss,
    )


def _bench_tmm() -> SFBenchmarkResult:
    """TMM, STO-3G, CAS(4,4), single SF from triplet."""
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
        basis="sto-3g", spin=2, verbose=0,
    )
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 4, (2, 2)

    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=4, n_active_electrons=4,
        subspaces=[Subspace("pi", [0, 1, 2, 3], 0, 8)],
    )

    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    ss, _ = mc_sf.fcisolver.spin_square(mc_sf.ci, ncas, nelecas)

    return SFBenchmarkResult(
        name="tmm",
        n_det_sf=count_sf_determinants(sf_config),
        n_det_cas=casci_determinant_count(ncas, nelecas),
        time_sf=time_sf,
        time_cas=time_cas,
        delta_e=abs(e_sf - e_ref),
        s2=ss,
    )


def _bench_n2() -> SFBenchmarkResult:
    """N2 stretched, STO-3G, CAS(6,6), single SF with ORMAS restriction."""
    mol = gto.M(atom="N 0 0 0; N 0 0 2.0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.verbose = 0
    mf.run()

    ncas, nelecas = 6, (3, 3)

    # ORMAS restriction: sigma and pi subspaces
    sf_config = SFORMASConfig(
        ref_spin=2, target_spin=0, n_spin_flips=1,
        n_active_orbitals=6, n_active_electrons=6,
        subspaces=[
            Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
            Subspace("pi", [3, 4, 5], min_electrons=0, max_electrons=4),
        ],
    )

    t0 = time.perf_counter()
    mc_ref = mcscf.CASCI(mf, ncas, nelecas)
    mc_ref.verbose = 0
    e_ref = mc_ref.kernel()[0]
    time_cas = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_sf = mcscf.CASCI(mf, ncas, nelecas)
    mc_sf.verbose = 0
    mc_sf.fcisolver = SFORMASFCISolver(sf_config)
    e_sf = mc_sf.kernel()[0]
    time_sf = time.perf_counter() - t0

    ss, _ = mc_sf.fcisolver.spin_square(mc_sf.ci, ncas, nelecas)

    return SFBenchmarkResult(
        name="n2",
        n_det_sf=count_sf_determinants(sf_config),
        n_det_cas=casci_determinant_count(ncas, nelecas),
        time_sf=time_sf,
        time_cas=time_cas,
        delta_e=abs(e_sf - e_ref),
        s2=ss,
    )


BENCHMARK_SYSTEMS: dict[str, callable] = {
    "h2": _bench_h2,
    "ethylene": _bench_ethylene,
    "tmm": _bench_tmm,
    "n2": _bench_n2,
}


def print_results(results: list[SFBenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print()
    print("SF-ORMAS-CI Benchmark Results")
    print("=" * 90)
    header = (
        f"{'System':<12} {'n_det(SF)':>10} {'n_det(CAS)':>11} "
        f"{'Reduction':>10} {'Time(SF)':>9} {'Time(CAS)':>10} "
        f"{'dE(Ha)':>10} {'<S^2>':>7}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        pct = r.n_det_sf / r.n_det_cas * 100 if r.n_det_cas > 0 else 0.0
        print(
            f"{r.name:<12} {r.n_det_sf:>10} {r.n_det_cas:>11} "
            f"{pct:>9.1f}% {r.time_sf:>8.3f}s {r.time_cas:>9.3f}s "
            f"{r.delta_e:>10.2e} {r.s2:>7.3f}"
        )

    print("=" * 90)
    print()


def main() -> None:
    """Run SF-ORMAS benchmarks."""
    parser = argparse.ArgumentParser(
        description="SF-ORMAS-CI benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    args = parser.parse_args()

    if args.systems:
        names = [s.strip() for s in args.systems.split(",")]
        for name in names:
            if name not in BENCHMARK_SYSTEMS:
                parser.error(f"Unknown system: {name}")
        systems = {k: BENCHMARK_SYSTEMS[k] for k in names}
    else:
        systems = BENCHMARK_SYSTEMS

    results = []
    for name, bench_fn in systems.items():
        print(f"Running {name}...", flush=True)
        results.append(bench_fn())

    print_results(results)


if __name__ == "__main__":
    main()
