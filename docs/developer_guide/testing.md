# Testing

## Running Tests

```bash
pytest                           # All tests
pytest tests/test_fcisolver.py   # Single file
pytest tests/test_fcisolver.py::test_function_name  # Single test
pytest -m "not slow"             # Skip slow tests
pytest --cov=pyscf.ormas_ci --cov-report=term-missing  # With coverage
```

## Test File Organization

Each source module has a corresponding test file, plus integration test
files:

| Test File | What It Tests |
|-----------|--------------|
| `test_fcisolver.py` | Core PySCF integration, `kernel()`, solver path selection |
| `test_determinants.py` | Determinant enumeration, `count_determinants`, `build_determinant_list` |
| `test_hamiltonian.py` | Hamiltonian matrix construction |
| `test_rdm.py` | Reduced density matrices (`make_rdm1`, `make_rdm12`) |
| `test_rdm_advanced.py` | Advanced RDM functionality |
| `test_molecules.py` | Multi-molecule validation across systems |
| `test_pyscf_compat.py` | PySCF compatibility methods (`contract_2e`, `absorb_h1e`, etc.) |
| `test_sci_integration.py` | PySCF `selected_ci` backend integration |
| `test_sf_integration.py` | Spin-flip ORMAS integration |
| `test_edge_cases.py` | Edge cases and error handling |
| `test_completeness.py` | Full CAS equivalence (ORMAS == CASCI when unrestricted) |
| `test_coverage.py` | Coverage for less-exercised code paths |
| `test_qdk_integration.py` | QDK/Chemistry integration (requires `qdk-chemistry`) |
| `test_qdk_quantum_benchmarks.py` | QDK quantum benchmark validation |

## Shared Fixtures

`tests/conftest.py` provides common fixtures:

```python
# Molecules and SCF
h2_mol        # H2 at equilibrium, 6-31G basis
h2_rhf        # Converged RHF for H2

# Reference calculations
h2_casci      # PySCF CASCI(2,2) reference

# ORMAS configuration
h2_ormas_config   # Unrestricted ORMAS config for CAS(2,2)
h2_ormas_solver   # Converged ORMAS CASCI(2,2)

# Raw integrals
h2_active_integrals   # (h1e, h2e_4d, ecore, e_fci) for H2 CAS(2,2)
```

## Standard Tolerances

Defined in `conftest.py` and used across all test files:

```python
ENERGY_ATOL = 1e-10   # Energy agreement with PySCF reference
RDM_ATOL = 1e-10      # RDM element agreement
S2_ATOL = 1e-6        # Spin expectation value agreement
```

## Writing New Tests

### General Pattern

The gold standard test: unrestricted ORMAS (single subspace, full
bounds) must reproduce PySCF native CASCI exactly.

```python
def test_my_feature(h2_rhf, h2_casci):
    """Unrestricted ORMAS reproduces CASCI for H2."""
    config = ORMASConfig.unrestricted(ncas=2, nelecas=(1, 1))
    mc = mcscf.CASCI(h2_rhf, 2, 2)
    mc.verbose = 0
    mc.fcisolver = ORMASFCISolver(config)
    e_test = mc.kernel()[0]

    assert abs(e_test - h2_casci.e_tot) < ENERGY_ATOL
```

### Guidelines

- Always compare against a PySCF native CASCI reference
- Test both small (dense `eigh`) and medium (Davidson) solver paths
  when relevant
- Set `mc.verbose = 0` to suppress PySCF output in tests
- Use `conftest.py` fixtures for common molecules
- Mark slow tests with `@pytest.mark.slow`

## CI Pipeline

The GitHub Actions CI (`.github/workflows/ci.yml`) runs on every push
and PR to `main`:

| Job | What It Does |
|-----|-------------|
| `test` | Lint (ruff), type check (pyright), tests (pytest + coverage) on Python 3.11, 3.12, 3.13 |
| `docs` | Build documentation with `sphinx-build -W` (warnings as errors) |
| `package` | Build sdist/wheel, smoke-test installed wheel import |
| `qdk-smoke` | Install with QDK extra, run QDK integration tests (allowed to fail) |
| `deploy-docs` | Deploy to GitHub Pages on merge to `main` |
