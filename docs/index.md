# ORMAS-CI Documentation

## Theory

Background on the quantum chemistry methods implemented in this package.

- [Complete Active Space CI (CASCI)](theory/casci.md)
- [Restricted Active Space CI (RASCI)](theory/rasci.md)
- [Occupation-Restricted Multiple Active Spaces CI (ORMAS-CI)](theory/ormas.md)
- [Generalized Active Space CI (GAS-CI)](theory/gas.md)
- [Comparison of Methods](theory/comparison.md)
- [Slater-Condon Rules](theory/slater_condon.md)
- [Quantum Computing Relevance](theory/quantum_computing.md)

## Architecture

How the code is structured and why.

- [Module Overview](architecture/overview.md)
- [Data Flow](architecture/data_flow.md)
- [Bit String Representation](architecture/bitstrings.md)
- [Integral Conventions](architecture/integrals.md)
- [Determinant Enumeration Algorithm](architecture/determinant_enumeration.md)
- [Phase Factor Computation](architecture/phase_factors.md)

## Design

Design decisions, tradeoffs, and rationale.

- [PySCF Integration Strategy](design/pyscf_integration.md)
- [QDK/Chemistry Integration Path](design/qdk_integration.md)
- [Subspace Configuration Model](design/subspace_model.md)
- [Performance Considerations](design/performance.md)
- [Future Directions](design/future.md)

## Integration

How to use this package with external tools.

- [PySCF CASCI Usage](integration/pyscf_casci.md)
- [QDK/Chemistry Pipeline](integration/qdk_chemistry.md)
- [Forte/Psi4 Comparison](integration/forte_comparison.md)

## API Reference

- [Public API](api/public_api.md)
- [Internal Modules](api/internals.md)

## Guides

- [Getting Started](guides/getting_started.md)
- [Choosing Subspaces](guides/choosing_subspaces.md)
- [Common Recipes](guides/recipes.md)
