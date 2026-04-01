# Forte/Psi4 Comparison

## What is Forte?

Forte is an open-source quantum chemistry package developed by the
Evangelista group at Emory University. It is implemented as a Psi4
plugin and provides a wide range of multireference methods including
CASCI, RASCI, GAS-CI, ORMAS-CI, selected CI (ACI), DSRG-MRPT2, and
various multireference perturbation theories.

Repository: github.com/evangelistalab/forte
License: LGPL-3.0

## Forte's RAS/GAS/ORMAS Implementation

Forte implements all three restricted active space variants in the
determinant basis with both local (ORMAS) and cumulative (GAS)
constraint support. Key features:

- Arbitrary number of subspaces
- Both local and cumulative constraints
- String-based determinant representation (similar to our approach)
- Efficient sigma-vector construction for direct CI
- Integration with Psi4's orbital optimization for RASSCF/GASSCF
- Support for state-averaged calculations
- Analytical gradients via 2-RDM

Forte is a mature, well-tested, production-quality implementation.

## How This Package Differs from Forte

**Integration target:** This package targets PySCF and QDK/Chemistry.
Forte targets Psi4. Since QDK/Chemistry currently has a PySCF plugin
but not a Psi4 plugin, our implementation provides the immediate
integration path.

**Scope:** This package implements ORMAS-CI only (with RAS as a special
case). Forte implements RAS, GAS, ORMAS, selected CI, MRPT2, and more.
Our scope is deliberately narrow to demonstrate the method and
integration, not to be a comprehensive multireference code.

**Performance:** Forte uses optimized C++ with efficient sigma-vector
algorithms (direct CI). Our implementation uses Python with a three-path
solver (dense diagonalization for small spaces, Davidson+einsum for
medium, PySCF's C-level selected CI sigma for large). Forte is faster
for large determinant spaces, but for typical ORMAS-restricted spaces
our solver is competitive.

**Architecture:** Forte is tightly integrated with Psi4's C++ infrastructure.
Our package is a standalone Python package with no C++ dependencies
(beyond numpy/scipy). This makes it easier to install and use as a
proof of concept, but limits performance.

## Complementary Roles

If QDK/Chemistry adds a Psi4 backend in the future, Forte's implementation
could provide production-quality RAS/GAS/ORMAS support directly. Our
package serves as the proof of concept and the PySCF integration path
for the current QDK/Chemistry architecture.

The two are not competing implementations. They serve different
integration paths within the same quantum chemistry ecosystem.

## Key Reference

Evangelista, F.A. et al. "Forte: A suite of advanced multireference
quantum chemistry methods." J. Chem. Phys. 2024, 161, 062502.
