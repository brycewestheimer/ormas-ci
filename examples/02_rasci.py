#!/usr/bin/env python3
"""RASCI partitioning on formaldehyde (n/pi/pi* system).

Demonstrates the classic three-space RAS approach:
- RAS1: occupied orbitals with limited holes
- RAS2: frontier orbitals (no restrictions)
- RAS3: virtual orbitals with limited particles

RASConfig is a convenience wrapper that converts to ORMASConfig.
"""

from pyscf import gto, mcscf, scf

from pyscf.ormas_ci import ORMASFCISolver
from pyscf.ormas_ci.determinants import casci_determinant_count, count_determinants
from pyscf.ormas_ci.subspaces import RASConfig

# Formaldehyde: classic n -> pi* photochemistry system
mol = gto.M(
    atom="C 0 0 0; O 0 0 1.208; H 0 0.943 -0.561; H 0 -0.943 -0.561",
    basis="cc-pVDZ",
    verbose=0,
)
mf = scf.RHF(mol)
mf.verbose = 0
mf.run()

ncas, nelecas = 3, (2, 2)

# Full CASCI reference
mc_ref = mcscf.CASCI(mf, ncas, sum(nelecas))
mc_ref.verbose = 0
e_casci = mc_ref.kernel()[0]
n_det_casci = casci_determinant_count(ncas, nelecas)

# RAS partitioning: n_O (RAS1) / pi (RAS2) / pi* (RAS3)
# Allow at most 1 hole in n_O and 1 particle in pi*
ras = RASConfig(
    ras1_orbitals=[0],       # oxygen lone pair
    ras2_orbitals=[1],       # pi bonding (unrestricted)
    ras3_orbitals=[2],       # pi* antibonding
    max_holes_ras1=1,        # at most 1 electron removed from n_O
    max_particles_ras3=1,    # at most 1 electron added to pi*
    nelecas=nelecas,
)

# RASConfig converts to ORMASConfig for the solver
ormas_config = ras.to_ormas_config()

mc_ras = mcscf.CASCI(mf, ncas, sum(nelecas))
mc_ras.verbose = 0
mc_ras.fcisolver = ORMASFCISolver(ormas_config)
e_ras = mc_ras.kernel()[0]
n_det_ras = count_determinants(ormas_config)

# Results
print("RASCI: Formaldehyde/cc-pVDZ CAS(4,3)")
print("=" * 55)
print(f"{'Method':<10} {'Energy (Ha)':>16} {'n_det':>6} {'dE (mHa)':>10}")
print("-" * 55)
print(f"{'CASCI':<10} {e_casci:>16.8f} {n_det_casci:>6} {'-':>10}")
print(f"{'RASCI':<10} {e_ras:>16.8f} {n_det_ras:>6} {(e_ras - e_casci) * 1000:>+10.3f}")
print()
print("RAS partitioning:")
print(f"  RAS1 (n_O):  orbital [0], max {ras.max_holes_ras1} hole(s)")
print("  RAS2 (pi):   orbital [1], unrestricted")
print(f"  RAS3 (pi*):  orbital [2], max {ras.max_particles_ras3} particle(s)")
print()
print(f"Determinant reduction: {n_det_casci} -> {n_det_ras} "
      f"({n_det_casci / n_det_ras:.1f}x)")
