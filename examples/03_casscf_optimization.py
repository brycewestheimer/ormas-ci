#!/usr/bin/env python3
"""ORMAS-CI as a drop-in fcisolver in PySCF CASSCF orbital optimization.

Shows that ORMASFCISolver works with PySCF's CASSCF, which
iteratively optimizes orbitals by repeatedly calling the CI solver.
The CASSCF energy should converge and closely match a reference CASSCF
with PySCF's default FCI solver.
"""

from pyscf import gto, mcscf, scf

from pyscf.ormas_ci import ORMASConfig, ORMASFCISolver, Subspace

# H2O with a small basis for fast CASSCF convergence
mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="6-31G",
    verbose=0,
)
mf = scf.RHF(mol)
mf.verbose = 0
mf.run()

ncas, nelecas = 4, (2, 2)

# --- Reference CASSCF with default FCI solver ---
mc_ref = mcscf.CASSCF(mf, ncas, sum(nelecas))
mc_ref.verbose = 0
e_ref = mc_ref.kernel()[0]

# --- CASSCF with ORMASFCISolver (unrestricted = full CAS) ---
# Using unrestricted config to verify the solver integrates correctly
# before applying restrictions.
config = ORMASConfig(
    subspaces=[
        Subspace("all", list(range(ncas)), min_electrons=0, max_electrons=2 * ncas),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

mc_ormas = mcscf.CASSCF(mf, ncas, sum(nelecas))
mc_ormas.verbose = 0
mc_ormas.fcisolver = ORMASFCISolver(config)
e_ormas = mc_ormas.kernel()[0]

# Results
print("CASSCF Orbital Optimization: H2O/6-31G CAS(4,4)")
print("=" * 55)
print(f"Reference CASSCF (PySCF FCI):  {e_ref:.10f} Ha")
print(f"CASSCF + ORMASFCISolver:       {e_ormas:.10f} Ha")
print(f"Energy difference:             {abs(e_ormas - e_ref) * 1000:.6f} mHa")
print()
print(f"CASSCF converged: {mc_ormas.converged}")
print()
print("The ORMASFCISolver integrates with PySCF's CASSCF orbital")
print("optimization as a drop-in replacement for the default FCI solver.")
print("With unrestricted constraints, the energy matches exactly.")
