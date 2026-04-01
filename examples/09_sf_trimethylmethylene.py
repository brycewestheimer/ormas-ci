#!/usr/bin/env python3
"""SF-ORMAS trimethylenemethane singlet-triplet gap.

Demonstrates spin-flip ORMAS-CI on a larger diradical system with 4 active
electrons. TMM is a benchmark from Mato & Gordon (PCCP 2018). The singlet-
triplet gap is computed via single SF from a triplet reference.
"""

from pyscf import gto, mcscf, scf
from pyscf.ormas_ci import SFORMASConfig, SFORMASFCISolver, Subspace

HA_TO_KCAL = 627.509

# Trimethylenemethane D3h geometry (Angstrom)
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

# --- Triplet ROHF reference ---
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

# --- PySCF CASCI reference CAS(4, (2,2)) ---
mc_ref = mcscf.CASCI(mf, 4, (2, 2))
mc_ref.verbose = 0
mc_ref.fcisolver.nroots = 3
mc_ref.kernel()
e_casci = mc_ref.e_tot

# --- SF-ORMAS CAS(4,4), single SF from triplet, 3 roots ---
sf_config = SFORMASConfig(
    ref_spin=2,
    target_spin=0,
    n_spin_flips=1,
    n_active_orbitals=4,
    n_active_electrons=4,
    subspaces=[
        Subspace("all", [0, 1, 2, 3], min_electrons=0, max_electrons=8),
    ],
)
mc_sf = mcscf.CASCI(mf, 4, (2, 2))
mc_sf.verbose = 0
mc_sf.fcisolver = SFORMASFCISolver(sf_config)
mc_sf.fcisolver.nroots = 3
mc_sf.kernel()
e_sf = mc_sf.e_tot

# --- Analyze spin for each SF-ORMAS root ---
print("SF-ORMAS Trimethylenemethane (6-31G, CAS(4,4))")
print("=" * 70)
print()
print("SF-ORMAS roots:")
print(f"  {'Root':>4}  {'Energy (Ha)':>16}  {'<S^2>':>8}  {'2S+1':>6}")
print("  " + "-" * 42)

s_values = []
for i in range(3):
    ss, mult = mc_sf.fcisolver.spin_square(mc_sf.ci[i], 4, (2, 2))
    s_val = (-1 + (1 + 4 * ss) ** 0.5) / 2
    s_values.append(s_val)
    print(f"  {i:>4}  {e_sf[i]:16.10f}  {ss:8.4f}  {mult:6.2f}")

print()

# --- PySCF CASCI reference ---
print("PySCF CASCI reference:")
print(f"  {'Root':>4}  {'Energy (Ha)':>16}")
print("  " + "-" * 26)
for i in range(3):
    print(f"  {i:>4}  {e_casci[i]:16.10f}")

print()

# --- SF-ORMAS vs CASCI comparison ---
print("SF-ORMAS vs CASCI energy differences:")
for i in range(3):
    diff_mh = (e_sf[i] - e_casci[i]) * 1000.0
    print(f"  Root {i}: {diff_mh:+.3e} mHa")

print()

# --- Singlet-triplet gap ---
# Find lowest singlet and lowest triplet by spin quantum number
singlet_energies = [e_sf[i] for i in range(3) if round(s_values[i]) == 0]
triplet_energies = [e_sf[i] for i in range(3) if round(s_values[i]) == 1]

if singlet_energies and triplet_energies:
    e_singlet = min(singlet_energies)
    e_triplet = min(triplet_energies)
    gap_ha = e_singlet - e_triplet
    gap_kcal = gap_ha * HA_TO_KCAL
    print(f"Lowest singlet energy:  {e_singlet:.10f} Ha")
    print(f"Lowest triplet energy:  {e_triplet:.10f} Ha")
    print(f"Singlet-triplet gap:    {gap_ha:+.10f} Ha = {gap_kcal:+.4f} kcal/mol")
else:
    print("Could not identify distinct singlet and triplet roots.")
    print(f"  S values: {[f'{s:.4f}' for s in s_values]}")
