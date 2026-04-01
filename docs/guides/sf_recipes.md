# Spin-Flip ORMAS Recipes

Common SF-ORMAS workflows, from the simplest case to advanced usage.
SF-ORMAS uses a high-spin ROHF reference and flips one or more spins
to access the target multiplicity, capturing strong correlation in
bond-breaking and diradical systems.

## Recipe 1: Minimal SF-ORMAS (H2 Single Spin-Flip)

The simplest possible SF-ORMAS calculation. Stretched H2 with a triplet
ROHF reference, single spin-flip to the singlet sector.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

# Stretched H2 (diradical character)
mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='6-31g', spin=2, verbose=0)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

# Single SF: triplet (2S=2) -> singlet (2S=0)
sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("sigma", [0, 1], 0, 4)],
)

n_a, n_b = sf_config.nelecas_target  # (1, 1)
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
e = mc.kernel()[0]

ss, mult = mc.fcisolver.spin_square(mc.ci, 2, (n_a, n_b))
print(f"SF-ORMAS energy: {e:.10f} Ha")
print(f"<S^2> = {ss:.6f}, 2S+1 = {mult:.3f}")
```

## Recipe 2: Diradical Excited States (Ethylene Torsion)

Multi-root SF-ORMAS on twisted ethylene. The M_s=0 sector contains
both singlet and triplet states.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

mol = gto.M(
    atom="""
    C  0.000  0.000  0.000
    C  1.340  0.000  0.000
    H -0.500  0.930  0.000
    H -0.500 -0.930  0.000
    H  1.840  0.000  0.930
    H  1.840  0.000 -0.930
    """,
    basis="6-31g", spin=2, verbose=0,
)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("pi", [0, 1], 0, 4)],
)

n_a, n_b = sf_config.nelecas_target
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
mc.fcisolver.nroots = 2
mc.kernel()

for i, e in enumerate(mc.e_tot):
    ci = mc.ci[i]
    ss, mult = mc.fcisolver.spin_square(ci, 2, (n_a, n_b))
    state = "singlet" if ss < 0.5 else "triplet"
    print(f"Root {i}: E = {e:.8f}, <S^2> = {ss:.4f}, {state}")
```

**Tip:** The root ordering depends on the system. At 90-degree twist,
singlet and triplet are nearly degenerate. Use S-squared to identify
which is which.

## Recipe 3: SF-ORMAS with Hole/Particle Spaces

Adding dynamic correlation via ORMAS restrictions. TMM
(trimethylenemethane) with a restricted multi-subspace config.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace
from pyscf.ormas_ci.spinflip import count_sf_determinants
from pyscf.ormas_ci.determinants import casci_determinant_count

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
    basis="6-31g", spin=2, verbose=0,
)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

# ORMAS with 2 subspaces: restrict inter-subspace excitations
sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=4, n_active_electrons=4,
    subspaces=[
        Subspace("pi_bond", [0, 1], min_electrons=1, max_electrons=3),
        Subspace("pi_anti", [2, 3], min_electrons=1, max_electrons=3),
    ],
)

n_a, n_b = sf_config.nelecas_target
mc = mcscf.CASCI(mf, 4, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
e = mc.kernel()[0]

n_sf = count_sf_determinants(sf_config)
n_cas = casci_determinant_count(4, (n_a, n_b))
print(f"SF-ORMAS energy: {e:.10f}")
print(f"Determinants: {n_sf} (vs {n_cas} full CAS, {n_sf/n_cas:.0%})")
```

**Tip:** Restricting inter-subspace excitations reduces determinant count
while retaining the most important configurations. The tighter the
bounds, the fewer determinants but the larger the energy error.

## Recipe 4: Multi-Root Excited States with Spin Diagnostics

Get multiple states and identify their spin character.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='cc-pvdz', spin=2, verbose=0)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("all", [0, 1], 0, 4)],
)

n_a, n_b = sf_config.nelecas_target
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
mc.fcisolver.nroots = 2
mc.kernel()

print(f"{'Root':>4} {'Energy (Ha)':>16} {'<S^2>':>8} {'2S+1':>6} {'State':>10}")
print("-" * 50)
for i, e in enumerate(mc.e_tot):
    ci = mc.ci[i]
    ss, mult = mc.fcisolver.spin_square(ci, 2, (n_a, n_b))
    if abs(ss) < 0.1:
        state = "singlet"
    elif abs(ss - 2.0) < 0.1:
        state = "triplet"
    else:
        state = f"mixed ({ss:.2f})"
    print(f"{i:>4} {e:>16.10f} {ss:>8.4f} {mult:>6.2f} {state:>10}")
```

**Tip:** S-squared of approximately 0 indicates a singlet, approximately
2 indicates a triplet, and approximately 6 indicates a quintet. Values
significantly off from these indicate spin contamination -- check your
subspace bounds.

## Recipe 5: SF-ORMAS + CASSCF Orbital Optimization

Combine spin-flip CI with orbital optimization.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='6-31g', spin=2, verbose=0)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("sigma", [0, 1], 0, 4)],
)

n_a, n_b = sf_config.nelecas_target

# CASCI (no orbital optimization)
mc_ci = mcscf.CASCI(mf, 2, (n_a, n_b))
mc_ci.verbose = 0
mc_ci.fcisolver = SFORMASFCISolver(sf_config)
e_ci = mc_ci.kernel()[0]

# CASSCF (with orbital optimization)
mc_scf = mcscf.CASSCF(mf, 2, (n_a, n_b))
mc_scf.verbose = 0
mc_scf.fcisolver = SFORMASFCISolver(sf_config)
e_scf = mc_scf.kernel()[0]

print(f"SF-CASCI:  {e_ci:.10f} Ha")
print(f"SF-CASSCF: {e_scf:.10f} Ha")
print(f"CASSCF lowers energy by {(e_ci - e_scf) * 1000:.4f} mHa")
```

**Tip:** CASSCF orbital optimization is especially important when the
ROHF orbitals are not optimal for the target state. For small active
spaces, the improvement may be small.

## Recipe 6: Comparing SF-ORMAS Against Full CASCI

Validate that SF-ORMAS with full CAS bounds reproduces CASCI exactly.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import SFORMASFCISolver, SFORMASConfig, Subspace

mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='6-31g', spin=2, verbose=0)
mf = scf.ROHF(mol)
mf.verbose = 0
mf.run()

ncas, nelecas = 2, (1, 1)

# PySCF native CASCI
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
mc_ref.verbose = 0
e_ref = mc_ref.kernel()[0]

# SF-ORMAS with unrestricted bounds (should match CASCI)
sf_config = SFORMASConfig(
    ref_spin=2, target_spin=0, n_spin_flips=1,
    n_active_orbitals=2, n_active_electrons=2,
    subspaces=[Subspace("all", [0, 1], 0, 4)],
)

mc_sf = mcscf.CASCI(mf, ncas, nelecas)
mc_sf.verbose = 0
mc_sf.fcisolver = SFORMASFCISolver(sf_config)
e_sf = mc_sf.kernel()[0]

diff = abs(e_ref - e_sf)
print(f"CASCI:     {e_ref:.10f} Ha")
print(f"SF-ORMAS:  {e_sf:.10f} Ha")
print(f"Difference: {diff:.2e} Ha")
assert diff < 1e-10, "Mismatch! Check your configuration."
print("Match confirmed -- SF-ORMAS reproduces CASCI exactly.")
```

**Tip:** Always run this validation first with a new system. If the
unrestricted SF-ORMAS does not match CASCI, there is a configuration
error.
