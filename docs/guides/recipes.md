# Common Recipes

## Recipe 1: Unrestricted ORMAS (Verify Against CASCI)

Always start here. If the unrestricted ORMAS energy doesn't match
CASCI exactly, something is wrong.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol).run()

# Full CASCI reference
mc_ref = mcscf.CASCI(mf, 2, 2)
e_ref = mc_ref.kernel()[0]

# Unrestricted ORMAS (single subspace, no constraints)
config = ORMASConfig(
    subspaces=[Subspace("all", [0, 1], min_electrons=0, max_electrons=4)],
    n_active_orbitals=2,
    nelecas=(1, 1),
)
mc = mcscf.CASCI(mf, 2, 2)
mc.fcisolver = ORMASFCISolver(config)
e_ormas = mc.kernel()[0]

assert abs(e_ref - e_ormas) < 1e-10, "Mismatch! Check installation."
print(f"Match confirmed: {e_ref:.10f}")
```

## Recipe 2: RASCI on N2

Sigma core in RAS1, valence in RAS2, virtual in RAS3.

```python
from pyscf.ormas_ci import RASConfig

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()

ncas, nelecas = 6, (3, 3)

ras = RASConfig(
    ras1_orbitals=[0],           # Sigma bonding (mostly occupied)
    ras2_orbitals=[1, 2, 3, 4],  # Valence pi/pi* (full CI)
    ras3_orbitals=[5],           # Sigma antibonding (mostly empty)
    max_holes_ras1=1,
    max_particles_ras3=1,
    nelecas=nelecas,
)

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(ras.to_ormas_config())
e_ras = mc.kernel()[0]
print(f"RASCI energy: {e_ras:.10f}")
```

## Recipe 3: Two-Subspace ORMAS (Metal + Ligand)

```python
mol = gto.M(atom='...', basis='cc-pvdz')  # your transition metal complex
mf = scf.RHF(mol).run()

ncas = 10
nelecas = (5, 5)

config = ORMASConfig(
    subspaces=[
        Subspace("metal_3d", [0,1,2,3,4], min_electrons=4, max_electrons=8),
        Subspace("ligand", [5,6,7,8,9], min_electrons=2, max_electrons=6),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
e_tot = mc.kernel()[0]
```

## Recipe 4: Three Subspaces (Sigma / Pi / Lone Pair)

```python
config = ORMASConfig(
    subspaces=[
        Subspace("sigma", [0, 1], min_electrons=1, max_electrons=3),
        Subspace("pi", [2, 3, 4, 5], min_electrons=3, max_electrons=5),
        Subspace("lone_pair", [6, 7], min_electrons=1, max_electrons=3),
    ],
    n_active_orbitals=8,
    nelecas=(4, 4),
)
```

## Recipe 5: Convergence Study (Relaxing Constraints)

Systematically relax constraints to see how the energy converges to CASCI.

```python
from pyscf.ormas_ci.determinants import count_determinants, casci_determinant_count

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()
ncas, nelecas = 6, (3, 3)

# CASCI reference
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
e_ref = mc_ref.kernel()[0]

print(f"{'Allowance':>10} {'Energy':>16} {'Error (mH)':>12} {'N_det':>8} {'%CASCI':>8}")
print("-" * 60)

for allow in [0, 1, 2, 3]:
    min_e = max(0, 3 - allow)
    max_e = min(6, 3 + allow)
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0, 1, 2], min_e, max_e),
            Subspace("B", [3, 4, 5], min_e, max_e),
        ],
        n_active_orbitals=ncas,
        nelecas=nelecas,
    )
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = ORMASFCISolver(config)
    mc.verbose = 0
    e = mc.kernel()[0]
    n = count_determinants(config)
    n_cas = casci_determinant_count(ncas, nelecas)
    error_mh = (e - e_ref) * 1000
    print(f"{allow:>10} {e:>16.10f} {error_mh:>12.4f} {n:>8} {n/n_cas:>7.1%}")
```

## Recipe 6: Natural Orbital Analysis

```python
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
mc.kernel()

# This calls make_rdm1 internally and prints occupation numbers
mc.analyze()

# Manual access to the 1-RDM
import numpy as np
rdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
occupations = np.linalg.eigvalsh(rdm1)[::-1]
print("Natural orbital occupations:", occupations)
```

## Recipe 7: Estimating Determinant Count Before Running

Check the space size before committing to a full calculation:

```python
from pyscf.ormas_ci.determinants import count_determinants, casci_determinant_count

config = ORMASConfig(
    subspaces=[
        Subspace("A", list(range(7)), min_electrons=5, max_electrons=9),
        Subspace("B", list(range(7, 14)), min_electrons=5, max_electrons=9),
    ],
    n_active_orbitals=14,
    nelecas=(7, 7),
)

n = count_determinants(config)
n_cas = casci_determinant_count(14, (7, 7))
print(f"ORMAS: {n:,} determinants")
print(f"CASCI: {n_cas:,} determinants")
print(f"Reduction: {n/n_cas:.2%}")

if n > 50_000:
    print("Warning: large determinant space. Consider tighter ORMAS constraints.")
```
