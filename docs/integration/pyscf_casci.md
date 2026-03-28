# PySCF CASCI Usage

## Basic Pattern

```python
from pyscf import gto, scf, mcscf
from ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

# 1. Standard PySCF setup (molecule, HF)
mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='cc-pvdz',
)
mf = scf.RHF(mol).run()

# 2. Standard CASCI setup (active space choice)
ncas = 8
nelecas = (4, 4)
mc = mcscf.CASCI(mf, ncas, nelecas)

# 3. Define ORMAS subspaces
config = ORMASConfig(
    subspaces=[
        Subspace("sigma", [0, 1, 2, 3], min_electrons=3, max_electrons=5),
        Subspace("pi", [4, 5, 6, 7], min_electrons=3, max_electrons=5),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

# 4. Replace the FCI solver (one line)
mc.fcisolver = ORMASFCISolver(config)

# 5. Run as normal
e_tot, ci_vec = mc.kernel()
print(f"ORMAS-CI energy: {e_tot}")
```

Steps 1-2 and 5 are identical to a standard PySCF CASCI calculation.
The only additions are steps 3-4: defining the subspace configuration
and swapping the solver.

## Requirements for Consistency

The ORMASConfig must be consistent with the CASCI active space:

- `config.n_active_orbitals` must equal `ncas`
- `config.nelecas` must equal the `nelecas` tuple passed to CASCI
- The union of all subspace orbital indices must be exactly `{0, 1, ..., ncas-1}`

If these don't match, `kernel()` raises a ValueError.

## Using RASConfig

For the traditional 3-subspace RAS, use the convenience constructor:

```python
from ormas_ci import RASConfig

ras = RASConfig(
    ras1_orbitals=[0, 1],         # Mostly occupied
    ras2_orbitals=[2, 3, 4, 5],   # Fully flexible
    ras3_orbitals=[6, 7],         # Mostly empty
    max_holes_ras1=2,
    max_particles_ras3=2,
    nelecas=(4, 4),
)

mc.fcisolver = ORMASFCISolver(ras.to_ormas_config())
```

## Accessing Results

After `mc.kernel()`, standard PySCF attributes work:

```python
# Total energy
print(mc.e_tot)

# CI coefficients
print(mc.ci)

# Natural orbital occupations (calls make_rdm1 internally)
mc.analyze()

# Manual RDM access
rdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
```

## Comparing Against Full CASCI

To verify that ORMAS constraints are reasonable, compare against
unrestricted CASCI:

```python
# Reference CASCI
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
e_ref = mc_ref.kernel()[0]

# ORMAS-CI
mc_ormas = mcscf.CASCI(mf, ncas, nelecas)
mc_ormas.fcisolver = ORMASFCISolver(config)
e_ormas = mc_ormas.kernel()[0]

print(f"CASCI energy:    {e_ref:.10f}")
print(f"ORMAS-CI energy: {e_ormas:.10f}")
print(f"Difference:      {e_ormas - e_ref:.6f} Hartree")
```

The ORMAS energy should be slightly above CASCI (variational principle).
If the difference is large, the subspace constraints may be too tight.

## Checking Determinant Reduction

```python
from ormas_ci.determinants import count_determinants, casci_determinant_count

n_ormas = count_determinants(config)
n_casci = casci_determinant_count(ncas, nelecas)
print(f"ORMAS: {n_ormas} determinants ({n_ormas/n_casci:.1%} of CASCI)")
```

## Logging

Enable logging to see determinant counts and subspace details:

```python
import logging
logging.basicConfig(level=logging.INFO)

mc.kernel()  # Will print ORMAS-CI determinant count info
```
