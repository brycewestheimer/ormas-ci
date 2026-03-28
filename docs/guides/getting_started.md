# Getting Started

## Installation

```bash
pip install pyscf-ormas-ci
```

Or from source:

```bash
git clone https://github.com/brycewestheimer/ormas-ci.git
cd ormas-ci
pip install -e .
```

Dependencies (installed automatically): numpy, scipy, pyscf.

## Your First ORMAS-CI Calculation

This example runs ORMAS-CI on H2O with a sigma/pi orbital partitioning.

```python
from pyscf import gto, scf, mcscf
from pyscf.ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

# Set up the molecule
mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='sto-3g',
)
mf = scf.RHF(mol).run()

# Choose an active space
ncas = 4       # 4 active orbitals
nelecas = (3, 3)  # 6 active electrons (3 alpha, 3 beta)

# Define two subspaces with occupation constraints
config = ORMASConfig(
    subspaces=[
        Subspace("sigma", [0, 1], min_electrons=1, max_electrons=4),
        Subspace("pi", [2, 3], min_electrons=1, max_electrons=4),
    ],
    n_active_orbitals=ncas,
    nelecas=nelecas,
)

# Run ORMAS-CI through PySCF's CASCI
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
e_tot, ci = mc.kernel()

print(f"ORMAS-CI energy: {e_tot:.10f} Hartree")
```

## Comparing Against Full CASCI

Always compare your ORMAS-CI results against unrestricted CASCI to
verify that your constraints are reasonable:

```python
# Full CASCI reference
mc_ref = mcscf.CASCI(mf, ncas, nelecas)
e_ref = mc_ref.kernel()[0]

print(f"CASCI energy:    {e_ref:.10f}")
print(f"ORMAS-CI energy: {e_tot:.10f}")
print(f"Difference:      {e_tot - e_ref:.2e} Hartree")
```

The ORMAS energy should be at or slightly above the CASCI energy.
If the difference is larger than a few milliHartree, your constraints
may be excluding important configurations.

## Checking the Determinant Reduction

```python
from pyscf.ormas_ci.determinants import count_determinants, casci_determinant_count

n_ormas = count_determinants(config)
n_casci = casci_determinant_count(ncas, nelecas)
print(f"ORMAS: {n_ormas} determinants")
print(f"CASCI: {n_casci} determinants")
print(f"Reduction: {n_ormas/n_casci:.1%}")
```

## Next Steps

- See [Choosing Subspaces](choosing_subspaces.md) for guidance on how
  to partition your active orbitals.
- See [Common Recipes](recipes.md) for worked examples of RASCI and
  multi-subspace ORMAS configurations.
- See [PySCF CASCI Usage](../integration/pyscf_casci.md) for the full
  integration reference.
