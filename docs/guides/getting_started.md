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
    basis='6-31g',
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

## Open-Shell Systems (SF-ORMAS)

For open-shell problems (diradicals, excited states, bond breaking),
SF-ORMAS starts from a high-spin ROHF reference and flips spins to
reach the target multiplicity:

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
mc = mcscf.CASCI(mf, 2, (n_a, n_b))
mc.fcisolver = SFORMASFCISolver(sf_config)
e = mc.kernel()[0]
print(f"SF-ORMAS energy: {e:.10f} Ha")
```

See [Spin-Flip Recipes](sf_recipes.md) for worked examples including
multi-root excited states, CASSCF orbital optimization, and QDK
integration.

## Next Steps

- See [Choosing Subspaces](choosing_subspaces.md) for guidance on how
  to partition your active orbitals.
- See [Common Recipes](recipes.md) for worked examples of RASCI and
  multi-subspace ORMAS configurations.
- See [Spin-Flip Recipes](sf_recipes.md) for open-shell workflows
  using SF-ORMAS.
- See [PySCF CASCI Usage](../integration/pyscf_casci.md) for the full
  integration reference.
