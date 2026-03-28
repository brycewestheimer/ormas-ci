# Choosing Subspaces

## General Principles

Good subspace definitions reflect the physical correlation structure of
the molecule. Orbitals that are strongly correlated with each other
should be in the same subspace. Orbitals that interact weakly across
groups can be in separate subspaces with limited inter-subspace excitation.

There is no single "correct" partitioning. The right choice depends on
the system, the property you're computing, and the accuracy you need.
Start with a chemically motivated partition, compare against full CASCI
on a small basis set, and adjust.

## Common Partitioning Strategies

### By Orbital Character (Metal + Ligand)

For transition metal complexes, the most natural partition separates
metal d-orbitals from ligand orbitals:

```python
config = ORMASConfig(
    subspaces=[
        Subspace("metal_3d", [0,1,2,3,4], min_electrons=4, max_electrons=8),
        Subspace("ligand_pi", [5,6,7,8,9], min_electrons=2, max_electrons=6),
    ],
    n_active_orbitals=10,
    nelecas=(5, 5),
)
```

The constraints allow moderate charge transfer (1-2 electrons) between
metal and ligand but exclude extreme charge states.

### By Spatial Locality

For extended systems (polymers, clusters), partition by spatial region:

```python
config = ORMASConfig(
    subspaces=[
        Subspace("site_A", [0,1,2], min_electrons=2, max_electrons=4),
        Subspace("bridge", [3,4], min_electrons=1, max_electrons=3),
        Subspace("site_B", [5,6,7], min_electrons=2, max_electrons=4),
    ],
    n_active_orbitals=8,
    nelecas=(4, 4),
)
```

### By Bonding Type (Sigma/Pi Separation)

For conjugated systems or multiple bond breaking:

```python
config = ORMASConfig(
    subspaces=[
        Subspace("sigma", [0,1], min_electrons=1, max_electrons=3),
        Subspace("pi", [2,3,4,5], min_electrons=2, max_electrons=6),
        Subspace("sigma_star", [6,7], min_electrons=0, max_electrons=2),
    ],
    n_active_orbitals=8,
    nelecas=(4, 4),
)
```

### RAS: Core / Valence / Virtual

The classic RAS partitioning:

```python
from pyscf.ormas_ci import RASConfig

ras = RASConfig(
    ras1_orbitals=[0, 1, 2],       # Mostly occupied
    ras2_orbitals=[3, 4, 5, 6],    # Full CI here
    ras3_orbitals=[7, 8, 9],       # Mostly empty
    max_holes_ras1=2,
    max_particles_ras3=2,
    nelecas=(5, 5),
)
config = ras.to_ormas_config()
```

## How to Set min/max Constraints

### Conservative (wider bounds)

Allow more inter-subspace excitations. Keeps more determinants, gives
better accuracy, but less reduction.

Rule of thumb: allow 2-3 electrons of deviation from the "natural"
occupation of each subspace.

### Aggressive (tighter bounds)

Allow fewer inter-subspace excitations. Removes more determinants,
gives larger reduction, but may miss important configurations.

Rule of thumb: allow only 1 electron of deviation.

### Testing Sensitivity

Run calculations with progressively tighter constraints and monitor
the energy:

```python
for allowance in [3, 2, 1, 0]:
    config = ORMASConfig(
        subspaces=[
            Subspace("A", [0,1,2], 3-allowance, 3+allowance),
            Subspace("B", [3,4,5], 3-allowance, 3+allowance),
        ],
        n_active_orbitals=6,
        nelecas=(3, 3),
    )
    mc = mcscf.CASCI(mf, 6, (3,3))
    mc.fcisolver = ORMASFCISolver(config)
    e = mc.kernel()[0]
    n = count_determinants(config)
    print(f"allowance={allowance}: E={e:.8f}, n_det={n}")
```

If the energy changes substantially when tightening from allowance=2
to allowance=1, the configurations you're excluding matter.

## Using AVAS or Natural Orbital Information

If you have access to AVAS (Atomic Valence Active Space) orbital
analysis (available in PySCF and QDK/Chemistry), the atomic character
of each active orbital tells you which subspace it belongs to. Orbitals
with >50% metal d character go in the metal subspace; orbitals with
>50% ligand character go in the ligand subspace.

Similarly, natural orbital occupations from a preliminary CASCI or
ASCI calculation indicate which orbitals are strongly correlated
(occupations far from 0 or 2). These should go in the most flexible
subspace.

## When Not to Use ORMAS

If the correlation structure doesn't have clear subspace boundaries
(all orbitals are entangled with all others equally), ORMAS won't
help. In that case, selected CI (ASCI/CIPSI) or a different method
is more appropriate.

ORMAS is most effective when the system has identifiable correlation
domains that interact moderately.
