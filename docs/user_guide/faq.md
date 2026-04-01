# Frequently Asked Questions

## My ORMAS energy is much higher than CASCI

Your occupation constraints are probably too tight, excluding important
configurations. Try widening the `min_electrons` / `max_electrons`
bounds by 1-2 electrons per subspace and compare. See
[Choosing Subspaces](../guides/choosing_subspaces.md) for a systematic
approach to testing constraint sensitivity.

## I get a ValueError about ncas or nelecas mismatch

The `n_active_orbitals` and `nelecas` in your `ORMASConfig` must
exactly match the `ncas` and `nelecas` you pass to PySCF's
`mcscf.CASCI(mf, ncas, nelecas)`. A common mistake is using
`nelecas` as a single integer instead of a `(n_alpha, n_beta)` tuple.
See [PySCF CASCI Usage](../integration/pyscf_casci.md) for the correct
setup pattern.

## My calculation is very slow

Check the determinant count with `count_determinants(config)`. If it
is large (> 5,000), consider tightening constraints or reducing the
active space. See [Performance Tuning](performance_tuning.md) for
strategies.

## Can I use ORMAS-CI with CASSCF?

Yes. Replace `mcscf.CASCI` with `mcscf.CASSCF`:

```python
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
mc.kernel()
```

The solver provides all methods CASSCF needs for orbital optimization
(`make_rdm1`, `make_rdm12`, `contract_2e`, etc.). See
[Common Recipes](../guides/recipes.md) for worked examples.

## What is the difference between ORMAS and RAS?

RAS is a special case of ORMAS with exactly 3 subspaces (RAS1, RAS2,
RAS3) and restrictions on holes in RAS1 and particles in RAS3. ORMAS
generalizes this to an arbitrary number of subspaces with per-subspace
min/max electron bounds. See
[Method Comparison](../theory/comparison.md) for a detailed comparison.

## Does ORMAS-CI support state-averaged CASCI/CASSCF?

Not yet. State-averaged calculations require state-averaged RDM
construction, which is planned for a future release. See the
[Development Roadmap](../design/future.md) for details. Multi-root
calculations (computing multiple roots independently) are fully
supported via `solver.nroots`.

## How do I use ORMAS-CI with QDK/Chemistry?

Install the `qdk` extra (`pip install -e ".[qdk]"`) and see the
[QDK/Chemistry Integration](../integration/qdk_chemistry.md) guide.
The `examples/05_qdk_pipeline.py` script provides a complete working
example.

## Why does SF-ORMAS require ROHF?

Spin-flip ORMAS starts from a high-spin reference state and flips
electron spins to reach the target multiplicity. ROHF (restricted
open-shell Hartree-Fock) provides the correct high-spin reference
orbitals. Using UHF would introduce spin contamination in the reference,
compromising the spin-flip procedure. See
[Spin-Flip Theory](../theory/spinflip.md) for details.

## What are the `selected_ci` messages in the output?

ORMAS-CI uses PySCF's `selected_ci` module as a computational backend
for sigma vector operations and RDM construction. This is an
implementation detail -- it does **not** mean that a Selected CI method
(CIPSI, ASCI) is being used. The determinant selection is entirely
ORMAS constraint-based. See
[Performance Considerations](../design/performance.md) for how and
when the SCI backend is invoked.

## How do I compare ORMAS-CI with Forte?

See the [Forte/Psi4 Comparison](../integration/forte_comparison.md)
page, which covers the differences in scope, integration target, and
performance characteristics between the two implementations.
