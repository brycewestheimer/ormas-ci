# PySCF Integration Strategy

## Design Goal

Plug into PySCF's CASCI framework without modifying PySCF's source code.
The user should be able to use all of PySCF's orbital setup, active space
selection, and analysis tools, with only the CI solver replaced.

## How PySCF's CASCI Works Internally

PySCF's `mcscf.CASCI` class orchestrates a multistep pipeline:

1. **Orbital selection:** Identifies active orbitals from the HF MO set.
2. **Integral transformation:** Transforms AO integrals to the active-space
   MO basis, producing h1e (1-electron) and h2e (2-electron) integrals,
   plus a core energy (ecore) from the inactive electrons and nuclear repulsion.
3. **FCI solver call:** Calls `self.fcisolver.kernel(h1e, eri, ncas, nelecas, ecore=ecore)`.
4. **Post-processing:** Calls `self.fcisolver.make_rdm1()` for natural
   orbital analysis, population analysis, and other properties.

The `fcisolver` attribute is a replaceable object. PySCF ships several
FCI solver implementations (direct_spin0, direct_spin1, etc.), and any
object that implements the `kernel()` and `make_rdm1()` methods with
the correct signatures can be substituted.

## Our Integration Point

`ORMASFCISolver` inherits from ``pyscf.lib.StreamObject`` and implements
the full PySCF FCI solver interface:

```python
class ORMASFCISolver(lib.StreamObject):
    # Core solver
    def kernel(self, h1e, eri, ncas, nelecas, ci0=None, ecore=0, **kwargs): ...

    # Reduced density matrices
    def make_rdm1(self, ci_vector, ncas, nelecas): ...
    def make_rdm1s(self, ci_vector, ncas, nelecas): ...
    def make_rdm12(self, ci_vector, ncas, nelecas): ...
    def make_rdm12s(self, ci_vector, ncas, nelecas): ...

    # Spin and analysis
    def spin_square(self, ci_vector, ncas, nelecas): ...
    def large_ci(self, ci_vector, ncas, nelecas, tol=0.1): ...

    # Hamiltonian operations
    def absorb_h1e(self, h1e, eri, ncas, nelecas, fac=1): ...
    def contract_1e(self, f1e, ci_vector, norb, nelec): ...
    def contract_2e(self, eri, ci_vector, ncas, nelecas): ...
    def contract_ss(self, ci_vector, norb, nelec): ...

    # Davidson solver support
    def make_hdiag(self, h1e, eri, norb, nelec): ...
    def pspace(self, h1e, eri, norb, nelec, hdiag=None, np_=400): ...
    def get_init_guess(self, norb, nelec, nroots, hdiag): ...

    # Orbital rotation
    def transform_ci_for_orbital_rotation(self, ci_vector, ncas, nelecas, u): ...
```

Inheriting from ``lib.StreamObject`` provides ``copy()``, ``stdout``,
``verbose``, and ``view()`` automatically, which is required for
compatibility with PySCF addons like ``pyscf.fci.addons.fix_spin_``.

The user plugs it in with one line:

```python
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = ORMASFCISolver(config)
mc.kernel()
```

## Where Subspace Information Lives

The subspace configuration (which orbitals belong to which subspace,
what the occupation constraints are) lives entirely on the ORMASFCISolver
object as `self.config`. PySCF never sees it.

When PySCF calls `kernel(h1e, eri, ncas, nelecas)`, the solver uses
`self.config` internally to enumerate restricted determinants. The
integrals and active space parameters come from PySCF; the restriction
logic is self-contained.

This means:
- **No PySCF source modifications needed.**
- **No monkey-patching or subclassing of PySCF classes needed.**
- **PySCF version compatibility is broad** (any version that supports
  the fcisolver.kernel interface, which has been stable since PySCF 1.x).

## Format Handling

PySCF has a few format quirks that the solver must handle:

**Compressed ERIs:** PySCF sometimes passes the 2-electron integrals
in a compressed triangular format (2D array) instead of the full 4-index
tensor. The solver detects this via `eri.ndim` and uses `ao2mo.restore()`
to convert. See [integrals.md](../architecture/integrals.md).

**nelecas format:** PySCF may pass `nelecas` as a tuple `(n_alpha, n_beta)`
or as a single integer (total electrons). The solver normalizes to a tuple.

**ecore as keyword:** The core energy may come as a positional or keyword
argument. The solver accepts both via `ecore=0` default.

## Consistency Checks

The solver validates that PySCF's active space parameters match the
ORMASConfig at the start of kernel():

- `ncas` must equal `config.n_active_orbitals`
- `nelecas` must equal `config.nelecas`

Mismatches raise a ValueError with a clear message. This catches the
common mistake of creating an ORMASConfig for one active space and then
changing the CASCI parameters without updating the config.

## What Works Through This Integration

- `mc.kernel()` - energy calculation (CASCI)
- `mc.mc2step()` / `mc.mc1step()` - orbital optimization (CASSCF)
- `mc.analyze()` - natural orbital occupations, population analysis
- `mc.e_tot` - total energy
- `mc.ci` - CI coefficient vector
- `mc.mo_coeff` - MO coefficients (from PySCF, unchanged)
- `mc.get_h1eff()` / `mc.get_h2eff()` - access to transformed integrals
- `mc.fcisolver.make_rdm1()` / `make_rdm12()` - 1- and 2-electron RDMs
- `mc.fcisolver.spin_square()` - S^2 expectation value and multiplicity
- `pyscf.fci.addons.fix_spin_()` - spin penalty addon

## What Doesn't Work (Yet)

- State-averaged CASCI - requires multi-root RDM construction.
- Analytical gradients - require response terms.
