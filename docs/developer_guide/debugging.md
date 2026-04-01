# Debugging

## Enabling Logging

ORMAS-CI uses Python's `logging` module. Enable it to see determinant
counts, solver path selection, and convergence information:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

For more detail (individual Davidson iterations, sigma timings):

```python
logging.basicConfig(level=logging.DEBUG)
```

## PySCF Verbose Levels

PySCF's own output is controlled by `mc.verbose`:

```python
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.verbose = 4   # Detailed CASCI output
mc.verbose = 5   # Very detailed (integrals, orbital info)
```

## Debugging Incorrect Energies

1. **Run unrestricted CASCI reference.** Compare `mc.e_tot` against
   PySCF's native `mcscf.CASCI` with no custom fcisolver.

2. **Check with unrestricted ORMAS.** Use
   `ORMASConfig.unrestricted(ncas, nelecas)` to create a single-subspace
   config with no restrictions. This must reproduce PySCF CASCI exactly.
   If it doesn't, there's a bug in the solver.

3. **Verify determinant list.** Check that the restricted determinant
   count matches expectations:
   ```python
   from pyscf.ormas_ci import count_determinants
   print(count_determinants(config))
   ```

4. **Check config consistency.** Ensure `config.n_active_orbitals`
   matches `ncas` and `config.nelecas` matches what you pass to CASCI.

## Debugging Slow Calculations

1. **Check determinant count.** Large determinant counts mean slow
   calculations regardless of the solver path.

2. **Identify the solver path.** With `logging.INFO` enabled, the
   solver logs which path it selected (dense, Davidson+einsum, or
   SCI fallback).

3. **Profile.** Use Python's built-in profiler:
   ```bash
   python -m cProfile -s cumtime your_script.py
   ```
   Look for time spent in `kernel()`, `_sigma()`, `contract_2e`, or
   `_ci_1d_to_2d` / `_ci_2d_to_1d`.

## CI Vector Format Issues

ORMAS-CI stores CI vectors as 1D arrays. PySCF's `selected_ci` backend
expects 2D arrays indexed by `(alpha_string_idx, beta_string_idx)`.
The conversion is handled by `_ci_1d_to_2d()` and `_ci_2d_to_1d()` in
`fcisolver.py`.

If you see shape mismatches or unexpected zero CI vectors, check:
- The SCI mapping built by `_build_sci_mapping()`
- That `alpha_strings` and `beta_strings` arrays match the CI vector
  length
- That string indices are consistent between the ORMAS and SCI
  representations

## SF-ORMAS Debugging

For spin-flip calculations:

1. **Validate the reference.** Call
   `validate_reference_consistency(sf_config)` to check that the
   reference state specification is self-consistent.

2. **Check nelecas_target.** Ensure `sf_config.nelecas_target` matches
   the `nelecas` passed to CASCI:
   ```python
   n_a, n_b = sf_config.nelecas_target
   mc = mcscf.CASCI(mf, ncas, (n_a, n_b))  # must match
   ```

3. **Verify ROHF convergence.** SF-ORMAS requires a converged ROHF
   reference. Check `mf.converged` after `mf.run()`.

## Common Pitfalls

**Orbital index confusion.** Subspace orbital indices are relative to
the active space (0 to ncas-1), not the full MO space. See
[Subspace Model](../design/subspace_model.md).

**Compressed ERI format.** PySCF passes 2-electron integrals in a
compressed format. `fcisolver.py` restores them to full 4-index arrays
using `ao2mo.restore(1, eri, norb)`. If you're writing code that
handles ERIs directly, make sure you're working with the correct format.

**nelecas as integer vs tuple.** PySCF sometimes passes `nelecas` as a
single integer (total electrons). ORMAS-CI normalizes this to
`(n_alpha, n_beta)` via `_normalize_nelecas()` in `fcisolver.py`.

**numpy int64 vs Python int.** Bit operations on numpy int64 can behave
unexpectedly (especially bitwise NOT). The codebase converts to Python
int at the boundary of `slater_condon.matrix_element()` via explicit
`int()` calls. See [Bit String Representation](../architecture/bitstrings.md).
