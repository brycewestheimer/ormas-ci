# Phase Factor Computation

## Why Phase Factors Exist

Slater determinants are antisymmetric: swapping two electrons changes
the sign of the wavefunction. When computing Hamiltonian matrix elements,
we need to account for this sign change when comparing two determinants
that differ by one or two orbital occupations.

The sign arises from reordering creation operators to bring one
determinant into alignment with another. Each swap of adjacent operators
picks up a factor of -1.

## Single Excitation Phase

For a single excitation from orbital q to orbital p in the same spin
channel, the phase is:

    sign = (-1)^n

where n is the number of occupied orbitals strictly between positions
min(p,q) and max(p,q) in the occupation string.

Implementation in `utils.py`:

```python
def compute_phase(string, p, q):
    if p == q:
        return 1
    lo, hi = min(p, q), max(p, q)
    mask = ((1 << hi) - 1) & ~((1 << (lo + 1)) - 1)
    n_between = popcount(string & mask)
    return 1 if n_between % 2 == 0 else -1
```

The mask isolates the bits strictly between lo and hi. ANDing with the
occupation string counts how many of those orbitals are occupied.

### Example

Occupation string: 0b10110 (orbitals 1, 2, 4 occupied).
Excitation: orbital 1 -> orbital 4.

Orbitals strictly between 1 and 4: orbital 2 and orbital 3.
Orbital 2 is occupied (bit 2 = 1). Orbital 3 is empty (bit 3 = 0).
n_between = 1 (odd), so sign = -1.

## Double Excitation Phase: The Intermediate Determinant

This is the highest-risk area for bugs. For a same-spin double
excitation (p,q -> r,s), the phase is NOT simply the product of
two independent single-excitation phases on the original string.

The correct procedure is sequential:

1. Compute the phase for the first excitation (p -> r) on the
   original occupation string.
2. Apply the first excitation to get the intermediate string:
   `string_mid = (string ^ (1 << p)) | (1 << r)`
3. Compute the phase for the second excitation (q -> s) on the
   intermediate string.
4. The total phase is the product: sign1 * sign2.

Implementation in `slater_condon.py`:

```python
sign1 = compute_phase(alpha_i, p, r)
alpha_mid = (alpha_i ^ (1 << p)) | (1 << r)
sign2 = compute_phase(alpha_mid, q, s)
sign = sign1 * sign2
```

### Why the Intermediate Matters

The second excitation (q -> s) sees a different occupation pattern
than the original because the first excitation already moved an
electron from p to r. If r is between q and s, the occupation between
q and s has changed, and the phase is different than if computed on
the original string.

Example:
- Original: 0b00111 (orbitals 0, 1, 2 occupied)
- Double: (0 -> 3) and (1 -> 4)
- First: 0 -> 3 on 0b00111. Between 0 and 3: orbitals 1,2 both occupied. n=2, sign1=+1.
- Intermediate: 0b01110 (orbitals 1, 2, 3 occupied)
- Second: 1 -> 4 on 0b01110. Between 1 and 4: orbitals 2,3 both occupied. n=2, sign2=+1.
- Total: +1 * +1 = +1.

If we had incorrectly computed both phases on the original string:
- Second: 1 -> 4 on 0b00111. Between 1 and 4: orbital 2 occupied. n=1, sign2=-1.
- WRONG total: +1 * -1 = -1.

## Opposite-Spin Double Excitation Phase

For opposite-spin doubles (alpha p -> r, beta q -> s), the two phases
are computed on separate strings and are independent:

```python
sign_a = compute_phase(alpha_i, p, r)
sign_b = compute_phase(beta_i, q, s)
sign = sign_a * sign_b
```

No intermediate determinant is needed because the alpha and beta
excitations operate on different occupation strings.

## Testing Strategy

Phase factor correctness is verified by building the full CI
Hamiltonian for small systems (H2, H2O in minimal basis) and comparing
eigenvalues against PySCF's FCI solver. If the phase factors are wrong,
the eigenvalues will not match.

The critical test is in `test_slater_condon.py::test_h2_full_hamiltonian_matches_pyscf`.
This test must pass before any other module can be trusted.
