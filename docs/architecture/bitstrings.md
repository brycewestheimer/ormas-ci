# Bit String Representation

## Convention

Slater determinants are represented as pairs of integers: one for the
alpha occupation string and one for the beta occupation string. Bit k
of an occupation string is 1 if orbital k is occupied in that spin channel.

Orbital 0 is the least significant bit.

## Examples

```
String 0b1010 (decimal 10):
    Bit 0 = 0  -> orbital 0 unoccupied
    Bit 1 = 1  -> orbital 1 occupied
    Bit 2 = 0  -> orbital 2 unoccupied
    Bit 3 = 1  -> orbital 3 occupied
    Occupied orbitals: [1, 3]

String 0b0111 (decimal 7):
    Occupied orbitals: [0, 1, 2]

String 0b1111 (decimal 15):
    Occupied orbitals: [0, 1, 2, 3]  (all occupied in 4-orbital space)
```

## Why Integers

Integer representation enables fast bitwise operations for the core
algorithms:

- **Popcount** (count occupied orbitals): `bin(n).count('1')` or the
  Python 3.10+ `int.bit_count()`.
- **XOR** (find differing orbitals): `str_i ^ str_j` gives a mask where
  set bits indicate orbitals that differ.
- **AND** (find holes/particles): `(str_i ^ str_j) & str_i` gives
  orbitals occupied in i but not j (holes for i->j excitation).
- **OR** (combine subspaces): since subspace orbitals are non-overlapping,
  the full-space string is the bitwise OR of subspace strings.

## Subspace-Local vs Full-Space Strings

During determinant enumeration, strings are first generated in a
subspace-local basis (orbital indices 0 to n_orb_k - 1 within the
subspace) and then mapped to the full active space basis.

For example, if subspace A contains orbitals [2, 5, 7] in the full
active space:
- Local string 0b101 means local orbitals 0 and 2 are occupied
- This maps to full-space orbitals 2 and 7
- Full-space string: (1 << 2) | (1 << 7) = 0b10000100 = 132

The mapping is performed by `_subspace_strings_to_full()` in
`determinants.py`. Since subspace orbital indices are non-overlapping,
the full-space strings from different subspaces can be combined by
bitwise OR without conflict.

## Storage

Occupation strings are stored as numpy int64 arrays. Python's arbitrary
precision integers are used inside the Slater-Condon rule evaluation to
avoid edge cases with numpy integer bit operations (numpy integers have
fixed width and can behave unexpectedly with bitwise NOT and shift
operations on negative values).

The conversion from numpy int64 to Python int happens at the boundary
of `matrix_element()` and related functions via explicit `int()` calls.

## Maximum Active Space Size

With int64 representation, the maximum active space is 63 orbitals (bit
63 is the sign bit in signed int64). In practice, active spaces above
~30 orbitals produce determinant counts too large for explicit
enumeration and matrix construction, so the 63-orbital limit is never
the binding constraint.
