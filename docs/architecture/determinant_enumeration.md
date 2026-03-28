# Determinant Enumeration Algorithm

## Problem Statement

Given K subspaces with orbital assignments and occupation constraints,
enumerate all Slater determinants where the total (alpha + beta) electron
count in each subspace falls within its [min, max] bounds, and the global
electron count matches (n_alpha, n_beta).

## Three-Stage Algorithm

### Stage 1: Enumerate Valid Occupation Distributions

An occupation distribution specifies how many alpha and beta electrons
are in each subspace. For K subspaces, it's a pair of K-tuples:

    alpha_dist = (n_alpha_1, n_alpha_2, ..., n_alpha_K)
    beta_dist  = (n_beta_1,  n_beta_2,  ..., n_beta_K)

A distribution is valid if:
- sum(alpha_dist) == n_alpha_total
- sum(beta_dist) == n_beta_total
- For each subspace k: 0 <= n_alpha_k <= n_orbitals_k
- For each subspace k: 0 <= n_beta_k <= n_orbitals_k
- For each subspace k: min_k <= n_alpha_k + n_beta_k <= max_k

This is solved by recursive enumeration over subspaces. At each level,
we try all valid (n_alpha_k, n_beta_k) pairs for the current subspace
and recurse with updated remaining electron counts.

**Pruning:** Before recursing, we check if the remaining electrons can
feasibly be distributed among the remaining subspaces. This prunes large
branches of the search tree early:
- Are there enough orbitals in remaining subspaces to hold the remaining electrons?
- Can the remaining subspaces satisfy their minimum occupation constraints
  with the remaining electrons?
- Would the remaining electrons exceed the maximum occupation of remaining
  subspaces?

### Stage 2: Generate Per-Subspace Strings

For each valid distribution, we generate all occupation strings within
each subspace. If subspace k has m_k orbitals and the distribution assigns
n_alpha_k alpha electrons to it, we generate all C(m_k, n_alpha_k) alpha
strings for that subspace. Likewise for beta.

Strings are generated in a subspace-local basis (orbital indices 0 through
m_k - 1) using itertools.combinations, then mapped to the full active
space basis by `_subspace_strings_to_full()`.

### Stage 3: Combine Across Subspaces and Form Determinant Pairs

The full alpha string for a determinant is the bitwise OR of all subspace
alpha strings (the subspace orbital indices are non-overlapping, so OR
is equivalent to concatenation in the bit representation).

For each valid distribution, the full alpha strings are the Cartesian
product of the per-subspace alpha strings (combined via OR). Likewise
for beta. The determinants for that distribution are all (alpha, beta)
pairs from the resulting combined string lists.

The final determinant list is the union over all valid distributions.

## Example

Active space: 4 orbitals, 4 electrons (2 alpha, 2 beta).
Two subspaces: A = orbitals [0,1], B = orbitals [2,3].
Constraints: A has 1-3 electrons, B has 1-3 electrons.

**Stage 1: Valid distributions**

Try all (n_alpha_A, n_beta_A, n_alpha_B, n_beta_B) with:
- n_alpha_A + n_alpha_B = 2
- n_beta_A + n_beta_B = 2
- 1 <= n_alpha_A + n_beta_A <= 3
- 1 <= n_alpha_B + n_beta_B <= 3

Valid distributions include (1,0,1,2), (1,1,1,1), (0,1,2,1), etc.
Excluded: (2,2,0,0) because B has 0 electrons (< min=1).
Excluded: (0,0,2,2) because A has 0 electrons (< min=1).

**Stage 2: Per-subspace strings**

For distribution (1,1,1,1):
- A alpha: C(2,1) = 2 strings: [0b01, 0b10]
- A beta: C(2,1) = 2 strings: [0b01, 0b10]
- B alpha: C(2,1) = 2 strings: [0b01, 0b10]
- B beta: C(2,1) = 2 strings: [0b01, 0b10]

Map to full space (A orbitals [0,1], B orbitals [2,3]):
- A alpha [0b01, 0b10] -> full [0b0001, 0b0010]
- B alpha [0b01, 0b10] -> full [0b0100, 0b1000]

**Stage 3: Combine**

Full alpha strings: 0b0001|0b0100=0b0101, 0b0001|0b1000=0b1001,
0b0010|0b0100=0b0110, 0b0010|0b1000=0b1010
(4 combined alpha strings)

Full beta strings: same 4 combinations.

Determinants for this distribution: 4 * 4 = 16 (alpha, beta) pairs.

Repeat for all valid distributions and take the union.

## Complexity

The cost is dominated by the number of valid distributions times the
product of per-subspace string counts. For K subspaces with m_k orbitals
each, the number of distributions grows polynomially with K and the
electron count. The per-subspace string counts are C(m_k, n_k), which
are individually manageable since each subspace is small.

The total determinant count can be computed without enumeration via
`count_determinants()`, which uses the same distribution enumeration
but replaces the string generation with combinatorial counting.
