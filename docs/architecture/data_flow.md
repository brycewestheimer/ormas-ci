# Data Flow

## End-to-End Pipeline

The data flows through three layers: the external framework (PySCF or
QDK/Chemistry), our solver, and back.

```
EXTERNAL FRAMEWORK                    OUR SOLVER
==================                    ==========

1. User defines molecule          
   and basis set                  
        |                         
2. HF calculation                 
   (orbital coefficients)         
        |                         
3. Active space selection         
   (ncas, nelecas)                
        |                         
4. Integral transformation        
   AO -> active-space MO          
   Produces h1e, h2e, ecore       
        |                         
        +--- h1e, h2e, ecore ---> 5. fcisolver.kernel()
             ncas, nelecas             |
                                       +-> ORMASConfig (from user)
                                       |
                                       +-> build_determinant_list(config)
                                       |   Returns: alpha_strings, beta_strings
                                       |
                                       +-> build_ci_hamiltonian(strings, h1e, h2e)
                                       |   Returns: H matrix (n_det x n_det)
                                       |
                                       +-> solve_ci(H, nroots)
                                       |   Returns: energies, ci_vectors
                                       |
        +--- e_tot, ci_vector <--- 6. Return e_ci + ecore, ci_vec
        |
7. mc.analyze()
   Calls fcisolver.make_rdm1()
        |
        +--- ci_vector ---------> 8. make_rdm1(ci_vec, strings, ncas)
                                       Returns: rdm1 (ncas x ncas)
        +--- rdm1 <------------- 
        |
9. Natural orbital analysis
   Occupation numbers
   Population analysis
```

## Data Types at Each Stage

**h1e:** numpy float64 array, shape (ncas, ncas). One-electron integrals
in the active-space MO basis. Symmetric.

**h2e:** numpy float64 array. PySCF may pass this in two formats:
- Full 4-index: shape (ncas, ncas, ncas, ncas). Chemists' notation (pq|rs).
- Compressed triangular: shape (ncas*(ncas+1)/2, ncas*(ncas+1)/2).
  The fcisolver handles both by detecting the dimensionality and calling
  ao2mo.restore(1, eri, ncas) to convert to full 4-index if needed.

**ecore:** Python float. The core energy: nuclear repulsion energy plus
the energy of inactive (frozen) electrons. Added to the CI eigenvalue
to give the total molecular energy.

**alpha_strings, beta_strings:** numpy int64 arrays, shape (n_det,).
Paired occupation bitstrings. Determinant i is the pair
(alpha_strings[i], beta_strings[i]).

**H:** numpy float64 array (n_det, n_det) or scipy.sparse.csr_matrix.
The CI Hamiltonian matrix. Real symmetric.

**energies:** numpy float64 array, shape (nroots,). CI eigenvalues,
sorted ascending.

**ci_vectors:** numpy float64 array, shape (n_det, nroots). CI
eigenvectors. Column k is the coefficient vector for root k.

**rdm1:** numpy float64 array, shape (ncas, ncas). One-particle reduced
density matrix. Symmetric. Trace equals n_alpha + n_beta.

## Where PySCF's Data Ends and Ours Begins

The boundary is at `fcisolver.kernel()`. Everything before that call
(molecule setup, HF, integral transformation) is PySCF's responsibility.
Everything inside that call (determinant enumeration, Hamiltonian
construction, diagonalization) is ours.

PySCF passes us the active-space integrals (already transformed from the
AO basis) and a core energy. We pass back a total energy and CI vector.
PySCF never sees our determinant lists, Hamiltonian matrix, or subspace
configuration.

The only other contact point is `make_rdm1()`, which PySCF calls during
`mc.analyze()`. It receives the CI vector and returns the RDM in the
active-space MO basis.
