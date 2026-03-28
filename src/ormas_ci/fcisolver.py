"""
PySCF-compatible FCI solver implementing ORMAS-CI.

This module provides ORMASFCISolver, a drop-in replacement for PySCF's
default FCI solver inside CASCI calculations. It restricts the CI expansion
to determinants satisfying ORMAS occupation constraints.

Integration design:
    The subspace configuration lives entirely on the ORMASFCISolver object.
    PySCF's CASCI framework does not need to be modified. It handles:
        - Orbital setup and selection
        - Integral transformation from AO to active-space MO basis
        - Core energy computation
        - Output formatting and analysis (via make_rdm1)

    Our solver handles:
        - Restricted determinant enumeration based on the ORMASConfig
        - CI Hamiltonian construction in the restricted basis
        - Diagonalization
        - RDM construction in the restricted basis

    The only connection point is mc.fcisolver: set it to an ORMASFCISolver
    instance, and PySCF's CASCI calls our kernel/make_rdm1 methods through
    its normal workflow.

    No modifications to PySCF source code are needed.
    No modifications to QDK/Chemistry are needed.
"""

import logging

import numpy as np
from pyscf import ao2mo, lib
from pyscf.fci import direct_spin1 as _direct_spin1
from pyscf.fci import selected_ci as _selected_ci
from scipy.sparse.linalg import LinearOperator, eigsh

from ormas_ci.davidson import davidson
from ormas_ci.determinants import (
    build_determinant_list,
    casci_determinant_count,
)
from ormas_ci.hamiltonian import build_ci_hamiltonian
from ormas_ci.rdm import _compute_s_minus_s_plus
from ormas_ci.rdm import make_rdm1 as _make_rdm1
from ormas_ci.rdm import make_rdm1s as _make_rdm1s
from ormas_ci.rdm import make_rdm12 as _make_rdm12
from ormas_ci.rdm import make_rdm12s as _make_rdm12s
from ormas_ci.sigma import SigmaEinsum
from ormas_ci.solver import solve_ci
from ormas_ci.subspaces import ORMASConfig
from ormas_ci.utils import bits_to_indices, popcount

__all__ = ["ORMASFCISolver"]

logger = logging.getLogger(__name__)


def _make_hdiag_vectorized(
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
) -> np.ndarray:
    """Compute diagonal Hamiltonian elements for all determinants at once.

    Precomputes Coulomb (J) and exchange (K) diagonal arrays once, then
    uses numpy advanced indexing to sum contributions for each determinant.
    """
    n_det = len(alpha_strings)
    norb = h1e.shape[0]

    # Precompute J and K arrays: jdiag[p,q] = (pp|qq), kdiag[p,q] = (pq|qp)
    jdiag = np.zeros((norb, norb))
    kdiag = np.zeros((norb, norb))
    for p in range(norb):
        for q in range(norb):
            jdiag[p, q] = h2e[p, p, q, q]
            kdiag[p, q] = h2e[p, q, q, p]
    h1e_diag = np.diag(h1e)

    hdiag = np.empty(n_det)
    for i in range(n_det):
        occ_a = np.array(bits_to_indices(int(alpha_strings[i])))
        occ_b = np.array(bits_to_indices(int(beta_strings[i])))

        # One-electron: sum h1e[p,p] for all occupied orbitals
        energy = h1e_diag[occ_a].sum() + h1e_diag[occ_b].sum()

        # Alpha-alpha J-K (upper triangle only)
        if len(occ_a) > 1:
            jk_aa = jdiag[np.ix_(occ_a, occ_a)] - kdiag[np.ix_(occ_a, occ_a)]
            energy += np.triu(jk_aa, k=1).sum()

        # Beta-beta J-K (upper triangle only)
        if len(occ_b) > 1:
            jk_bb = jdiag[np.ix_(occ_b, occ_b)] - kdiag[np.ix_(occ_b, occ_b)]
            energy += np.triu(jk_bb, k=1).sum()

        # Alpha-beta Coulomb only
        if len(occ_a) > 0 and len(occ_b) > 0:
            energy += jdiag[np.ix_(occ_a, occ_b)].sum()

        hdiag[i] = energy

    return hdiag


def _transform_ci(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
    u: np.ndarray,
) -> np.ndarray:
    """Transform CI vector under orbital rotation.

    Uses the determinant-minor approach: for each pair of determinants
    (I, J), the transformation coefficient is the product of minors
    det(U[occ_alpha_I, occ_alpha_J]) * det(U[occ_beta_I, occ_beta_J]).

    The implementation factorizes the transformation by precomputing
    separate alpha and beta transformation matrices over the unique
    occupation strings, avoiding redundant minor evaluations.

    Args:
        ci_vector: CI coefficients, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.
        u: Orbital rotation matrix, shape (ncas, ncas). If a list/tuple
            of two matrices, u[0] is alpha rotation and u[1] is beta.

    Returns:
        Transformed CI vector, shape (n_det,).
    """
    # Handle u as single matrix or (u_alpha, u_beta) pair
    if isinstance(u, np.ndarray) and u.ndim == 2:
        ua = ub = u
    else:
        ua, ub = u

    # Get unique alpha and beta strings
    unique_alpha = np.unique(alpha_strings)
    unique_beta = np.unique(beta_strings)

    # Precompute alpha transformation matrix (n_unique_alpha x n_unique_alpha)
    n_ua = len(unique_alpha)
    t_alpha = np.zeros((n_ua, n_ua))
    for i in range(n_ua):
        occ_i = bits_to_indices(int(unique_alpha[i]))
        for j in range(n_ua):
            occ_j = bits_to_indices(int(unique_alpha[j]))
            minor = ua[np.ix_(occ_i, occ_j)]
            t_alpha[i, j] = np.linalg.det(minor)

    # Precompute beta transformation matrix (n_unique_beta x n_unique_beta)
    n_ub = len(unique_beta)
    t_beta = np.zeros((n_ub, n_ub))
    for i in range(n_ub):
        occ_i = bits_to_indices(int(unique_beta[i]))
        for j in range(n_ub):
            occ_j = bits_to_indices(int(unique_beta[j]))
            minor = ub[np.ix_(occ_i, occ_j)]
            t_beta[i, j] = np.linalg.det(minor)

    # Create mapping from string value to unique index
    alpha_to_idx = {int(s): i for i, s in enumerate(unique_alpha)}
    beta_to_idx = {int(s): i for i, s in enumerate(unique_beta)}

    # Map each determinant to its (alpha_idx, beta_idx) pair
    det_alpha_idx = np.array([alpha_to_idx[int(s)] for s in alpha_strings])
    det_beta_idx = np.array([beta_to_idx[int(s)] for s in beta_strings])

    # Compute transformed CI vector
    n_det = len(ci_vector)
    new_ci = np.zeros(n_det)
    for k in range(n_det):  # new determinant
        ia = det_alpha_idx[k]
        ib = det_beta_idx[k]
        for ll in range(n_det):  # old determinant
            ja = det_alpha_idx[ll]
            jb = det_beta_idx[ll]
            new_ci[k] += t_alpha[ia, ja] * t_beta[ib, jb] * ci_vector[ll]
    return new_ci


class ORMASFCISolver(lib.StreamObject):
    """ORMAS-CI solver compatible with PySCF's CASCI fcisolver interface.

    Inherits from ``pyscf.lib.StreamObject`` to provide ``copy()``,
    ``stdout``, ``verbose``, and other standard PySCF object behaviours.

    Usage:
        from pyscf import gto, scf, mcscf
        from ormas_ci import ORMASFCISolver, ORMASConfig, Subspace

        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                     basis='ccpvdz')
        mf = scf.RHF(mol).run()

        # Standard CASCI setup
        mc = mcscf.CASCI(mf, ncas=6, nelecas=(4, 4))

        # Define ORMAS subspaces
        config = ORMASConfig(
            subspaces=[
                Subspace("sigma", [0, 1, 2], min_electrons=2, max_electrons=6),
                Subspace("pi", [3, 4, 5], min_electrons=2, max_electrons=6),
            ],
            n_active_orbitals=6,
            nelecas=(4, 4),
        )

        # Replace the solver (one line) and run as normal
        mc.fcisolver = ORMASFCISolver(config)
        mc.kernel()

    The ORMASConfig carries the subspace information. PySCF never sees it.
    PySCF calls kernel() with the active-space integrals and gets back an
    energy and CI vector, same as with any other fcisolver.
    """

    def __init__(self, config: ORMASConfig):
        """Initialize the solver with an ORMAS configuration.

        Args:
            config: ORMASConfig defining the subspace partitioning.
                Must be consistent with the CASCI active space size
                and electron count.
        """
        self.config = config
        config.validate()

        # PySCF interface attributes (stdout, verbose inherited from StreamObject)
        self.nroots = 1
        self.max_memory = lib.param.MAX_MEMORY
        self.max_cycle = 100
        self.max_space = 12
        self.conv_tol = 1e-12
        self.conv_tol_residual = None
        self.pspace_size = 400
        self.lindep = 1e-14
        self.level_shift = 0.001
        self.davidson_only = False
        self.spin = None
        self.mol = None  # Set by PySCF's CASCI/CASSCF
        self.orbsym = None
        self.wfnsym = None

        # State variables set by kernel()
        self.converged = False
        self.norb = None
        self.nelec = None
        self.eci = None
        self.ci = None

        # Cached data from the last kernel() call, needed for RDM construction
        # and for contract_2e (which needs the original integrals).
        self._alpha_strings = None
        self._beta_strings = None
        self._ci_vector = None
        self._h1e = None
        self._eri = None
        self._h_ci_cached = None

        # PySCF's `selected_ci` module provides C-level routines that
        # operate on arbitrary determinant string sets. We use it for
        # matrix-free sigma vectors and C-level RDMs — not as a Selected
        # CI method (CIPSI/ASCI), but as an efficient computational backend
        # for our ORMAS-restricted determinant space.
        self._sci_unique_alpha = None
        self._sci_unique_beta = None
        self._sci_det_row = None  # Maps ORMAS det index -> SCI 2D row
        self._sci_det_col = None  # Maps ORMAS det index -> SCI 2D col
        self._sci_link_index = None
        self._sci_eri_absorbed = None  # Cached absorbed integrals

        # Threshold for using matrix-free SCI path vs explicit Hamiltonian.
        # Below this, explicit Hamiltonian + dense eigh is faster.
        self.direct_ci_threshold = 200

        # Einsum sigma engine (pure-Python, precomputed intermediates).
        # Used when unique string count per channel is below this threshold.
        # Above this, memory becomes excessive and we fall back to PySCF SCI.
        self._sigma_einsum = None
        self.einsum_string_threshold = 300

    def _normalize_nelecas(self, nelecas):
        """Normalize nelecas to a (n_alpha, n_beta) tuple.

        If nelecas is already a tuple, use it directly. If it is an integer
        (total electron count), use the config's nelecas tuple if available
        (which encodes the correct spin decomposition). Otherwise raise an
        error for open-shell cases where the decomposition is ambiguous.
        """
        if isinstance(nelecas, (int, np.integer)):
            total = int(nelecas)
            # Use the config's nelecas if it matches the total
            cfg_ne = self.config.nelecas
            if isinstance(cfg_ne, tuple) and sum(cfg_ne) == total:
                return cfg_ne
            # Fallback: assume closed-shell (equal split)
            na = total // 2
            nb = total - na
            if na != nb:
                raise ValueError(
                    f"Cannot determine (n_alpha, n_beta) from nelecas={total} "
                    f"(odd electron count). Pass a tuple (n_alpha, n_beta) "
                    f"explicitly for open-shell systems."
                )
            return (na, nb)
        return tuple(nelecas)

    def _restore_eri(self, eri, ncas):
        """Restore ERI to full 4-index tensor if compressed."""
        if eri.ndim != 4:
            return ao2mo.restore(1, eri, ncas)
        return eri

    def _get_or_build_strings(self, norb, nelecas):
        """Return cached determinant strings, or build from config."""
        if self._alpha_strings is not None:
            assert self._beta_strings is not None
            return self._alpha_strings, self._beta_strings
        alpha_strings, beta_strings = build_determinant_list(self.config)
        self._alpha_strings = alpha_strings
        self._beta_strings = beta_strings
        return alpha_strings, beta_strings

    def _get_or_build_hamiltonian(self, h1e, eri, norb):
        """Return cached Hamiltonian matrix, or build and cache it."""
        if self._h_ci_cached is not None:
            return self._h_ci_cached
        eri_4d = self._restore_eri(eri, norb)
        alpha_strings, beta_strings = self._get_or_build_strings(norb, None)
        h_ci = build_ci_hamiltonian(alpha_strings, beta_strings, h1e, eri_4d)
        self._h_ci_cached = h_ci
        return h_ci

    # ------------------------------------------------------------------
    # PySCF `selected_ci` module integration for matrix-free operations.
    #
    # PySCF's `pyscf.fci.selected_ci` module provides C-level routines
    # (sigma vectors, RDMs, diagnostics) that operate on arbitrary sets
    # of determinant strings — unlike `direct_spin1` which only handles
    # the full CAS product space. We use these routines as a fast
    # computational backend for our ORMAS-restricted determinant space.
    # This is NOT Selected CI as a method (CIPSI, ASCI, etc.); ORMAS
    # determinant selection remains entirely constraint-based.
    # ------------------------------------------------------------------

    def _build_sci_mapping(self):
        """Build mapping between ORMAS 1D and PySCF selected_ci 2D formats.

        PySCF's selected_ci routines operate on a 2D (unique_alpha x
        unique_beta) matrix. This method builds the index maps to convert
        between our 1D determinant-pair arrays and that 2D representation.
        """
        assert self._alpha_strings is not None
        assert self._beta_strings is not None
        unique_a = np.unique(self._alpha_strings)
        unique_b = np.unique(self._beta_strings)
        self._sci_unique_alpha = unique_a.astype(np.int64)
        self._sci_unique_beta = unique_b.astype(np.int64)

        a_map = {int(v): i for i, v in enumerate(unique_a)}
        b_map = {int(v): i for i, v in enumerate(unique_b)}
        self._sci_det_row = np.array(
            [a_map[int(s)] for s in self._alpha_strings]
        )
        self._sci_det_col = np.array(
            [b_map[int(s)] for s in self._beta_strings]
        )

    def _build_sci_link_tables(self, ncas, nelecas):
        """Generate PySCF C-level link tables for the determinant string space."""
        ci_strs = (self._sci_unique_alpha, self._sci_unique_beta)
        self._sci_link_index = _selected_ci._all_linkstr_index(
            ci_strs, ncas, nelecas
        )

    def _ci_1d_to_2d(self, ci_1d):
        """Scatter ORMAS 1D CI vector into selected_ci 2D matrix."""
        assert self._sci_unique_alpha is not None
        assert self._sci_unique_beta is not None
        ci_2d = np.zeros(
            (len(self._sci_unique_alpha), len(self._sci_unique_beta))
        )
        ci_2d[self._sci_det_row, self._sci_det_col] = ci_1d
        return ci_2d

    def _ci_2d_to_1d(self, ci_2d):
        """Gather valid positions from selected_ci 2D matrix back to ORMAS 1D."""
        assert self._sci_det_row is not None
        assert self._sci_det_col is not None
        return np.asarray(ci_2d)[self._sci_det_row, self._sci_det_col]

    def _as_sci_vector(self, ci_1d):
        """Convert ORMAS 1D CI vector to PySCF SCIvector."""
        ci_2d = self._ci_1d_to_2d(ci_1d)
        ci_strs = (self._sci_unique_alpha, self._sci_unique_beta)
        return _selected_ci._as_SCIvector(ci_2d, ci_strs)

    def _sigma_sci(self, ci_1d):
        """Compute H @ ci using PySCF's C-level selected_ci sigma vector."""
        civec = self._as_sci_vector(ci_1d)
        sigma_2d = _selected_ci.contract_2e(
            self._sci_eri_absorbed, civec, self.norb, self.nelec,
            link_index=self._sci_link_index,
        )
        return self._ci_2d_to_1d(sigma_2d)

    def _make_hdiag_sci(self, h1e, eri, ncas, nelecas):
        """Compute diagonal H elements using PySCF's C-level selected_ci routine."""
        assert self._sci_unique_alpha is not None
        assert self._sci_unique_beta is not None
        ci_strs = (self._sci_unique_alpha, self._sci_unique_beta)
        hdiag_flat = _selected_ci.make_hdiag(h1e, eri, ci_strs, ncas, nelecas)
        hdiag_2d = hdiag_flat.reshape(
            len(self._sci_unique_alpha), len(self._sci_unique_beta)
        )
        return hdiag_2d[self._sci_det_row, self._sci_det_col]

    @property
    def _use_sci(self):
        """Whether the pyscf.fci.selected_ci backend is available for this solve."""
        return self._sci_link_index is not None

    def _solve_iterative(self, h1e, eri, ncas, nelecas, n_det):
        """Iterative eigensolver for the large-space path.

        Tries the einsum sigma + Davidson path first (pure-Python, faster
        for spaces with < einsum_string_threshold unique strings per
        channel). Falls back to PySCF's selected_ci sigma + ARPACK eigsh
        if the space is too large for dense excitation matrices.
        """
        assert self._sci_unique_alpha is not None
        assert self._sci_unique_beta is not None
        n_ua = len(self._sci_unique_alpha)
        n_ub = len(self._sci_unique_beta)
        use_einsum = (
            max(n_ua, n_ub) <= self.einsum_string_threshold
        )

        if use_einsum:
            return self._solve_davidson_einsum(
                h1e, eri, ncas, nelecas, n_det, n_ua, n_ub,
            )
        else:
            return self._solve_eigsh_sci(
                h1e, eri, ncas, nelecas, n_det,
            )

    def _solve_davidson_einsum(self, h1e, eri, ncas, nelecas, n_det,
                               n_ua, n_ub):
        """Davidson eigensolver with pure-Python einsum sigma vector."""
        logger.info(
            f"  Using Davidson + einsum sigma "
            f"({n_ua} alpha, {n_ub} beta unique strings)"
        )
        self._sigma_einsum = SigmaEinsum(
            self._sci_unique_alpha, self._sci_unique_beta,
            h1e, eri, nelec=nelecas,
        )
        logger.info(
            f"  Einsum memory: {self._sigma_einsum.memory_estimate_mb():.1f} MB"
        )

        # Diagonal elements for preconditioner
        hdiag_1d = self._make_hdiag_sci(h1e, eri, ncas, nelecas)

        # Initial guess from lowest diagonal elements
        if self.nroots == 1:
            v0 = np.zeros(n_det)
            v0[np.argmin(hdiag_1d)] = 1.0
        else:
            idx = np.argsort(hdiag_1d)[:self.nroots]
            v0 = np.zeros((n_det, self.nroots))
            for k, i in enumerate(idx):
                v0[i, k] = 1.0

        sigma_engine = self._sigma_einsum
        assert sigma_engine is not None

        def sigma_1d(ci_1d):
            ci_2d = self._ci_1d_to_2d(ci_1d)
            sigma_2d = sigma_engine.sigma(ci_2d)
            return self._ci_2d_to_1d(sigma_2d)

        def precond(r, e):
            return r / (hdiag_1d - e + self.level_shift)

        energies, ci_vectors = davidson(
            aop=sigma_1d,
            x0=v0,
            precond=precond,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=self.nroots,
            lindep=self.lindep,
        )
        return energies, ci_vectors

    def _solve_eigsh_sci(self, h1e, eri, ncas, nelecas, n_det):
        """ARPACK eigsh with PySCF selected_ci sigma vector (fallback)."""
        logger.info("  Using eigsh + PySCF selected_ci sigma (large space fallback)")
        hdiag_1d = self._make_hdiag_sci(h1e, eri, ncas, nelecas)
        v0 = np.zeros(n_det)
        v0[np.argmin(hdiag_1d)] = 1.0

        h_op = LinearOperator(
            shape=(n_det, n_det),
            matvec=self._sigma_sci,  # pyright: ignore[reportCallIssue]
            dtype=np.float64,
        )
        energies, ci_vectors = eigsh(
            h_op, k=self.nroots, which='SA',
            v0=v0, tol=self.conv_tol,  # type: ignore[arg-type]
        )
        order = np.argsort(energies)
        return energies[order], ci_vectors[:, order]

    def kernel(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
        ci0: np.ndarray | None = None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[float | np.ndarray, np.ndarray | list[np.ndarray]]:
        """Solve the ORMAS-CI eigenvalue problem.

        This method signature matches PySCF's fcisolver.kernel interface.
        PySCF's CASCI calls this after transforming integrals to the active space.

        Args:
            h1e: One-electron integrals in the active space, shape (ncas, ncas).
            eri: Two-electron integrals. PySCF may pass these in compressed
                triangular form (2D) or full 4-index form (4D).
            ncas: Number of active orbitals (from PySCF's CASCI).
            nelecas: (n_alpha, n_beta) or total electron count.
            ci0: Initial CI vector guess (optional, currently unused).
            ecore: Core energy (inactive electrons + nuclear repulsion).
                Added to the CI eigenvalue to give the total energy.

        Returns:
            (e_tot, ci_vector) for nroots=1, or
            (e_tot_array, ci_vector_array) for nroots>1.
        """
        nelecas = self._normalize_nelecas(nelecas)

        # Verify consistency between PySCF's CASCI and our config
        if ncas != self.config.n_active_orbitals:
            raise ValueError(
                f"Active space size mismatch: PySCF CASCI has ncas={ncas}, "
                f"but ORMASConfig has n_active_orbitals="
                f"{self.config.n_active_orbitals}. These must match."
            )

        if nelecas != self.config.nelecas:
            raise ValueError(
                f"Electron count mismatch: PySCF CASCI has nelecas={nelecas}, "
                f"but ORMASConfig has nelecas={self.config.nelecas}. "
                f"These must match."
            )

        eri = self._restore_eri(eri, ncas)

        # Clear caches from any previous kernel() call
        self._h_ci_cached = None
        self._alpha_strings = None
        self._beta_strings = None
        self._sci_unique_alpha = None
        self._sci_unique_beta = None
        self._sci_det_row = None
        self._sci_det_col = None
        self._sci_link_index = None
        self._sci_eri_absorbed = None
        self._sigma_einsum = None

        # Set PySCF state variables early (needed by selected_ci methods)
        self.norb = ncas
        self.nelec = nelecas

        # Cache integrals for contract_2e, which needs the originals
        # rather than the absorbed form produced by absorb_h1e.
        self._h1e = h1e
        self._eri = eri

        # Enumerate restricted determinants
        alpha_strings, beta_strings = self._get_or_build_strings(ncas, nelecas)
        n_det = len(alpha_strings)
        n_casci = casci_determinant_count(ncas, nelecas)

        if n_casci > 0:
            reduction = n_det / n_casci
        else:
            reduction = 1.0

        logger.info(
            f"ORMAS-CI: {n_det} determinants "
            f"(CASCI would have {n_casci}, "
            f"reduction to {reduction:.1%})"
        )
        for sub in self.config.subspaces:
            logger.info(
                f"  Subspace '{sub.name}': {sub.n_orbitals} orbitals, "
                f"electrons in [{sub.min_electrons}, {sub.max_electrons}]"
            )

        # Always build the SCI mapping — needed for RDMs, spin_square, etc.
        self._build_sci_mapping()
        self._build_sci_link_tables(ncas, nelecas)
        self._sci_eri_absorbed = _direct_spin1.absorb_h1e(
            h1e, eri, ncas, sum(nelecas), fac=0.5,  # pyright: ignore[reportArgumentType]
        )

        if n_det > self.direct_ci_threshold:
            if self.nroots >= n_det:
                # Fall back to explicit Hamiltonian for full diagonalization
                h_ci = build_ci_hamiltonian(
                    alpha_strings, beta_strings, h1e, eri
                )
                self._h_ci_cached = h_ci
                energies, ci_vectors = solve_ci(h_ci, n_roots=self.nroots)
            else:
                energies, ci_vectors = self._solve_iterative(
                    h1e, eri, ncas, nelecas, n_det,
                )
        else:
            # Explicit Hamiltonian path for small determinant spaces
            h_ci = build_ci_hamiltonian(alpha_strings, beta_strings, h1e, eri)
            self._h_ci_cached = h_ci
            energies, ci_vectors = solve_ci(h_ci, n_roots=self.nroots)

        self.converged = True

        # Number of roots actually found (may be less than nroots if
        # nroots >= n_det and full diagonalization was used).
        n_roots_found = len(energies)

        if self.nroots == 1:
            e_ci = float(energies[0])
            ci_vec = ci_vectors[:, 0]
            self._ci_vector = ci_vec
            self.eci = e_ci + ecore
            self.ci = ci_vec
            return e_ci + ecore, ci_vec
        else:
            self._ci_vector = ci_vectors
            e_tot = energies + ecore
            ci_list = [ci_vectors[:, i] for i in range(n_roots_found)]
            self.eci = e_tot
            self.ci = ci_list
            return e_tot, ci_list

    def make_rdm1(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
    ) -> np.ndarray:
        """Compute the 1-RDM from a CI vector.

        Matches PySCF's fcisolver.make_rdm1 interface. Called by PySCF's
        CASCI during mc.analyze() and for natural orbital computation.

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: (n_alpha, n_beta) or total electron count (unused,
                present for interface compatibility).

        Returns:
            One-particle RDM, shape (ncas, ncas).
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before make_rdm1(). "
                "The determinant list is not yet available."
            )
        assert self._beta_strings is not None
        if self._use_sci:
            nelecas = self._normalize_nelecas(nelecas)
            civec = self._as_sci_vector(ci_vector)
            return _selected_ci.make_rdm1(civec, ncas, nelecas)
        return _make_rdm1(
            ci_vector, self._alpha_strings, self._beta_strings, ncas
        )

    def make_rdm1s(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spin-separated one-particle density matrices.

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: Electron count as (n_alpha, n_beta) or int.

        Returns:
            Tuple (rdm1a, rdm1b) of shape (ncas, ncas) each.

        Raises:
            RuntimeError: If kernel() has not been called yet.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before make_rdm1s()."
            )
        assert self._beta_strings is not None
        if self._use_sci:
            nelecas = self._normalize_nelecas(nelecas)
            civec = self._as_sci_vector(ci_vector)
            return _selected_ci.make_rdm1s(civec, ncas, nelecas)
        return _make_rdm1s(
            ci_vector, self._alpha_strings, self._beta_strings, ncas
        )

    def make_rdm12(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute spin-traced 1-RDM and 2-RDM from a CI vector.

        Required for orbital optimization (CASSCF). The returned RDMs
        follow PySCF conventions:
            rdm1[p,q] = <a+_p a_q>
            rdm2[p,q,r,s] = <a+_p a+_r a_s a_q>

        Energy from RDMs:
            E = einsum('pq,qp', h1e, rdm1) + 0.5 * einsum('pqrs,pqrs', eri, rdm2)

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: Electron count as (n_alpha, n_beta) or int.

        Returns:
            Tuple (rdm1, rdm2) where rdm1 has shape (ncas, ncas) and
            rdm2 has shape (ncas, ncas, ncas, ncas).

        Raises:
            RuntimeError: If kernel() has not been called yet.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before make_rdm12()."
            )
        assert self._beta_strings is not None
        if self._use_sci:
            nelecas = self._normalize_nelecas(nelecas)
            civec = self._as_sci_vector(ci_vector)
            rdm1 = _selected_ci.make_rdm1(civec, ncas, nelecas)
            rdm2 = _selected_ci.make_rdm2(civec, ncas, nelecas)
            return rdm1, rdm2
        return _make_rdm12(
            ci_vector, self._alpha_strings, self._beta_strings, ncas
        )

    def make_rdm12s(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Spin-separated one- and two-particle density matrices.

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: Electron count as (n_alpha, n_beta) or int.

        Returns:
            Tuple ((rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)).

        Raises:
            RuntimeError: If kernel() has not been called yet.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before make_rdm12s()."
            )
        assert self._beta_strings is not None
        if self._use_sci:
            nelecas = self._normalize_nelecas(nelecas)
            civec = self._as_sci_vector(ci_vector)
            rdm1s = _selected_ci.make_rdm1s(civec, ncas, nelecas)
            rdm2s = _selected_ci.make_rdm2s(civec, ncas, nelecas)
            return rdm1s, rdm2s
        return _make_rdm12s(
            ci_vector, self._alpha_strings, self._beta_strings, ncas
        )

    def spin_square(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
    ) -> tuple[float, float]:
        """Compute the expectation value of the total spin operator S^2.

        Evaluates <S^2> = S_z(S_z + 1) + <S_-S_+> directly from the CI
        vector and determinant bit strings using the fermionic spin-flip
        operator, without requiring the 2-RDM.

        Args:
            ci_vector (np.ndarray): CI coefficient vector from kernel().
            ncas (int): Number of active orbitals.
            nelecas (int | tuple[int, int]): Electron count as
                (n_alpha, n_beta) tuple or total int.

        Returns:
            ss (float): The <S^2> expectation value.
            multip (float): The spin multiplicity 2S+1.

        Raises:
            RuntimeError: If kernel() has not been called yet.

        Examples:
            >>> mc.kernel()
            >>> ss, mult = mc.fcisolver.spin_square(mc.ci, ncas, nelecas)
            >>> print(f"<S^2> = {ss:.4f}, 2S+1 = {mult:.1f}")
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before spin_square(). "
                "The determinant list is not yet available."
            )
        assert self._beta_strings is not None

        if self._use_sci:
            nelecas = self._normalize_nelecas(nelecas)
            civec = self._as_sci_vector(ci_vector)
            return _selected_ci.spin_square(civec, ncas, nelecas)

        if isinstance(nelecas, (int, np.integer)):
            na = int(nelecas) // 2
            nb = int(nelecas) - na
        else:
            na, nb = nelecas

        sz = (na - nb) / 2.0
        s_minus_s_plus = _compute_s_minus_s_plus(
            ci_vector, self._alpha_strings, self._beta_strings, ncas
        )
        ss = sz * (sz + 1) + s_minus_s_plus
        s = np.sqrt(ss + 0.25) - 0.5
        multip = s * 2 + 1
        return ss, multip

    def transform_ci_for_orbital_rotation(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
        u: np.ndarray,
    ) -> np.ndarray:
        """Transform CI coefficients under orbital rotation.

        When orbitals are rotated by unitary matrix U, the CI coefficients
        in the new basis are obtained by applying determinant minors of U
        to each pair of old/new determinants.

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: Electron count (unused, for PySCF interface compatibility).
            u: Orbital rotation matrix, shape (ncas, ncas), or tuple
                (u_alpha, u_beta) for spin-dependent rotations.

        Returns:
            Transformed CI coefficient vector, same shape as input.

        Raises:
            RuntimeError: If kernel() has not been called yet.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before transform_ci_for_orbital_rotation()."
            )
        assert self._beta_strings is not None
        return _transform_ci(
            ci_vector, self._alpha_strings, self._beta_strings, ncas, u
        )

    def large_ci(
        self,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
        tol: float = 0.1,
        return_strs: bool = True,
    ) -> list:
        """Return CI coefficients with magnitude above threshold.

        Identifies the most significant configurations in the CI expansion.

        Args:
            ci_vector: CI coefficient vector from kernel().
            ncas: Number of active orbitals.
            nelecas: Electron count as (n_alpha, n_beta) or int.
            tol: Minimum absolute coefficient value to include.
            return_strs: If True, return binary strings; if False,
                return occupation lists.

        Returns:
            List of (coefficient, alpha_string, beta_string) tuples,
            sorted by descending absolute coefficient value.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before large_ci()."
            )
        assert self._beta_strings is not None
        result = []
        for k in range(len(ci_vector)):
            if abs(ci_vector[k]) > tol:
                ai = int(self._alpha_strings[k])
                bi = int(self._beta_strings[k])
                if return_strs:
                    result.append((ci_vector[k], bin(ai), bin(bi)))
                else:
                    from ormas_ci.utils import bits_to_indices

                    result.append((
                        ci_vector[k],
                        list(bits_to_indices(ai)),
                        list(bits_to_indices(bi)),
                    ))
        # Sort by descending absolute coefficient
        result.sort(key=lambda x: -abs(x[0]))
        if not result:
            # If no coefficient exceeds tol, return the largest one
            k_max = int(np.argmax(np.abs(ci_vector)))
            ai = int(self._alpha_strings[k_max])
            bi = int(self._beta_strings[k_max])
            if return_strs:
                result = [(ci_vector[k_max], bin(ai), bin(bi))]
            else:
                from ormas_ci.utils import bits_to_indices

                result = [(
                    ci_vector[k_max],
                    list(bits_to_indices(ai)),
                    list(bits_to_indices(bi)),
                )]
        return result

    def absorb_h1e(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
        fac: float = 1,
    ) -> np.ndarray:
        """Absorb one-electron integrals into two-electron integrals.

        Modifies the two-electron integrals so that the total Hamiltonian
        can be evaluated using only the modified two-electron operator
        via PySCF's ``E_pq E_rs`` contraction.  This matches the
        implementation in ``pyscf.fci.direct_spin1.absorb_h1e`` exactly.

        CASSCF calls ``absorb_h1e`` with ``fac=0.5`` and then passes
        the result to ``contract_2e``.

        Args:
            h1e (np.ndarray): One-electron integrals, shape (ncas, ncas).
            eri (np.ndarray): Two-electron integrals, shape
                (ncas, ncas, ncas, ncas) or compressed.
            ncas (int): Number of active orbitals.
            nelecas (int | tuple[int, int]): Electron count.
            fac (float): Scaling factor applied to the final result.

        Returns:
            np.ndarray: Modified two-electron integrals with h1e absorbed,
            in 4-fold symmetric compressed form.
        """
        if not isinstance(nelecas, (int, np.integer)):
            nelecas = sum(nelecas)
        h2e = ao2mo.restore(1, eri.copy(), ncas)
        f1e = h1e - np.einsum('jiik->jk', h2e) * 0.5
        f1e = f1e * (1.0 / (nelecas + 1e-100))
        for k in range(ncas):
            h2e[k, k, :, :] += f1e
            h2e[:, :, k, k] += f1e
        return ao2mo.restore(4, h2e, ncas) * fac

    def contract_2e(
        self,
        eri: np.ndarray,
        ci_vector: np.ndarray,
        ncas: int,
        nelecas: int | tuple[int, int],
        link_index: None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the Hamiltonian-vector product ``H @ ci_vector``.

        Uses the cached CI Hamiltonian built from the original integrals
        stored during the last ``kernel()`` call.  The ``eri`` argument
        (typically the output of ``absorb_h1e``) is intentionally unused.

        Args:
            eri (np.ndarray): Two-electron integrals (ignored; present
                for PySCF interface compatibility).
            ci_vector (np.ndarray): CI coefficient vector, shape (n_det,).
            ncas (int): Number of active orbitals.
            nelecas (int | tuple[int, int]): Electron count.
            link_index: Unused.  Present for PySCF interface compatibility.

        Returns:
            np.ndarray: Sigma vector ``H @ ci_vector``, same shape as
            *ci_vector*.
        """
        if self._alpha_strings is None:
            raise RuntimeError(
                "kernel() must be called before contract_2e()."
            )
        assert self._beta_strings is not None
        if self._use_sci:
            return self._sigma_sci(ci_vector)
        h_ci = self._get_or_build_hamiltonian(self._h1e, self._eri, ncas)
        return h_ci @ ci_vector

    def contract_1e(
        self,
        f1e: np.ndarray,
        ci_vector: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        link_index: None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the 1-electron contribution to H @ ci_vector.

        Builds a 1-electron-only Hamiltonian (h2e=0) and multiplies.

        Args:
            f1e (np.ndarray): One-electron integrals, shape (norb, norb).
            ci_vector (np.ndarray): CI coefficient vector, shape (n_det,).
            norb (int): Number of active orbitals.
            nelec: Electron count.
            link_index: Unused.  Present for PySCF interface compatibility.

        Returns:
            np.ndarray: 1-electron contribution to sigma vector.
        """
        alpha_strings, beta_strings = self._get_or_build_strings(norb, nelec)
        h2e_zero = np.zeros((norb, norb, norb, norb))
        h1_only = build_ci_hamiltonian(alpha_strings, beta_strings, f1e, h2e_zero)
        return h1_only @ ci_vector

    def make_hdiag(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        compress: bool = False,
    ) -> np.ndarray:
        """Compute diagonal elements of the CI Hamiltonian.

        Used by PySCF's Davidson solver as a preconditioner.

        Args:
            h1e (np.ndarray): One-electron integrals, shape (norb, norb).
            eri (np.ndarray): Two-electron integrals (any PySCF format).
            norb (int): Number of active orbitals.
            nelec: Electron count.
            compress (bool): Unused. Present for PySCF compatibility.

        Returns:
            np.ndarray: Diagonal Hamiltonian elements, shape (n_det,).
        """
        if self._use_sci:
            nelec = self._normalize_nelecas(nelec)
            return self._make_hdiag_sci(h1e, eri, norb, nelec)
        h2e = self._restore_eri(eri, norb)
        alpha_strings, beta_strings = self._get_or_build_strings(norb, nelec)
        return _make_hdiag_vectorized(alpha_strings, beta_strings, h1e, h2e)

    def pspace(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        hdiag: np.ndarray | None = None,
        np_: int = 400,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return P-space Hamiltonian subblock and determinant addresses.

        Constructs the Hamiltonian subblock for the ``np_`` lowest-energy
        determinants (by diagonal element).  PySCF uses this to improve
        the Davidson preconditioner.

        Args:
            h1e (np.ndarray): One-electron integrals, shape (norb, norb).
            eri (np.ndarray): Two-electron integrals.
            norb (int): Number of active orbitals.
            nelec: Electron count.
            hdiag (np.ndarray | None): Diagonal elements.  Computed if None.
            np_ (int): Number of lowest-energy determinants to include.

        Returns:
            Tuple (H_sub, addr) where H_sub is the subblock and addr are
            the determinant indices.
        """
        if hdiag is None:
            hdiag = self.make_hdiag(h1e, eri, norb, nelec)

        n_det = len(hdiag)
        np_ = min(np_, n_det)

        addr = np.argsort(hdiag)[:np_]

        h_ci = self._get_or_build_hamiltonian(h1e, eri, norb)
        if hasattr(h_ci, 'toarray'):
            h_ci = h_ci.toarray()  # type: ignore[union-attr]
        h_sub = h_ci[np.ix_(addr, addr)]
        return h_sub, addr

    def get_init_guess(
        self,
        norb: int,
        nelec: int | tuple[int, int],
        nroots: int,
        hdiag: np.ndarray,
    ) -> list[np.ndarray]:
        """Generate initial guess CI vectors from diagonal elements.

        Returns ``nroots`` unit vectors at the positions of the lowest
        diagonal Hamiltonian elements.

        Args:
            norb (int): Number of active orbitals.
            nelec: Electron count.
            nroots (int): Number of states to solve.
            hdiag (np.ndarray): Diagonal Hamiltonian elements.

        Returns:
            list[np.ndarray]: Initial guess CI vectors.
        """
        n_det = len(hdiag)
        addr = np.argsort(hdiag)
        ci0 = []
        for i in range(min(nroots, n_det)):
            c = np.zeros(n_det)
            c[addr[i]] = 1.0
            ci0.append(c)
        return ci0

    def contract_ss(
        self,
        ci_vector: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
    ) -> np.ndarray:
        """Compute S^2 acting on a CI vector: ``S^2 |Psi>``.

        Required by ``pyscf.fci.addons.fix_spin_`` to add a spin penalty.

        Uses: S^2 = S_z(S_z + 1) + S_-S_+, applied as an operator to
        the CI vector in the ORMAS determinant basis.

        Args:
            ci_vector (np.ndarray): CI coefficient vector, shape (n_det,).
            norb (int): Number of active orbitals.
            nelec: Electron count as (n_alpha, n_beta) or int.

        Returns:
            np.ndarray: Result of S^2 applied to ci_vector.
        """
        nelec = self._normalize_nelecas(nelec)
        na, nb = nelec
        alpha_strings, beta_strings = self._get_or_build_strings(norb, nelec)
        n_det = len(alpha_strings)
        sigma = np.zeros(n_det)

        for i in range(n_det):
            ai = int(alpha_strings[i])
            bi = int(beta_strings[i])
            sz_i = (popcount(ai) - popcount(bi)) / 2.0

            for j in range(n_det):
                cj = ci_vector[j]
                if abs(cj) < 1e-15:
                    continue

                aj = int(alpha_strings[j])
                bj = int(beta_strings[j])

                n_diff_a = popcount(ai ^ aj)
                n_diff_b = popcount(bi ^ bj)

                if n_diff_a == 0 and n_diff_b == 0:
                    # Diagonal: S_z(S_z+1) + n_beta_only
                    n_beta_only = popcount(bj & ~aj)
                    sigma[i] += cj * (sz_i * (sz_i + 1) + n_beta_only)

                elif n_diff_a == 2 and n_diff_b == 2:
                    # Off-diagonal S_-S_+ contribution
                    alpha_diff = ai ^ aj
                    beta_diff = bi ^ bj

                    p_alpha = bits_to_indices(alpha_diff & aj)
                    q_alpha = bits_to_indices(alpha_diff & ai)

                    if len(p_alpha) != 1 or len(q_alpha) != 1:
                        continue

                    p = p_alpha[0]
                    q = q_alpha[0]

                    beta_gained_in_i = bits_to_indices(beta_diff & bi)
                    beta_lost_from_i = bits_to_indices(beta_diff & bj)

                    if len(beta_gained_in_i) != 1 or len(beta_lost_from_i) != 1:
                        continue
                    if beta_gained_in_i[0] != p or beta_lost_from_i[0] != q:
                        continue

                    # Phase from applying S_-S_+ operators to |J>
                    phase_1 = popcount(bj & ((1 << q) - 1))
                    new_beta = bj ^ (1 << q)
                    phase_2 = popcount(aj & ((1 << q) - 1))
                    new_alpha = aj | (1 << q)
                    phase_3 = popcount(new_alpha & ((1 << p) - 1))
                    phase_4 = popcount(new_beta & ((1 << p) - 1))
                    total = phase_1 + phase_2 + phase_3 + phase_4
                    phase = 1 if total % 2 == 0 else -1

                    sigma[i] += cj * phase

        return sigma

    def dump_flags(self, verbose: int | None = None) -> None:
        """Print solver configuration to the logger.

        Args:
            verbose: Verbosity level. If None, uses self.verbose.
        """
        log = logging.getLogger(__name__)
        log.info('ORMAS-CI FCI solver')
        log.info('nroots = %d', self.nroots)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('Number of subspaces: %d', self.config.n_subspaces)
        for sub in self.config.subspaces:
            log.info(
                '  %s: orbitals %s, electrons [%d, %d]',
                sub.name,
                sub.orbital_indices,
                sub.min_electrons,
                sub.max_electrons,
            )
