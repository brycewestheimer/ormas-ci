"""
Hamiltonian matrix element evaluation using Slater-Condon rules.

All integrals use chemists' notation: h2e[p,q,r,s] = (pq|rs).
This matches PySCF's internal convention for CASCI active-space integrals.

Phase factor convention: for an excitation from orbital q to orbital p in
a given spin channel, the sign is (-1)^n where n is the number of occupied
orbitals (in that spin channel) strictly between p and q.

WARNING: The phase factor for double excitations requires using the intermediate
determinant (after the first excitation) for the second phase computation.
This is the most common source of bugs in determinant-based CI codes.
"""

import numpy as np

from ormas_ci.utils import bits_to_indices, compute_phase, popcount

__all__ = ["matrix_element", "excitation_info"]


def excitation_info(
    str_i: int, str_j: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Analyze the excitation between two same-spin occupation strings.

    Args:
        str_i: Bra occupation string.
        str_j: Ket occupation string.

    Returns:
        (n_excitations, holes, particles) where:
        - n_excitations: Number of orbitals that differ (= number of holes = number of particles)
        - holes: Orbitals occupied in i but not in j
        - particles: Orbitals occupied in j but not in i
    """
    diff = str_i ^ str_j
    holes = bits_to_indices(diff & str_i)
    particles = bits_to_indices(diff & str_j)
    return len(holes), holes, particles


def _compute_diagonal(
    alpha: int, beta: int,
    h1e: np.ndarray, h2e: np.ndarray,
) -> float:
    """Compute diagonal Hamiltonian element <det|H|det>.

    E = sum_p h[p,p] * n_p
      + sum_{p>q, same spin} (J[p,q] - K[p,q])
      + sum_{p(alpha), q(beta)} J[p,q]

    where J[p,q] = h2e[p,p,q,q] and K[p,q] = h2e[p,q,q,p].
    """
    occ_a = bits_to_indices(alpha)
    occ_b = bits_to_indices(beta)

    energy = 0.0

    # One-electron terms
    for p in occ_a:
        energy += h1e[p, p]
    for p in occ_b:
        energy += h1e[p, p]

    # Alpha-alpha Coulomb and exchange
    for i, p in enumerate(occ_a):
        for q in occ_a[i + 1:]:
            energy += h2e[p, p, q, q] - h2e[p, q, q, p]

    # Beta-beta Coulomb and exchange
    for i, p in enumerate(occ_b):
        for q in occ_b[i + 1:]:
            energy += h2e[p, p, q, q] - h2e[p, q, q, p]

    # Alpha-beta Coulomb only (no exchange for opposite spin)
    for p in occ_a:
        for q in occ_b:
            energy += h2e[p, p, q, q]

    return energy


def _compute_single(
    alpha_i: int, beta_i: int,
    alpha_j: int, beta_j: int,
    h1e: np.ndarray, h2e: np.ndarray,
) -> float:
    """Compute matrix element for a single excitation (one spin-orbital differs).

    For an excitation p -> q in the alpha channel (beta unchanged):
    <I|H|J> = sign * [ h1e[p,q] + sum_r(alpha, r!=p) (J[p,q,r] - K[p,r,q])
                                 + sum_r(beta) J[p,q,r] ]
    where J[p,q,r] = h2e[p,q,r,r] and K[p,r,q] = h2e[p,r,r,q].
    """
    a_ndiff = popcount(alpha_i ^ alpha_j) // 2
    b_ndiff = popcount(beta_i ^ beta_j) // 2

    if a_ndiff == 1 and b_ndiff == 0:
        # Alpha single excitation
        _, holes, particles = excitation_info(alpha_i, alpha_j)
        p = holes[0]   # annihilate from p
        q = particles[0]  # create at q
        sign = compute_phase(alpha_i, p, q)

        element = sign * h1e[p, q]

        # Coulomb-exchange with other alpha electrons
        for r in bits_to_indices(alpha_i):
            if r != p:
                element += sign * (h2e[p, q, r, r] - h2e[p, r, r, q])

        # Coulomb with beta electrons
        for r in bits_to_indices(beta_i):
            element += sign * h2e[p, q, r, r]

        return element

    elif a_ndiff == 0 and b_ndiff == 1:
        # Beta single excitation
        _, holes, particles = excitation_info(beta_i, beta_j)
        p = holes[0]
        q = particles[0]
        sign = compute_phase(beta_i, p, q)

        element = sign * h1e[p, q]

        # Coulomb-exchange with other beta electrons
        for r in bits_to_indices(beta_i):
            if r != p:
                element += sign * (h2e[p, q, r, r] - h2e[p, r, r, q])

        # Coulomb with alpha electrons
        for r in bits_to_indices(alpha_i):
            element += sign * h2e[p, q, r, r]

        return element

    return 0.0


def _compute_double(
    alpha_i: int, beta_i: int,
    alpha_j: int, beta_j: int,
    h1e: np.ndarray, h2e: np.ndarray,
) -> float:
    """Compute matrix element for a double excitation (two spin-orbitals differ).

    Three cases: alpha-alpha, beta-beta, alpha-beta.

    For same-spin doubles, the phase uses the INTERMEDIATE determinant
    for the second excitation.
    """
    a_ndiff = popcount(alpha_i ^ alpha_j) // 2
    b_ndiff = popcount(beta_i ^ beta_j) // 2

    if a_ndiff == 2 and b_ndiff == 0:
        # Alpha-alpha double excitation
        _, holes, particles = excitation_info(alpha_i, alpha_j)
        p, q = holes[0], holes[1]      # annihilate from p and q
        r, s = particles[0], particles[1]  # create at r and s

        # Phase for first excitation: p -> r on original string
        sign1 = compute_phase(alpha_i, p, r)

        # The phase for a same-spin double excitation cannot be computed
        # from the original string alone. After the first excitation p -> r,
        # the occupation changes: bit p is cleared and bit r is set. This
        # affects the count of occupied orbitals between q and s, which
        # determines the second phase factor. Using the original string
        # would give an incorrect sign.
        alpha_mid = (alpha_i ^ (1 << p)) | (1 << r)

        # Phase for second excitation: q -> s on intermediate string
        sign2 = compute_phase(alpha_mid, q, s)

        sign = sign1 * sign2
        # Slater-Condon rule for same-spin double excitation:
        # <I|H|J> = sign * [(pq|rs) - (ps|rq)]  in chemists' notation
        # The exchange term (ps|rq) arises from the antisymmetry of
        # same-spin electrons under particle exchange.
        return sign * (h2e[p, r, q, s] - h2e[p, s, q, r])

    elif a_ndiff == 0 and b_ndiff == 2:
        # Beta-beta double excitation
        _, holes, particles = excitation_info(beta_i, beta_j)
        p, q = holes[0], holes[1]
        r, s = particles[0], particles[1]

        sign1 = compute_phase(beta_i, p, r)
        beta_mid = (beta_i ^ (1 << p)) | (1 << r)
        sign2 = compute_phase(beta_mid, q, s)

        sign = sign1 * sign2
        return sign * (h2e[p, r, q, s] - h2e[p, s, q, r])

    elif a_ndiff == 1 and b_ndiff == 1:
        # Alpha-beta double excitation (opposite spin)
        # Alpha and beta phases are independent (different strings)
        _, a_holes, a_particles = excitation_info(alpha_i, alpha_j)
        _, b_holes, b_particles = excitation_info(beta_i, beta_j)

        p = a_holes[0]      # alpha annihilation
        r = a_particles[0]  # alpha creation
        q = b_holes[0]      # beta annihilation
        s = b_particles[0]  # beta creation

        sign_a = compute_phase(alpha_i, p, r)
        sign_b = compute_phase(beta_i, q, s)

        # Opposite-spin: Coulomb only, no exchange
        return sign_a * sign_b * h2e[p, r, q, s]

    return 0.0


def matrix_element(
    alpha_i: int, beta_i: int,
    alpha_j: int, beta_j: int,
    h1e: np.ndarray, h2e: np.ndarray,
) -> float:
    """Compute <det_i|H|det_j> using Slater-Condon rules.

    This is the top-level dispatch function. It determines the excitation
    level (0, 1, 2, or 3+) and calls the appropriate handler.

    Args:
        alpha_i: Bra alpha occupation string as int.
        beta_i: Bra beta occupation string as int.
        alpha_j: Ket alpha occupation string as int.
        beta_j: Ket beta occupation string as int.
        h1e: One-electron integrals in the active space, shape (ncas, ncas).
        h2e: Two-electron integrals in chemists' notation, shape
            (ncas, ncas, ncas, ncas). h2e[p,q,r,s] = (pq|rs).

    Returns:
        Hamiltonian matrix element as a float.
    """
    # Count spin-orbital differences in each channel
    # Each channel's XOR has 2 bits set per excitation (one hole, one particle)
    n_diff_a = popcount(alpha_i ^ alpha_j) // 2
    n_diff_b = popcount(beta_i ^ beta_j) // 2
    n_excitations = n_diff_a + n_diff_b

    if n_excitations == 0:
        return _compute_diagonal(alpha_i, beta_i, h1e, h2e)
    elif n_excitations == 1:
        return _compute_single(
            alpha_i, beta_i, alpha_j, beta_j, h1e, h2e
        )
    elif n_excitations == 2:
        return _compute_double(
            alpha_i, beta_i, alpha_j, beta_j, h1e, h2e
        )
    else:
        return 0.0
