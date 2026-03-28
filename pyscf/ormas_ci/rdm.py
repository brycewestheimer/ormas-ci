"""
Reduced density matrix construction from CI eigenvectors.

Computes the one-particle RDM (and optionally the two-particle RDM) from
the CI coefficient vector and the determinant list.

Index convention (matching PySCF):
    rdm1[p,q] = <Psi| a+_p a_q |Psi>
    rdm2[p,q,r,s] = <Psi| a+_p a+_r a_s a_q |Psi>
"""

import numpy as np

from pyscf.ormas_ci.slater_condon import excitation_info
from pyscf.ormas_ci.utils import bits_to_indices, compute_phase, popcount

__all__ = ["make_rdm1", "make_rdm1s", "make_rdm12", "make_rdm12s"]


def make_rdm1(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> np.ndarray:
    """Compute the one-particle reduced density matrix.

    D1[p,q] = sum_{I,J} c_I c_J <det_I| a+_p a_q |det_J>

    The operator a+_p a_q connects determinants that differ by a single
    excitation from q to p in one spin channel, with the other channel unchanged.

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation strings, shape (n_det,).
        beta_strings: Beta occupation strings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        One-particle RDM, shape (ncas, ncas). Symmetric.
    """
    n_det = len(alpha_strings)
    rdm1 = np.zeros((ncas, ncas), dtype=np.float64)

    for i in range(n_det):
        ci_i = ci_vector[i]
        if abs(ci_i) < 1e-15:
            continue

        ai = int(alpha_strings[i])
        bi = int(beta_strings[i])

        for j in range(n_det):
            ci_j = ci_vector[j]
            ci_ij = ci_i * ci_j
            if abs(ci_ij) < 1e-15:
                continue

            aj = int(alpha_strings[j])
            bj = int(beta_strings[j])

            n_diff_a = popcount(ai ^ aj) // 2
            n_diff_b = popcount(bi ^ bj) // 2

            # Alpha-alpha contribution (requires identical beta strings)
            if bi == bj:
                if n_diff_a == 0:
                    # Diagonal: each occupied alpha orbital contributes
                    for p in bits_to_indices(ai):
                        rdm1[p, p] += ci_ij
                elif n_diff_a == 1:
                    # Single alpha excitation: q -> p
                    holes = bits_to_indices((ai ^ aj) & ai)
                    parts = bits_to_indices((ai ^ aj) & aj)
                    q = holes[0]   # occupied in i, not in j
                    p = parts[0]   # occupied in j, not in i
                    sign = compute_phase(ai, p, q)
                    rdm1[p, q] += sign * ci_ij

            # Beta-beta contribution (requires identical alpha strings)
            if ai == aj:
                if n_diff_b == 0:
                    for p in bits_to_indices(bi):
                        rdm1[p, p] += ci_ij
                elif n_diff_b == 1:
                    holes = bits_to_indices((bi ^ bj) & bi)
                    parts = bits_to_indices((bi ^ bj) & bj)
                    q = holes[0]
                    p = parts[0]
                    sign = compute_phase(bi, p, q)
                    rdm1[p, q] += sign * ci_ij

    return rdm1


def make_rdm1s(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spin-separated one-particle density matrices.

    Computes the alpha and beta components of the 1-RDM independently.
    The spin-traced 1-RDM equals rdm1a + rdm1b.

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        Tuple (rdm1a, rdm1b) where:
            rdm1a[p,q] = <a+_{p,alpha} a_{q,alpha}>, shape (ncas, ncas).
            rdm1b[p,q] = <a+_{p,beta} a_{q,beta}>, shape (ncas, ncas).
    """
    n_det = len(alpha_strings)
    rdm1a = np.zeros((ncas, ncas), dtype=np.float64)
    rdm1b = np.zeros((ncas, ncas), dtype=np.float64)

    for i in range(n_det):
        ci_i = ci_vector[i]
        if abs(ci_i) < 1e-15:
            continue

        ai = int(alpha_strings[i])
        bi = int(beta_strings[i])

        for j in range(n_det):
            ci_j = ci_vector[j]
            ci_ij = ci_i * ci_j
            if abs(ci_ij) < 1e-15:
                continue

            aj = int(alpha_strings[j])
            bj = int(beta_strings[j])

            n_diff_a = popcount(ai ^ aj) // 2
            n_diff_b = popcount(bi ^ bj) // 2

            # Alpha-alpha contribution (requires identical beta strings)
            if bi == bj:
                if n_diff_a == 0:
                    for p in bits_to_indices(ai):
                        rdm1a[p, p] += ci_ij
                elif n_diff_a == 1:
                    holes = bits_to_indices((ai ^ aj) & ai)
                    parts = bits_to_indices((ai ^ aj) & aj)
                    q = holes[0]
                    p = parts[0]
                    sign = compute_phase(ai, p, q)
                    rdm1a[p, q] += sign * ci_ij

            # Beta-beta contribution (requires identical alpha strings)
            if ai == aj:
                if n_diff_b == 0:
                    for p in bits_to_indices(bi):
                        rdm1b[p, p] += ci_ij
                elif n_diff_b == 1:
                    holes = bits_to_indices((bi ^ bj) & bi)
                    parts = bits_to_indices((bi ^ bj) & bj)
                    q = holes[0]
                    p = parts[0]
                    sign = compute_phase(bi, p, q)
                    rdm1b[p, q] += sign * ci_ij

    return rdm1a, rdm1b


def _make_rdm12s_impl(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute spin-separated 1- and 2-particle reduced density matrices.

    This is the core implementation that computes all spin components of
    both the 1-RDM and 2-RDM in a single pass over determinant pairs.

    Index conventions (PySCF standard):
        rdm1a[p,q] = <a+_{p,alpha} a_{q,alpha}>
        rdm1b[p,q] = <a+_{p,beta}  a_{q,beta}>
        rdm2aa[p,q,r,s] = <a+_{p,alpha} a+_{r,alpha} a_{s,alpha} a_{q,alpha}>
        rdm2ab[p,q,r,s] = <a+_{p,alpha} a+_{r,beta}  a_{s,beta}  a_{q,alpha}>
        rdm2bb[p,q,r,s] = <a+_{p,beta}  a+_{r,beta}  a_{s,beta}  a_{q,beta}>

    Algorithm: Double loop over determinant pairs (I, J). For each pair,
    classify by excitation level (0, 1, or 2 in each spin channel) and
    accumulate weighted contributions c_I * c_J into the density matrix
    elements. The phase factor for each excitation accounts for the
    fermionic anticommutation of creation/annihilation operators.

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        Tuple ((rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)) where each
        rdm1 has shape (ncas, ncas) and each rdm2 has shape
        (ncas, ncas, ncas, ncas).
    """
    n_det = len(alpha_strings)
    rdm1a = np.zeros((ncas, ncas), dtype=np.float64)
    rdm1b = np.zeros((ncas, ncas), dtype=np.float64)
    rdm2aa = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)
    rdm2ab = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)
    rdm2bb = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)

    for i in range(n_det):
        ci_i = ci_vector[i]
        if abs(ci_i) < 1e-15:
            continue

        ai = int(alpha_strings[i])
        bi = int(beta_strings[i])

        for j in range(n_det):
            ci_j = ci_vector[j]
            ci_ij = ci_i * ci_j
            if abs(ci_ij) < 1e-15:
                continue

            aj = int(alpha_strings[j])
            bj = int(beta_strings[j])

            n_diff_a = popcount(ai ^ aj) // 2
            n_diff_b = popcount(bi ^ bj) // 2
            n_excitations = n_diff_a + n_diff_b

            # Skip if more than 2 total excitations (no 2-body contribution)
            if n_excitations > 2:
                continue

            # -----------------------------------------------------------
            # Case 0: Diagonal (identical determinants)
            # -----------------------------------------------------------
            if n_excitations == 0:
                occ_a = bits_to_indices(ai)
                occ_b = bits_to_indices(bi)

                # 1-RDM diagonal contributions
                for p in occ_a:
                    rdm1a[p, p] += ci_ij
                for p in occ_b:
                    rdm1b[p, p] += ci_ij

                # 2-RDM alpha-alpha pairs
                for idx_p, p in enumerate(occ_a):
                    for q in occ_a[idx_p + 1:]:
                        rdm2aa[p, p, q, q] += ci_ij
                        rdm2aa[p, q, q, p] -= ci_ij
                        rdm2aa[q, q, p, p] += ci_ij
                        rdm2aa[q, p, p, q] -= ci_ij

                # 2-RDM beta-beta pairs
                for idx_p, p in enumerate(occ_b):
                    for q in occ_b[idx_p + 1:]:
                        rdm2bb[p, p, q, q] += ci_ij
                        rdm2bb[p, q, q, p] -= ci_ij
                        rdm2bb[q, q, p, p] += ci_ij
                        rdm2bb[q, p, p, q] -= ci_ij

                # 2-RDM alpha-beta pairs (no antisymmetry between spins)
                for p in occ_a:
                    for q in occ_b:
                        rdm2ab[p, p, q, q] += ci_ij

            # -----------------------------------------------------------
            # Case 1a: Single alpha excitation (beta unchanged)
            # -----------------------------------------------------------
            elif n_diff_a == 1 and n_diff_b == 0:
                # holes = orbitals in i but not j; particles = orbitals in j but not i
                # For <i|a+_p a_q|j>, we need a_q to annihilate from j (q in j not i)
                # and a+_p to create to match i (p in i not j)
                _, holes, particles = excitation_info(ai, aj)
                p_cre = holes[0]      # orbital in bra i, not ket j -> created
                q_ann = particles[0]  # orbital in ket j, not bra i -> annihilated
                sign = compute_phase(ai, p_cre, q_ann)

                # 1-RDM alpha contribution
                rdm1a[p_cre, q_ann] += sign * ci_ij

                # 2-RDM alpha-alpha: spectator alpha orbitals
                # Use ket j's alpha string to find common occupied orbitals
                # The spectators are orbitals occupied in both i and j
                occ_a_common = bits_to_indices(ai & aj)
                for r in occ_a_common:
                    # Coulomb: <a+_p a+_r a_r a_q>
                    rdm2aa[p_cre, q_ann, r, r] += sign * ci_ij
                    rdm2aa[r, r, p_cre, q_ann] += sign * ci_ij
                    # Exchange: <a+_p a+_r a_q a_r> with sign flip
                    rdm2aa[p_cre, r, r, q_ann] -= sign * ci_ij
                    rdm2aa[r, q_ann, p_cre, r] -= sign * ci_ij

                # 2-RDM alpha-beta: spectator beta orbitals
                # bi == bj since n_diff_b == 0
                occ_b = bits_to_indices(bi)
                for r in occ_b:
                    rdm2ab[p_cre, q_ann, r, r] += sign * ci_ij

            # -----------------------------------------------------------
            # Case 1b: Single beta excitation (alpha unchanged)
            # -----------------------------------------------------------
            elif n_diff_a == 0 and n_diff_b == 1:
                _, holes, particles = excitation_info(bi, bj)
                p_cre = holes[0]
                q_ann = particles[0]
                sign = compute_phase(bi, p_cre, q_ann)

                # 1-RDM beta contribution
                rdm1b[p_cre, q_ann] += sign * ci_ij

                # 2-RDM beta-beta: spectator beta orbitals
                occ_b_common = bits_to_indices(bi & bj)
                for r in occ_b_common:
                    rdm2bb[p_cre, q_ann, r, r] += sign * ci_ij
                    rdm2bb[r, r, p_cre, q_ann] += sign * ci_ij
                    rdm2bb[p_cre, r, r, q_ann] -= sign * ci_ij
                    rdm2bb[r, q_ann, p_cre, r] -= sign * ci_ij

                # 2-RDM alpha-beta: spectator alpha orbitals
                # ai == aj since n_diff_a == 0
                # Note: rdm2ab[p,q,r,s] = <a+_{p,alpha} a+_{r,beta} a_{s,beta} a_{q,alpha}>
                # For a beta excitation with alpha spectator r:
                # <a+_{r,alpha} a+_{p_cre,beta} a_{q_ann,beta} a_{r,alpha}>
                # = rdm2ab[r, r, p_cre, q_ann]
                occ_a = bits_to_indices(ai)
                for r in occ_a:
                    rdm2ab[r, r, p_cre, q_ann] += sign * ci_ij

            # -----------------------------------------------------------
            # Case 2a: Same-spin alpha-alpha double excitation
            # -----------------------------------------------------------
            elif n_diff_a == 2 and n_diff_b == 0:
                _, holes, particles = excitation_info(ai, aj)
                # holes = orbitals in bra i, not in ket j
                # particles = orbitals in ket j, not in bra i
                p, q = holes[0], holes[1]      # created orbitals (in i)
                r, s = particles[0], particles[1]  # annihilated orbitals (in j)

                # Phase using intermediate determinant (same pattern as slater_condon.py)
                # First excitation: p <- r on bra string (annihilate r from ai-like, create p)
                # But we compute phase on ai: excite p <-> r
                sign1 = compute_phase(ai, p, r)
                # Intermediate: remove p from ai, add r (going from i toward j)
                # Wait - we need to be careful about direction.
                # In slater_condon.py, for excitation_info(alpha_i, alpha_j):
                #   holes are in i, particles in j
                #   sign1 = compute_phase(alpha_i, p=holes[0], r=particles[0])
                #   alpha_mid = (alpha_i ^ (1 << holes[0])) | (1 << particles[0])
                # This means: remove holes[0] from i, add particles[0] -> intermediate
                alpha_mid = (ai ^ (1 << p)) | (1 << r)
                sign2 = compute_phase(alpha_mid, q, s)
                sign = sign1 * sign2

                # The operator a+_p a+_q a_s a_r applied to |j> should give |i>
                # (up to phase). In PySCF convention:
                # rdm2aa[p,q,r,s] = <a+_p a+_r a_s a_q>
                # Our excitation has creation at p,q and annihilation at r,s.
                # Matching: a+_p a+_q a_s a_r  corresponds to
                #   rdm2aa[p, r, q, s] (mapping: 1st-create->p, 2nd-annihilate->r,
                #                                 2nd-create->q, 1st-annihilate->s)
                # Wait, let me be more careful.
                #
                # rdm2aa[A,B,C,D] = <a+_A a+_C a_D a_B>
                # We want to represent: a+_p a+_q a_s a_r
                # = <a+_A a+_C a_D a_B> with A=p, C=q, D=s, B=r
                # So: rdm2aa[p, r, q, s]
                #
                # Antisymmetry: swapping the two annihilation operators (r <-> s)
                # gives a sign flip: a+_p a+_q a_r a_s = -a+_p a+_q a_s a_r
                # = <a+_A a+_C a_D a_B> with A=p, C=q, D=r, B=s
                # So: rdm2aa[p, s, q, r] gets -sign * ci_ij
                #
                # Similarly, swapping creation operators (p <-> q):
                # a+_q a+_p a_s a_r = -a+_p a+_q a_s a_r
                # = <a+_A a+_C a_D a_B> with A=q, C=p, D=s, B=r
                # So: rdm2aa[q, r, p, s] gets -sign * ci_ij
                #
                # Both swaps (p<->q AND r<->s):
                # a+_q a+_p a_r a_s = +a+_p a+_q a_s a_r
                # = rdm2aa[q, s, p, r] gets +sign * ci_ij

                rdm2aa[p, r, q, s] += sign * ci_ij
                rdm2aa[p, s, q, r] -= sign * ci_ij
                rdm2aa[q, r, p, s] -= sign * ci_ij
                rdm2aa[q, s, p, r] += sign * ci_ij

            # -----------------------------------------------------------
            # Case 2b: Same-spin beta-beta double excitation
            # -----------------------------------------------------------
            elif n_diff_a == 0 and n_diff_b == 2:
                _, holes, particles = excitation_info(bi, bj)
                p, q = holes[0], holes[1]
                r, s = particles[0], particles[1]

                sign1 = compute_phase(bi, p, r)
                beta_mid = (bi ^ (1 << p)) | (1 << r)
                sign2 = compute_phase(beta_mid, q, s)
                sign = sign1 * sign2

                rdm2bb[p, r, q, s] += sign * ci_ij
                rdm2bb[p, s, q, r] -= sign * ci_ij
                rdm2bb[q, r, p, s] -= sign * ci_ij
                rdm2bb[q, s, p, r] += sign * ci_ij

            # -----------------------------------------------------------
            # Case 2c: Opposite-spin alpha-beta double excitation
            # -----------------------------------------------------------
            elif n_diff_a == 1 and n_diff_b == 1:
                _, a_holes, a_particles = excitation_info(ai, aj)
                _, b_holes, b_particles = excitation_info(bi, bj)

                # Alpha channel: create at a_holes[0], annihilate at a_particles[0]
                p_a = a_holes[0]      # alpha orbital in bra i (created)
                q_a = a_particles[0]  # alpha orbital in ket j (annihilated)

                # Beta channel: create at b_holes[0], annihilate at b_particles[0]
                p_b = b_holes[0]      # beta orbital in bra i (created)
                q_b = b_particles[0]  # beta orbital in ket j (annihilated)

                sign_a = compute_phase(ai, p_a, q_a)
                sign_b = compute_phase(bi, p_b, q_b)
                sign = sign_a * sign_b

                # The operator: a+_{p_a,alpha} a+_{p_b,beta} a_{q_b,beta} a_{q_a,alpha}
                # PySCF: rdm2ab[A,B,C,D] = <a+_{A,alpha} a+_{C,beta} a_{D,beta} a_{B,alpha}>
                # Match: A=p_a, B=q_a, C=p_b, D=q_b
                rdm2ab[p_a, q_a, p_b, q_b] += sign * ci_ij

    return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)


def make_rdm12s(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Spin-separated one- and two-particle density matrices.

    Computes the alpha/beta components of the 1-RDM and the aa/ab/bb
    components of the 2-RDM.

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        Tuple ((rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)) where:
            rdm1a[p,q] = <a+_{p,alpha} a_{q,alpha}>, shape (ncas, ncas).
            rdm1b[p,q] = <a+_{p,beta}  a_{q,beta}>,  shape (ncas, ncas).
            rdm2aa[p,q,r,s] = <a+_{p,a} a+_{r,a} a_{s,a} a_{q,a}>.
            rdm2ab[p,q,r,s] = <a+_{p,a} a+_{r,b} a_{s,b} a_{q,a}>.
            rdm2bb[p,q,r,s] = <a+_{p,b} a+_{r,b} a_{s,b} a_{q,b}>.
    """
    return _make_rdm12s_impl(ci_vector, alpha_strings, beta_strings, ncas)


def _compute_s_minus_s_plus(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> float:
    """Compute <S_- S_+> expectation value for spin_square calculation.

    Evaluates the S_-S_+ component of <S^2> directly from the CI vector
    and determinant bit strings, without requiring the 2-RDM.

    The operator S_-S_+ = sum_{p,q} a+_{p,beta} a_{p,alpha} a+_{q,alpha} a_{q,beta}
    connects determinant pairs that differ by spin-flip excitations.

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        The expectation value <Psi|S_-S_+|Psi>.
    """
    n_det = len(alpha_strings)
    result = 0.0

    for i in range(n_det):
        ci_i = ci_vector[i]
        if abs(ci_i) < 1e-15:
            continue

        ai = int(alpha_strings[i])
        bi = int(beta_strings[i])

        for j in range(n_det):
            ci_j = ci_vector[j]
            ci_ij = ci_i * ci_j
            if abs(ci_ij) < 1e-15:
                continue

            aj = int(alpha_strings[j])
            bj = int(beta_strings[j])

            n_diff_a = popcount(ai ^ aj)
            n_diff_b = popcount(bi ^ bj)

            if n_diff_a == 0 and n_diff_b == 0:
                # Diagonal: count orbitals with beta but not alpha
                count = popcount(bi & ~ai)
                result += ci_ij * count

            elif n_diff_a == 2 and n_diff_b == 2:
                # Off-diagonal: single swap in each spin channel.
                # Extract which orbitals differ.
                # In alpha: p is in J but not I (hole in I), q is in I but not J
                alpha_diff = ai ^ aj
                beta_diff = bi ^ bj

                # alpha orbital in J but not I
                p_alpha = bits_to_indices(alpha_diff & aj)
                # alpha orbital in I but not J
                q_alpha = bits_to_indices(alpha_diff & ai)

                if len(p_alpha) != 1 or len(q_alpha) != 1:
                    continue

                p = p_alpha[0]  # alpha present in J, absent in I
                q = q_alpha[0]  # alpha present in I, absent in J

                # For S_-S_+ connecting J to I:
                # S_+ flips beta_q -> alpha_q (so J has beta at q, I has alpha at q)
                # S_- flips alpha_p -> beta_p (so J has alpha at p, I has beta at p)
                # Check beta consistency: p gained in I's beta, q lost from I's beta
                beta_gained_in_i = bits_to_indices(beta_diff & bi)
                beta_lost_from_i = bits_to_indices(beta_diff & bj)

                if len(beta_gained_in_i) != 1 or len(beta_lost_from_i) != 1:
                    continue

                if beta_gained_in_i[0] != p or beta_lost_from_i[0] != q:
                    continue

                # Compute phase by applying operators right-to-left to |J>:
                # Step 1: a_{q,beta} |J> -- annihilate beta at q
                phase_1 = popcount(bj & ((1 << q) - 1))
                new_beta = bj ^ (1 << q)

                # Step 2: a+_{q,alpha} -- create alpha at q
                phase_2 = popcount(aj & ((1 << q) - 1))
                new_alpha = aj | (1 << q)

                # Step 3: a_{p,alpha} -- annihilate alpha at p
                phase_3 = popcount(new_alpha & ((1 << p) - 1))

                # Step 4: a+_{p,beta} -- create beta at p
                phase_4 = popcount(new_beta & ((1 << p) - 1))

                total_phase_parity = phase_1 + phase_2 + phase_3 + phase_4
                phase = 1 if total_phase_parity % 2 == 0 else -1

                result += ci_ij * phase

    return result


def make_rdm12(
    ci_vector: np.ndarray,
    alpha_strings: np.ndarray,
    beta_strings: np.ndarray,
    ncas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spin-traced one- and two-particle reduced density matrices.

    Uses the spin-separated implementation and combines:
        rdm1 = rdm1a + rdm1b
        rdm2 = rdm2aa + rdm2ab + rdm2ab.transpose(2,3,0,1) + rdm2bb

    The 2-RDM follows PySCF convention:
        rdm2[p,q,r,s] = <Psi| a+_p a+_r a_s a_q |Psi>

    Energy from RDMs:
        E = einsum('pq,qp', h1e, rdm1) + 0.5 * einsum('pqrs,pqrs', eri, rdm2)

    Args:
        ci_vector: CI coefficient vector, shape (n_det,).
        alpha_strings: Alpha occupation bitstrings, shape (n_det,).
        beta_strings: Beta occupation bitstrings, shape (n_det,).
        ncas: Number of active orbitals.

    Returns:
        Tuple (rdm1, rdm2) where:
            rdm1 has shape (ncas, ncas).
            rdm2 has shape (ncas, ncas, ncas, ncas).
    """
    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = _make_rdm12s_impl(
        ci_vector, alpha_strings, beta_strings, ncas
    )
    rdm1 = rdm1a + rdm1b
    # The ba component is rdm2ab transposed: rdm2ba[r,s,p,q] = rdm2ab[p,q,r,s]
    rdm2 = rdm2aa + rdm2ab + rdm2ab.transpose(2, 3, 0, 1) + rdm2bb
    return rdm1, rdm2
