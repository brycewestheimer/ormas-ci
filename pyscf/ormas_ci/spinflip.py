"""Spin-flip ORMAS-CI determinant enumeration and reference analysis.

This module handles the translation from SF-ORMAS configuration to
standard ORMAS determinant enumeration. The key operation is converting
an SFORMASConfig (which specifies reference spin, target spin, and
spin-flip count) into a standard ORMASConfig in the target M_s sector,
then delegating to the existing determinant enumeration machinery.
"""

import numpy as np

from pyscf.ormas_ci.determinants import build_determinant_list, count_determinants
from pyscf.ormas_ci.subspaces import SFORMASConfig


def generate_sf_determinants(
    sf_config: SFORMASConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate all determinants for an SF-ORMAS calculation.

    Translates the SF config to a standard ORMAS config in the target
    M_s sector and delegates to build_determinant_list().

    Args:
        sf_config: The spin-flip ORMAS configuration.

    Returns:
        Tuple of (alpha_strings, beta_strings) as int64 numpy arrays,
        same format as standard ORMAS.
    """
    ormas_config = sf_config.to_ormas_config()
    return build_determinant_list(ormas_config)


def count_sf_determinants(sf_config: SFORMASConfig) -> int:
    """Count determinants without full enumeration.

    Args:
        sf_config: The spin-flip ORMAS configuration.

    Returns:
        Total number of determinants in the SF-ORMAS space.
    """
    ormas_config = sf_config.to_ormas_config()
    return count_determinants(ormas_config)


def validate_reference_consistency(
    sf_config: SFORMASConfig,
    mo_occ: np.ndarray | None = None,
) -> dict:
    """Validate the SF configuration against a ROHF reference.

    Checks that the reference occupation pattern is consistent with the
    subspace assignments and that the SF-CAS contains the singly-occupied
    reference orbitals.

    Args:
        sf_config: The spin-flip ORMAS configuration.
        mo_occ: MO occupation numbers from ROHF (shape: (n_mo,) with
            values 0, 1, 2). If provided, checks consistency with
            subspace assignments.

    Returns:
        Diagnostic information including:
        - 'n_det': determinant count
        - 'nelecas_ref': reference (alpha, beta) in active space
        - 'nelecas_target': target (alpha, beta) in active space
        - 'ref_singly_occupied': indices of singly-occupied orbitals
        - 'warnings': list of warning strings
    """
    warnings: list[str] = []
    ref_a, ref_b = sf_config.nelecas_reference
    tgt_a, tgt_b = sf_config.nelecas_target

    if tgt_a < 0:
        raise ValueError(
            f"Target alpha electrons ({tgt_a}) is negative. "
            f"Too many spin flips ({sf_config.n_spin_flips}) for "
            f"reference with {ref_a} alpha electrons."
        )

    n_det = count_sf_determinants(sf_config)

    # Identify singly-occupied reference orbitals if mo_occ provided
    ref_singly_occ: list[int] = []
    if mo_occ is not None:
        ref_singly_occ = [
            i for i, occ in enumerate(mo_occ) if abs(occ - 1.0) < 0.1
        ]

        # Check: singly occupied orbitals should be in the SF-CAS subspace
        sf_cas_indices: set[int] = set()
        for sub in sf_config.subspaces:
            if sub.name.lower() in ("sf_cas", "ras2_sf", "ras2"):
                sf_cas_indices.update(sub.orbital_indices)

        if sf_cas_indices:
            misplaced = [
                i for i in ref_singly_occ if i not in sf_cas_indices
            ]
            if misplaced:
                warnings.append(
                    f"Singly-occupied reference orbitals {misplaced} "
                    f"are not in the SF-CAS subspace. This may lead to "
                    f"a spin-incomplete expansion. Consider adjusting "
                    f"subspace assignments."
                )

    return {
        "n_det": n_det,
        "nelecas_ref": (ref_a, ref_b),
        "nelecas_target": (tgt_a, tgt_b),
        "ref_singly_occupied": ref_singly_occ,
        "warnings": warnings,
    }


def build_reference_determinant(
    sf_config: SFORMASConfig,
    active_occ: np.ndarray | None = None,
) -> tuple[int, int]:
    """Construct the reference determinant bit strings.

    If active_occ is not provided, constructs the aufbau reference
    (fill from orbital 0 upward).

    Args:
        sf_config: The spin-flip ORMAS configuration.
        active_occ: Active space occupation numbers (0, 1, or 2 per
            orbital).

    Returns:
        (alpha_string, beta_string) for the reference determinant.
    """
    ref_a, ref_b = sf_config.nelecas_reference
    norb = sf_config.n_active_orbitals

    if active_occ is not None:
        alpha_str = 0
        beta_str = 0
        for i in range(norb):
            occ = int(round(active_occ[i]))
            if occ == 2:
                alpha_str |= 1 << i
                beta_str |= 1 << i
            elif occ == 1:
                # Singly occupied: assign to alpha (high-spin reference)
                alpha_str |= 1 << i
        return (alpha_str, beta_str)
    else:
        # Aufbau: fill beta first (doubly occupy lowest), then alpha
        alpha_str = 0
        beta_str = 0
        for i in range(ref_b):
            alpha_str |= 1 << i
            beta_str |= 1 << i
        for i in range(ref_b, ref_a):
            alpha_str |= 1 << i
        return (alpha_str, beta_str)
