"""Backward compatibility — use ``from pyscf.ormas_ci import ...`` instead."""

import warnings

warnings.warn(
    "Importing from 'ormas_ci' is deprecated. "
    "Use 'from pyscf.ormas_ci import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pyscf.ormas_ci import *  # noqa: E402, F401, F403
from pyscf.ormas_ci import __all__, __version__  # noqa: E402, F401
