"""
Data model for ORMAS/RAS subspace definitions and constraint validation.

These classes define the partitioning of the active orbital space into subspaces
with occupation constraints. They carry all the information needed to enumerate
the restricted determinant space, and they live on the ORMASFCISolver object.

PySCF and QDK/Chemistry don't need to know about these classes. The subspace
information is consumed entirely by our solver internals. The user creates a
config, passes it to ORMASFCISolver, and plugs that solver into PySCF's CASCI.
No modifications to PySCF or QDK/Chemistry are required.
"""

from dataclasses import dataclass

__all__ = ["Subspace", "ORMASConfig", "RASConfig"]


@dataclass
class Subspace:
    """A single orbital subspace with occupation constraints.

    Attributes:
        name: Human-readable label (e.g., "metal_3d", "ligand_pi").
        orbital_indices: 0-based indices into the active orbital space. These
            are positions within the active orbitals, NOT the full MO space.
        min_electrons: Minimum total occupation (alpha + beta) allowed.
        max_electrons: Maximum total occupation (alpha + beta) allowed.
    """
    name: str
    orbital_indices: list[int]
    min_electrons: int
    max_electrons: int

    @property
    def n_orbitals(self) -> int:
        return len(self.orbital_indices)

    def validate(self) -> None:
        """Check internal consistency. Raises ValueError if invalid."""
        if self.min_electrons < 0:
            raise ValueError(
                f"Subspace '{self.name}': min_electrons ({self.min_electrons}) "
                f"cannot be negative"
            )
        if self.max_electrons > 2 * self.n_orbitals:
            raise ValueError(
                f"Subspace '{self.name}': max_electrons ({self.max_electrons}) "
                f"exceeds 2 * n_orbitals ({2 * self.n_orbitals})"
            )
        if self.min_electrons > self.max_electrons:
            raise ValueError(
                f"Subspace '{self.name}': min_electrons ({self.min_electrons}) "
                f"> max_electrons ({self.max_electrons})"
            )
        if len(self.orbital_indices) != len(set(self.orbital_indices)):
            raise ValueError(
                f"Subspace '{self.name}': orbital_indices contains duplicates"
            )
        if any(i < 0 for i in self.orbital_indices):
            raise ValueError(
                f"Subspace '{self.name}': orbital_indices contains negative values"
            )


@dataclass
class ORMASConfig:
    """Complete ORMAS partitioning specification for a molecular active space.

    This is the primary configuration object. It defines how the active orbitals
    are partitioned into subspaces and what occupation constraints apply to each.

    The config is passed to ORMASFCISolver, which uses it during kernel() to
    enumerate the restricted determinant space. PySCF never sees this object.

    Attributes:
        subspaces: List of Subspace objects defining the partitioning.
        n_active_orbitals: Total number of active orbitals. Must equal the sum
            of all subspace sizes.
        nelecas: Tuple of (n_alpha, n_beta) active electrons.
    """
    subspaces: list[Subspace]
    n_active_orbitals: int
    nelecas: tuple[int, int]

    @property
    def n_alpha(self) -> int:
        return self.nelecas[0]

    @property
    def n_beta(self) -> int:
        return self.nelecas[1]

    @property
    def n_electrons(self) -> int:
        return self.nelecas[0] + self.nelecas[1]

    @property
    def n_subspaces(self) -> int:
        return len(self.subspaces)

    def validate(self) -> None:
        """Validate the complete configuration.

        Checks individual subspaces, orbital coverage, and global electron
        count feasibility. Raises ValueError with a descriptive message
        if anything is inconsistent.
        """
        for sub in self.subspaces:
            sub.validate()

        # Orbital indices must be non-overlapping
        all_indices = []
        for sub in self.subspaces:
            for idx in sub.orbital_indices:
                if idx in all_indices:
                    raise ValueError(
                        f"Orbital index {idx} appears in multiple subspaces"
                    )
                all_indices.append(idx)

        # Orbital indices must cover the full active space
        if sorted(all_indices) != list(range(self.n_active_orbitals)):
            raise ValueError(
                f"Subspace orbital indices {sorted(all_indices)} do not "
                f"cover the full active space [0, {self.n_active_orbitals})"
            )

        # Global electron constraints must be satisfiable
        total_min = sum(sub.min_electrons for sub in self.subspaces)
        total_max = sum(sub.max_electrons for sub in self.subspaces)
        n_total = self.n_electrons

        if n_total < total_min:
            raise ValueError(
                f"Total electrons ({n_total}) is less than the sum of "
                f"subspace min_electrons ({total_min})"
            )
        if n_total > total_max:
            raise ValueError(
                f"Total electrons ({n_total}) exceeds the sum of "
                f"subspace max_electrons ({total_max})"
            )

    def get_subspace_for_orbital(self, orbital_idx: int) -> "Subspace":
        """Return the subspace containing the given orbital index."""
        for sub in self.subspaces:
            if orbital_idx in sub.orbital_indices:
                return sub
        raise ValueError(f"Orbital {orbital_idx} not found in any subspace")


@dataclass
class RASConfig:
    """Convenience class for the traditional 3-subspace RAS specification.

    Converts to an ORMASConfig via to_ormas_config(). This exists purely for
    user convenience; internally everything goes through ORMASConfig.

    In the RAS scheme:
        RAS1: Mostly doubly occupied orbitals. At most max_holes electrons
              can be removed (creating holes).
        RAS2: Fully flexible orbitals (full CI within this subspace).
        RAS3: Mostly empty orbitals. At most max_particles electrons can
              be added.

    Attributes:
        ras1_orbitals: Orbital indices for RAS1. Can be empty list.
        ras2_orbitals: Orbital indices for RAS2. Can be empty list.
        ras3_orbitals: Orbital indices for RAS3. Can be empty list.
        max_holes_ras1: Maximum number of holes (missing electrons) in RAS1.
        max_particles_ras3: Maximum number of particles (electrons) in RAS3.
        nelecas: Tuple of (n_alpha, n_beta) active electrons.
    """
    ras1_orbitals: list[int]
    ras2_orbitals: list[int]
    ras3_orbitals: list[int]
    max_holes_ras1: int
    max_particles_ras3: int
    nelecas: tuple[int, int]

    def to_ormas_config(self) -> ORMASConfig:
        """Convert to an ORMASConfig.

        RAS1 default occupation is fully occupied (2 * n_ras1 electrons).
        max_holes controls how many can be removed.
        RAS2 has no restrictions (min=0, max=2*n_ras2).
        RAS3 default occupation is empty. max_particles controls how many
        electrons it can accept.
        """
        n_ras1 = len(self.ras1_orbitals)
        n_ras2 = len(self.ras2_orbitals)
        n_ras3 = len(self.ras3_orbitals)
        n_total = self.nelecas[0] + self.nelecas[1]

        subspaces = []

        if n_ras1 > 0:
            ras1_default = min(2 * n_ras1, n_total)
            subspaces.append(Subspace(
                name="RAS1",
                orbital_indices=list(self.ras1_orbitals),
                min_electrons=max(0, ras1_default - self.max_holes_ras1),
                max_electrons=min(2 * n_ras1, n_total),
            ))

        if n_ras2 > 0:
            subspaces.append(Subspace(
                name="RAS2",
                orbital_indices=list(self.ras2_orbitals),
                min_electrons=0,
                max_electrons=2 * n_ras2,
            ))

        if n_ras3 > 0:
            subspaces.append(Subspace(
                name="RAS3",
                orbital_indices=list(self.ras3_orbitals),
                min_electrons=0,
                max_electrons=min(self.max_particles_ras3, 2 * n_ras3),
            ))

        n_active = n_ras1 + n_ras2 + n_ras3

        config = ORMASConfig(
            subspaces=subspaces,
            n_active_orbitals=n_active,
            nelecas=self.nelecas,
        )
        config.validate()
        return config
