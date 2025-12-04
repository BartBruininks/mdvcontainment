# Python
from pathlib import Path

# Python External
import numpy as np
import MDAnalysis as mda


def voxels_to_gro(
    path: Path,
    arr,
    scale: float = 1.0,
    place_in_center: bool = True,
    universe=None,
    nodes=None,
):
    # Optional filtering
    if nodes is not None:
        mask = np.isin(arr, nodes)
        # Keep only positions + values that match
        filtered_indices = np.array(np.where(mask)).T        # shape (M, 3)
        filtered_values = arr[mask].ravel()
        n_atoms = filtered_indices.shape[0]
    else:
        filtered_indices = np.array(list(np.ndindex(arr.shape)))
        filtered_values = arr.flatten()
        n_atoms = filtered_indices.shape[0]

    # Scaling
    if universe is None:
        scale *= 10.0  # Å to nm
        box_scale = scale
    else:
        box_scale = universe.dimensions[:3] / np.array(arr.shape)

    # Build Universe
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_atoms,
        atom_resindex=range(n_atoms),
        residue_segindex=[0] * n_atoms,
        trajectory=True,
    )

    # Coordinates
    u.atoms.positions = filtered_indices * box_scale

    if place_in_center:
        u.atoms.positions += box_scale / 2

    # Box dimensions always represent full array volume
    box = np.array(arr.shape) * box_scale
    u.dimensions = np.array([*box, 90, 90, 90], dtype=np.int32)

    # Residue/atom names come from array values
    names = [str(v) for v in filtered_values]
    u.add_TopologyAttr("name", names)

    # Output
    u.atoms.write(path)
