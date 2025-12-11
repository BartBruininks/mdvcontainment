"""
Mapping atoms to and from voxels, taking MDAnalysis.Atomgroups as the atomic representation.
"""

# Python
from typing import Dict, List, Optional, Tuple, Union

# Python External
import numpy as np
import numpy.typing as npt
import MDAnalysis as mda

# Cython Module
from .atoms_voxels_mapping import create_efficient_mapping_cy, voxels2atomgroup_cy


def dim2lattice(
    x: float,
    y: float,
    z: float,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0
) -> npt.NDArray[np.float64]:
    """
    Convert unit cell dimensions to a lattice matrix representation.

    Parameters
    ----------
    x : float
        Length of the first unit cell edge (Angstroms).
    y : float
        Length of the second unit cell edge (Angstroms).
    z : float
        Length of the third unit cell edge (Angstroms).
    alpha : float, default=90.0
        Angle between y and z edges (degrees).
    beta : float, default=90.0
        Angle between x and z edges (degrees).
    gamma : float, default=90.0
        Angle between x and y edges (degrees).
    
    Returns
    -------
    np.ndarray
        3×3 box matrix representing the unit cell in Angstroms.
    
    Notes
    -----
    The returned matrix follows the convention where the first row is along x,
    the second row is in the xy-plane, and the third row completes the cell.
    """
    cosa = np.cos(np.pi * alpha / 180)
    cosb = np.cos(np.pi * beta / 180)
    cosg = np.cos(np.pi * gamma / 180)
    sing = np.sin(np.pi * gamma / 180)

    zx = z * cosb
    zy = z * (cosa - cosb * cosg) / sing
    zz = np.sqrt(z**2 - zx**2 - zy**2)

    return np.array([x, 0, 0, y * cosg, y * sing, 0, zx, zy, zz]).reshape((3, 3))


def linear_blur(
    array: npt.NDArray,
    box: npt.NDArray[np.float64],
    span: int,
    inplace: bool = True
) -> npt.NDArray:
    """
    Apply linear blurring to a 3D array with periodic boundary conditions.
    
    This function performs a rolling average by shifting the array in each direction
    (x, y, z) up to the specified span, accounting for the triclinic box geometry.

    Parameters
    ----------
    array : np.ndarray
        3D array to be blurred.
    box : np.ndarray
        3×3 box matrix for proper periodic boundary condition handling.
    span : int
        Number of voxels to include in each direction for blurring.
    inplace : bool, default=True
        If True, performs cumulative blurring where each rolled array
        is the current blurred state. If False, always rolls the original array.

    Returns
    -------
    np.ndarray
        Blurred 3D array with the same shape as input.
    
    Notes
    -----
    The box matrix is assumed to be lower triangular, which affects how
    shifts are applied in y and z directions to maintain periodicity.
    """
    blurred = np.copy(array)

    if inplace:
        other = blurred
    else:
        other = array
        
    for shift in range(1, span + 1):
        # The box matrix is triangular
        
        # ... so a roll over x is just okay
        blurred += np.roll(other, shift, axis=0)
        blurred += np.roll(other, -shift, axis=0)
        
        # ... but a roll over y may have an x-shift
        xshift = int(shift * box[1, 0])
        rolled = np.roll(other, shift, 1)
        rolled[:, 0, :] = np.roll(rolled[:, 0, :], -xshift, 0)
        blurred += rolled
        
        rolled = np.roll(other, -shift, 1)
        rolled[:, -1, :] = np.roll(rolled[:, -1, :], xshift, 0)
        blurred += rolled
        
        # ... and a roll over z may have an x- and a y-shift
        xyshift = (shift * box[2, :2]).astype(int)
        rolled = np.roll(other, shift, 2)
        rolled[:, :, 0] = np.roll(rolled[:, :, 0], xyshift, (0, 1))
        blurred += rolled
        
        rolled = np.roll(other, -shift, 2)
        rolled[:, :, -1] = np.roll(rolled[:, :, -1], -xyshift, (0, 1))
        blurred += rolled
        
    return blurred


def _voxelate_atomgroup(
    atomgroup: mda.AtomGroup,
    resolution: float,
    max_offset: Union[float, bool] = 0.05
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Convert atomgroup positions to voxel indices at specified resolution.
    
    Bins atoms into voxels as close to the target resolution as possible.
    Raises an error if the actual resolution deviates too much from the target.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        The atomgroup to voxelate.
    resolution : float
        Target voxel resolution (nm).
    max_offset : float or bool, default=0.05
        Maximum allowed fractional deviation from target resolution.
        If True, checks are enabled with default 5% tolerance.
        If float, uses that value as the tolerance threshold.

    Returns
    -------
    voxels : np.ndarray
        (N, 3) array of voxel indices for each atom.
    nbox : np.ndarray
        3×3 diagonal matrix containing the number of voxels in each dimension.
    
    Raises
    ------
    ValueError
        If resolution deviation exceeds max_offset in any dimension.
    
    Notes
    -----
    Atom positions are modified in-place to snap to voxel centers.
    """
    check = max_offset is True
    resolution = abs(resolution)

    box = dim2lattice(*atomgroup.dimensions)
    # The 10 is for going from nm to Angstrom
    nbox = (box / (10 * resolution)).round().astype(int)  # boxels
    unit = np.linalg.inv(nbox) @ box  # voxel shape
    deviation = (0.1 * (unit**2).sum(axis=1)**0.5 - resolution) / resolution

    # Check for scaling artifacts
    if check and (np.abs(deviation) > max_offset).any():
        raise ValueError(
            f'A scaling artifact of more than {max_offset * 100}% deviation '
            f'from the target resolution was detected in frame '
            f'{atomgroup.universe.trajectory.frame}. Consider increasing the resolution.'
        )

    transform = np.linalg.inv(box) @ nbox  # transformation to voxel indices
    fraxels = atomgroup.positions @ transform
    voxels = np.floor(fraxels).astype(np.int32)
    fraxels -= voxels
    
    # Put everything in brick at origin
    for dim in (2, 1, 0):
        shifts = voxels[:, dim] // nbox[dim, dim]
        voxels -= shifts[:, None] * nbox[dim, :]

    # Put every particle in the correct voxel
    atomgroup.positions = (voxels + fraxels) @ np.linalg.inv(transform)
        
    return voxels, nbox


def _create_efficient_mapping(
    voxels: npt.NDArray[np.int32],
    atom_indices: npt.NDArray[np.int64]
) -> Dict[str, Union[npt.NDArray, Dict]]:
    """
    Create a memory-efficient hash-based mapping from voxels to atoms.
    
    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) array of voxel coordinates for each atom.
    atom_indices : np.ndarray
        (N,) array of atom indices.
    
    Returns
    -------
    dict
        Mapping dictionary containing:
        - 'atom_voxels': (N, 3) array mapping each atom to its voxel coords
        - 'voxel_to_atoms': dict mapping voxel (x,y,z) tuple to atom indices array
        - 'atom_indices': original atom indices array
    """
    # Ensure correct dtypes
    voxels = np.asarray(voxels, dtype=np.int32)
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    
    return create_efficient_mapping_cy(voxels, atom_indices)


def create_voxels(
    atomgroup: mda.AtomGroup,
    resolution: float,
    max_offset: Union[float, bool] = 0.05,
    return_mapping: bool = True
) -> Tuple[npt.NDArray[np.bool_], Optional[Dict]]:
    """
    Convert an atomgroup into a voxelized boolean occupancy grid.
    
    Creates a 3D boolean array where True indicates voxel occupancy by atoms,
    optionally returning a mapping structure for reverse lookup.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        The atomgroup to voxelate.
    resolution : float
        Target voxel resolution (nm).
    max_offset : float or bool, default=0.05
        Maximum allowed fractional deviation from target resolution (5% default).
    return_mapping : bool, default=True
        Whether to return the voxel-to-atom mapping structure.
    
    Returns
    -------
    voxels : np.ndarray
        Boolean 3D array indicating voxel occupancy.
    mapping : dict or None
        If return_mapping=True, contains:
        - 'atom_voxels': (N, 3) array mapping each atom to voxel coordinates
        - 'voxel_to_atoms': dict mapping voxel tuple to atom indices
        - 'atom_indices': original atom indices from atomgroup
        If return_mapping=False, returns None.
    
    Examples
    --------
    >>> import MDAnalysis as mda
    >>> u = mda.Universe("protein.pdb")
    >>> voxels, mapping = create_voxels(u.atoms, resolution=0.5)
    >>> print(f"Grid shape: {voxels.shape}")
    >>> print(f"Occupied voxels: {voxels.sum()}")
    """
    voxels, nbox = _voxelate_atomgroup(atomgroup, resolution, max_offset=max_offset)
    x, y, z = voxels.T
    
    # Create explicit voxel grid
    explicit = np.zeros(np.diagonal(nbox), dtype=bool)
    explicit[x, y, z] = True
    
    if return_mapping:
        # Create memory-efficient mapping
        mapping = _create_efficient_mapping(voxels, atomgroup.ix)
        return explicit, mapping
    else:
        return explicit, None


def voxels2atomgroup(
    voxels: npt.NDArray[np.int32],
    mapping: Dict[str, Union[npt.NDArray, Dict]],
    atomgroup: mda.AtomGroup
) -> mda.AtomGroup:
    """
    Convert voxel coordinates back to their corresponding atoms.
    
    Uses hash-based lookup for efficient reverse mapping from voxels to atoms.

    Parameters
    ----------
    voxels : array-like
        (M, 3) array of voxel coordinates to look up.
    mapping : dict
        Mapping structure from create_voxels containing:
        - 'voxel_to_atoms': dict mapping (x,y,z) tuples to atom indices
        - 'atom_indices': original atom indices
    atomgroup : mda.AtomGroup
        Original atomgroup used to create the mapping.
    
    Returns
    -------
    mda.AtomGroup
        Atomgroup containing all atoms located in the specified voxels.
        Returns an empty atomgroup if no atoms are found.
    
    Examples
    --------
    >>> voxels, mapping = create_voxels(protein, resolution=0.5)
    >>> occupied = np.argwhere(voxels)
    >>> atoms = voxels2atomgroup(occupied, mapping, protein)
    >>> print(f"Retrieved {len(atoms)} atoms")
    """
    voxels_array = np.asarray(voxels, dtype=np.int32)
    voxel_to_atoms = mapping['voxel_to_atoms']
    atom_indices = mapping['atom_indices']
    
    if len(voxels_array) == 0:
        return atomgroup.universe.atoms[[]]
    
    selected_atom_indices = voxels2atomgroup_cy(
        voxels_array, voxel_to_atoms, atom_indices
    )
    
    if len(selected_atom_indices) == 0:
        return atomgroup.universe.atoms[[]]
    
    return atomgroup.universe.atoms[selected_atom_indices]


def dilate_voxels(voxels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Dilate voxel occupancy by expanding occupied regions outward.
    
    Performs morphological dilation by one voxel in all directions,
    effectively growing the occupied regions.

    Parameters
    ----------
    voxels : np.ndarray
        Boolean 3D array of voxel occupancy.

    Returns
    -------
    np.ndarray
        Boolean 3D array after dilation operation.
    
    Notes
    -----
    The operation is performed in-place on a copy of the input array.
    """
    nbox = np.diag(voxels.shape)
    voxels = linear_blur(voxels, nbox, 1)
    voxels = voxels.astype(bool)
    return voxels


def erode_voxels(voxels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Erode voxel occupancy by shrinking occupied regions inward.
    
    Performs morphological erosion by one voxel in all directions,
    effectively removing the outer layer of occupied regions.

    Parameters
    ----------
    voxels : np.ndarray
        Boolean 3D array of voxel occupancy.

    Returns
    -------
    np.ndarray
        Boolean 3D array after erosion operation.
    
    Notes
    -----
    The operation is performed in-place on a copy of the input array.
    """
    nbox = np.diag(voxels.shape)
    voxels = linear_blur(~voxels, nbox, 1)
    voxels = ~voxels
    return voxels


def morph_voxels(
    voxels: npt.NDArray[np.bool_],
    morph_str: str = 'de'
) -> npt.NDArray[np.bool_]:
    """
    Apply a sequence of morphological operations to voxel occupancy.
    
    Sequentially applies dilation and/or erosion operations as specified
    by the morph string. Common patterns include closing ('de') to fill
    small holes and opening ('ed') to remove small protrusions.

    Parameters
    ----------
    voxels : np.ndarray
        Boolean 3D array of voxel occupancy.
    morph_str : str, default='de'
        String specifying the sequence of operations:
        - 'd' for dilation (expand occupied regions)
        - 'e' for erosion (shrink occupied regions)
   
    Returns
    -------
    np.ndarray
        Boolean 3D array after applying all morphological operations.
    
    Raises
    ------
    ValueError
        If morph_str contains characters other than 'd' or 'e'.
    
    Examples
    --------
    >>> # Closing operation (fill small holes)
    >>> closed = morph_voxels(voxels, 'de')
    >>> 
    >>> # Opening operation (remove small protrusions)
    >>> opened = morph_voxels(voxels, 'ed')
    >>> 
    >>> # Multiple operations
    >>> smoothed = morph_voxels(voxels, 'dede')
    """
    for operation in morph_str:
        if operation == 'd':
            voxels = dilate_voxels(voxels)
        elif operation == 'e':
            voxels = erode_voxels(voxels)
        else:
            raise ValueError(
                f"Unknown morph operation '{operation}' in morph string '{morph_str}'. "
                f"Use 'd' for dilation and 'e' for erosion."
            )
    return voxels


def close_voxels(voxels: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Close voxel occupancy by filling small holes and gaps.
    
    Performs morphological closing (dilation followed by erosion) to
    fill small holes while preserving the overall structure size.

    Parameters
    ----------
    voxels : np.ndarray
        Boolean 3D array of voxel occupancy.
    
    Returns
    -------
    np.ndarray
        Boolean 3D array after closing operation.
    
    Notes
    -----
    This is equivalent to calling morph_voxels(voxels, 'de').
    """
    return morph_voxels(voxels, morph_str='de')


def voxels_to_universe(
    arr: npt.NDArray[Union[np.int_, np.bool_]],
    scale: float = 1.0,
    place_in_center: bool = True,
    universe: Optional[mda.Universe] = None,
    nodes: Optional[List[int]] = None,
) -> mda.Universe:
    """
    Convert a labeled voxel array into an MDAnalysis Universe.
    
    Creates a Universe where each selected voxel becomes an atom positioned
    at the voxel coordinates. Useful for visualizing voxel-based analyses.

    Parameters
    ----------
    arr : np.ndarray
        Labeled voxel array (integer or boolean). Each unique value
        represents a different node/cluster.
    scale : float, default=1.0
        Scaling factor applied to voxel indices to get positions (nm).
        Automatically converted to Angstroms (×10) if no universe provided.
    place_in_center : bool, default=True
        If True, positions atoms at voxel centers rather than corners.
    universe : mda.Universe, optional
        Reference universe to inherit box dimensions from. If provided,
        scaling is calculated from universe.dimensions / arr.shape.
    nodes : list of int, optional
        Subset of node IDs to include in the output universe.
        If None, includes all voxels with non-zero values.

    Returns
    -------
    mda.Universe
        Universe with atoms positioned at voxel coordinates.
        Atom names correspond to the voxel node IDs from the array.
    
    Examples
    --------
    >>> # Convert boolean occupancy to universe
    >>> voxels, _ = create_voxels(protein, resolution=0.5)
    >>> voxel_universe = voxels_to_universe(voxels.astype(int), scale=0.5)
    >>> 
    >>> # Convert labeled clusters, selecting specific nodes
    >>> cluster_universe = voxels_to_universe(
    ...     labeled_array, universe=protein, nodes=[1, 2, 3]
    ... )
    """
    # Optional filtering
    if nodes is not None:
        mask = np.isin(arr, nodes)
        # Keep only positions + values that match
        filtered_indices = np.array(np.where(mask)).T  # shape (M, 3)
        filtered_values = arr[mask].ravel()
        n_atoms = filtered_indices.shape[0]
    else:
        filtered_indices = np.array(list(np.ndindex(arr.shape)))
        filtered_values = arr.flatten()
        n_atoms = filtered_indices.shape[0]

    # Scaling
    if universe is None:
        scale *= 10.0  # nm to Å
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
    u.dimensions = np.array([*box, 90, 90, 90], dtype=np.float64)

    # Residue/atom names come from array values
    names = [str(v) for v in filtered_values]
    u.add_TopologyAttr("name", names)
    
    return u