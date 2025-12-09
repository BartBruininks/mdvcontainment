"""
Mapping atoms to and from voxels, taking MDAnalysis.Atomgroups as the atomic representation.
"""

# Python
from pathlib import Path

# Python External
import numpy as np
import MDAnalysis as mda

# Cython Module
from .atoms_voxels_mapping import create_efficient_mapping_cy, voxels2atomgroup_cy


def dim2lattice(x, y, z, alpha=90, beta=90, gamma=90):
    """Convert dimensions (lengths/angles) to lattice matrix.

    Parameters
    ----------
    x, y, z : float
        Lengths of the unit cell edges.
    alpha, beta, gamma : float, optional
        Angles between the edges in degrees. Default is 90 degrees.
    
    Returns
    -------
    box : 3x3 numpy array
        The box matrix representing the unit cell.
    """
    cosa = np.cos( np.pi * alpha / 180 )
    cosb = np.cos( np.pi * beta / 180 )
    cosg = np.cos( np.pi * gamma / 180 )
    sing = np.sin( np.pi * gamma / 180 )

    zx = z * cosb
    zy = z * ( cosa - cosb * cosg ) / sing
    zz = np.sqrt( z**2 - zx**2 - zy**2 )

    return np.array([x, 0, 0, y * cosg, y * sing, 0, zx, zy, zz]).reshape((3,3))

def linear_blur(array, box, span, inplace=True):
    """
    Perform linear blurring of an array by rolling
    over x, y, and z directions, for each value 
    up to span. If inplace is True, the rolled 
    array is always the target array, causing
    a full blur.
    
    Parameters
    ----------
    array: 3D numpy array
        The array to be blurred.
    box: 3x3 numpy array
        The box matrix for PBC handling.
    span: int
        The number of voxels to blur over in each direction.
    inplace: bool, optional
        Whether to perform the blurring in place. Default is True.

    Returns
    -------
    blurred: 3D numpy array
        The blurred array.
    """

    blurred = np.copy(array)

    if inplace:
        other = blurred
    else:
        other = array
        
    for shift in range(1, span+1):
        # The box matrix is triangular
        
        # ... so a roll over x is just okay
        blurred += np.roll(other, shift, axis=0)
        blurred += np.roll(other, -shift, axis=0)
        
        # ... but a roll over y may have an x-shift
        #
        xshift = shift * box[1, 0]
        rolled = np.roll(other,  shift, 1)
        rolled[:, 0, :] = np.roll(rolled[:, 0, :], -xshift, 0)
        blurred += rolled
        #
        rolled = np.roll(other, -shift, 1)
        rolled[:, -1, :] = np.roll(rolled[:, -1, :], xshift, 0)
        blurred += rolled
        
        # .. and a roll over z may have an x- and a y-shift
        #
        xyshift = shift * box[2, :2]
        rolled = np.roll(other,  shift, 2)
        rolled[:, :, 0] = np.roll(rolled[:, :, 0], xyshift, (0, 1))
        blurred += rolled
        #
        rolled = np.roll(other, -shift, 2)
        rolled[:, :, -1] = np.roll(rolled[:, :, -1], -xyshift, (0, 1))
        blurred += rolled
    return blurred

def _voxelate_atomgroup(atomgroup, resolution, max_offset=0.05):
    """
    Takes an atomgroup and bins it as close to the resolution as
    possible. If the offset of the actual resolution in at least one
    dimension is more than by default 5%, the function will stop and
    return an error specifying the actual offset in all dimensions
    plus the frame in which the mapping error occurred.

    Parameters
    ----------
    atomgroup: MDAnalysis AtomGroup
        The atomgroup to voxelate.
    resolution: float
        The target resolution in nm.
    max_offset: float, optional
        The maximum allowed offset from the target resolution in any dimension
        as a fraction of the target resolution. Default is 0.05 (5%). 
    """
    check = max_offset == True
    resolution = abs(resolution)

    box = dim2lattice(*atomgroup.dimensions)
    # The 10 is for going from nm to Angstrom
    nbox = (box / (10 * resolution)).round().astype(int) # boxels
    unit = np.linalg.inv(nbox) @ box                     # voxel shape
    deviation = (0.1 * (unit**2).sum(axis=1)**0.5 - resolution) / resolution

    # check for scaling artifacts
    if check and (np.abs(deviation) > max_offset).any():
        raise ValueError(
            'A scaling artifact has occurred of more than {}% '
            'deviation from the target resolution in frame {} was '
            'detected. You could consider increasing the '
            'resolution.'.format(max_offset,
            atomgroup.universe.trajectory.frame)
        )

    transform = np.linalg.inv(box) @ nbox                 # transformation to voxel indices
    fraxels = atomgroup.positions @ transform         
    voxels = np.floor(fraxels).astype(int)
    fraxels -= voxels
    
    # Put everything in brick at origin
    for dim in (2, 1, 0):
        shifts = voxels[:, dim] // nbox[dim, dim]
        voxels -= shifts[:, None] * nbox[dim, :]

    # Put every particle in the correct voxel
    atomgroup.positions = (voxels + fraxels) @ np.linalg.inv(transform)
        
    return voxels, nbox

def _create_efficient_mapping(voxels, atom_indices):
    """
    Creates a memory-efficient mapping structure using hash-based indexing.
    
    Parameters
    ----------
    voxels: (N, 3) array of voxel coordinates per atom
    atom_indices: (N,) array of atom indices
    
    Returns
    -------
    mapping: dict containing:
        - 'atom_voxels': (N, 3) array mapping each atom to its voxel coords
        - 'voxel_to_atoms': dict mapping voxel coords (as tuple) -> array of atom indices
        - 'atom_indices': original atom indices array
    """
    # Ensure correct dtypes
    voxels = np.asarray(voxels, dtype=np.int32)
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    
    return create_efficient_mapping_cy(voxels, atom_indices)

def create_voxels(atomgroup, resolution, max_offset=0.05, return_mapping=True):
    """
    Takes an atomgroup and bins it as close to the resolution as possible.
    
    Parameters
    ----------
    atomgroup: MDAnalysis AtomGroup
        The atomgroup to voxelate.
    resolution: float
        The target resolution in nm.
    max_offset: float, optional
        The maximum allowed offset from the target resolution in any dimension
        as a fraction of the target resolution. Default is 0.05 (5%).
    return_mapping: bool, optional
        Whether to return the mapping from voxels to atoms. Default is True.
    
    Returns
    -------
    voxels: boolean 3D array of voxel occupancy
    mapping: dict with keys:
        - 'atom_clusters': (n_atoms,) array mapping each atom to its voxel cluster ID
        - 'cluster_indices': dict mapping voxel cluster ID to array of atom positions
        - 'cluster_coords': dict mapping voxel cluster ID to (x,y,z) voxel coordinates
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

def voxels2atomgroup(voxels, mapping, atomgroup):
    """
    Returns an atomgroup corresponding to specified voxels.
    Uses hash-based lookup instead of memory-intensive broadcasting.
    
    Parameters
    ----------
    voxels: array-like of shape (M, 3)
        Voxel positions to convert back to atoms.
    mapping: dict from create_voxels containing:
        - 'voxel_to_atoms': dict mapping voxel (x,y,z) tuple -> atom position indices
        - 'atom_indices': original atom indices
    atomgroup: MDAnalysis AtomGroup
        The original atomgroup from which the mapping was created.
    
    Returns
    -------
    atomgroup: MDAnalysis AtomGroup
        The atomgroup corresponding to the provided voxel positions.
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

def dilate_voxels(voxels):
    """
    Dilates in place.

    Parameters
    ----------
    voxels: boolean 3D array of voxel occupancy

    Returns
    -------
    voxels: boolean 3D array of voxel occupancy after dilation
    """
    nbox = np.diag(voxels.shape)
    # Possible dilation and erosion to remove small holes for CG data.
    voxels = linear_blur(voxels, nbox, 1)
    voxels = voxels.astype(bool)
    return voxels

def erode_voxels(voxels):
    """
    Erodes in place.

    Parameters
    ----------
    voxels: boolean 3D array of voxel occupancy

    Returns
    -------
    voxels: boolean 3D array of voxel occupancy after erosion
    """
    nbox = np.diag(voxels.shape)
    # Possible dilation and erosion to remove small holes for CG data.
    voxels = linear_blur(~voxels, nbox, 1)
    voxels = ~voxels
    return voxels

def morph_voxels(voxels, morph_str='de'):
    """
    Morphs the voxels by sequentially applying dilations and or erosions
    as specified by the morph string.

    Example (closing): 'de' means dilation followed by erosion.
    Example (opening): 'ed' means erosion followed by dilation.

    Parameters
    ----------
    voxels: boolean 3D array of voxel occupancy
    morph_str: str
        String specifying the morph operations to apply.
        'd' for dilation, 'e' for erosion.
   
    Returns
    -------
    voxels: boolean 3D array of voxel occupancy after morphing
    """
    for operation in morph_str:
        if operation == 'd':
            voxels = dilate_voxels(voxels)
        elif operation == 'e':
            voxels = erode_voxels(voxels)
        else:
            raise ValueError(f"Unknown morph operation '{operation}' in morph string '{morph_str}'. Use 'd' for dilation and 'e' for erosion.") 
    return voxels

def close_voxels(voxels):
    """Closes the voxels by performing a dilation followed by an erosion."""
    return morph_voxels(voxels, morph_str='de')

def voxels_to_universe(
    arr: np.array,
    scale: float = 1.0,
    place_in_center: bool = True,
    universe: mda.Universe = None,
    nodes: list = None,
) -> mda.Universe:
    """
    Coverts the (selected) nodes in the voxel array into an MDA.Universe.

    Parameters
    ----------
    arr : np.array(int)
        A labeled voxel array (int or bool).
    scale : float
        The scaling applied to the voxel indices to convert them to positions.
    universe : MDAnalysis.Universe
        A universe to inherit scaling from based on its dimensions.
    nodes : list(int)
        A (sub)selection of nodes to add to the created universe.

    Returns
    -------
    universe : MDAnalysis.Universe
        The voxel positions as atoms in a universe with their node IDs as atom names.
    """
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
    
    return u

