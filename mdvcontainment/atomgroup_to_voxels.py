import numpy as np

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

def create_voxels(atomgroup, resolution, max_offset=0.05, return_mapping=True):
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
    return_mapping: bool, optional
        Whether to return the mapping from voxels to atoms. Default is True.
    
    Returns
    -------
    voxels: boolean 3D array of voxel occupancy
    atom2voxel: tuple of (atom_voxels, atom_indices) if return_mapping is True, else None
        atom_voxels: (N, 3) array of voxel coordinates per atom
        atom_indices: (N,) array of atom indices corresponding to atom_voxels
    """
    voxels, nbox = _voxelate_atomgroup(atomgroup, resolution, max_offset=max_offset)
    
    x, y, z = voxels.T
    explicit = np.zeros(np.diagonal(nbox), dtype=bool)
    explicit[x, y, z] = True
    
    if return_mapping:
        # Store atom indices directly with voxels for faster lookup
        atom_voxels = voxels.copy()  # Keep the voxel coords per atom
        atom_indices = atomgroup.ix
        return explicit, (atom_voxels, atom_indices)
    else:
        return explicit, None

def close_voxels(voxels):
    """
    Dilates and erodes once in place (closing).

    Parameters
    ----------
    voxels: boolean 3D array of voxel occupancy

    Returns
    -------
    voxels: boolean 3D array of voxel occupancy after closure
    """
    nbox = np.diag(voxels.shape)
    # Possible dilation and erosion to remove small holes for CG data.
    voxels = linear_blur(voxels, nbox, 1)
    voxels = linear_blur(~voxels, nbox, 1)
    voxels = ~voxels
    return voxels

def voxels2atomgroup(voxels, mapping, atomgroup):
    """
    Returns an atomgroup.

    Fast vectorized version using boolean masking.
    mapping is now (atom_voxels, atom_indices) from create_voxels.

    Parameters
    ----------
    voxels: array-like of shape (M, 3)
        Voxel positions to convert back to atoms.
    mapping: tuple of (atom_voxels, atom_indices)
        atom_voxels: (N, 3) array of voxel coordinates per atom
        atom_indices: (N,) array of atom indices corresponding to atom_voxels
    atomgroup: MDAnalysis AtomGroup
        The original atomgroup from which the mapping was created.

    Returns
    -------
    atomgroup: MDAnalysis AtomGroup
        The atomgroup corresponding to the provided voxel positions.
    """
    atom_voxels, atom_indices = mapping
    voxels_array = np.array(voxels)
    
    # Create boolean mask: which atoms are in any of the target voxels?
    # This uses broadcasting to compare all atoms against all target voxels
    mask = (atom_voxels[:, None] == voxels_array[None, :]).all(axis=2).any(axis=1)
    
    selected_indices = atom_indices[mask]
    return atomgroup.universe.atoms[selected_indices]
