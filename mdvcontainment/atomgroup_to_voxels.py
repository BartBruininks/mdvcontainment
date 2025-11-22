import numpy as np
import networkx as nx

import MDAnalysis as mda
import collections

from .containment_main import VoxelContainment
from .voxels_to_gro import voxels_to_gro

# Adding the MDA wrapper functionalities

# Generating the PBC in the right format MDA->Matrix
def dim2lattice(x, y, z, alpha=90, beta=90, gamma=90):
    """Convert dimensions (lengths/angles) to lattice matrix"""
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

def voxelate_atomgroup(atomgroup, resolution, hyperres=False, max_offset=0.05):
    check = max_offset == True
    resolution = abs(resolution)

    box = dim2lattice(*atomgroup.dimensions)
    # The 10 is for going from nm to Angstrom
    nbox = (box / (10 * resolution)).round().astype(int) # boxels
    unit = np.linalg.inv(nbox) @ box                     # voxel shape
    error = unit - 10 * resolution * np.eye(3)           # error: deviation from cubic
    deviation = (0.1 * (unit**2).sum(axis=1)**0.5 - resolution) / resolution

    # don't check if told so through a negative resolution 
    if check and (np.abs(deviation) > max_offset).any():
        raise ValueError(
            'A scaling artifact has occured of more than {}% '
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


def gen_explicit_matrix(atomgroup, resolution=1, hyperres=False, max_offset=0.05, return_mapping=True):
    """
    Takes an atomgroup and bins it as close to the resolution as
    possible. If the offset of the actual resolution in at least one
    dimension is more than by default 5%, the function will stop and
    return an error specifying the actual offset in all dimensions
    plus the frame in which the mapping error occured.
    
    Returns
    (array) 3d boolean with True for occupied bins
    (dictionary) atom2voxel mapping

    """
    voxels, nbox = voxelate_atomgroup(atomgroup, resolution, hyperres, max_offset=max_offset)
    # Using np.unique gives a small performance hit.
    # Might still be necessary with large coordinate sets?
    # unique = np.unique(voxels, axis=0)
    x, y, z = voxels.T
    explicit = np.zeros(np.diagonal(nbox), dtype=bool)
    explicit[x, y, z] = True

    # generating the mapping dictionary
    voxel2atom = collections.defaultdict(list)
    # atom index starts from 0 here and is the index in the array, not the
    #  selection atom index in atom_select (these start from 1)
    if hyperres:
        linear_blur(explicit, nbox, 1, inplace=True)
        indices = atomgroup.ix
        #indices = np.repeat(atomgroup.ix, 27)
    else:
        indices = atomgroup.ix
    
    if return_mapping:
        for idx, voxel in zip(indices, voxels):
            voxel2atom[tuple(voxel)].append(idx)
        return explicit, voxel2atom, nbox
    else:
        return explicit, None, nbox

def create_voxels(atomgroup, resolution, hyperres=False, max_offset=0.05, return_mapping=True):
    """
    Returns 3 voxel objects:
    
    all_voxels, voxels, inv_voxels. Where voxels contain all the atoms in the atomgroup, all_voxels
    contains all atoms in the universe belonging to the atomgroup. Inv_voxels contains all atoms which 
    are in the universe of the atomgroup, but not in the atomgroup.
    """
    # Getting the all voxel mask for later indexin when we want to get void particles.
    grid, voxel2atom, nbox = gen_explicit_matrix(
            atomgroup, resolution, hyperres, max_offset, return_mapping)

    return grid, voxel2atom, nbox

def close_voxels(voxels, nbox):
    """
    Dilates and erodes once in place (closure).
    """
    # Possible dilation and erosion to remove small holes for CG data.
    voxels = linear_blur(voxels, nbox, 1)
    voxels = linear_blur(~voxels, nbox, 1)
    voxels = ~voxels
    return voxels

def voxels2atomgroup(voxels, voxel2atom, atomgroup):
    """
    Converts the voxels in a voxel list back to an atomgroup.
    
    Takes a voxel list and uses the voxel2atom mapping with respect to the
    atomgroup.universe to generate a corresponding atomgroup with the voxel 
    list. This is the inverse of gen_explicit_matrix.
    
    Returns an atomgroup.
    """
    # It is not important that every index only occurs onec,
    # as long as each atom is only selected once.
    indices = { idx for v in voxels for idx in voxel2atom[tuple(v)] }
    return atomgroup.universe.atoms[list(indices)]
