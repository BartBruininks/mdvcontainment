import cython
import numpy as np
cimport numpy as cnp

from libc.stdint cimport int32_t, int64_t


# --- hash function (pure C) ---
cdef inline int64_t hash_voxel(int32_t x, int32_t y, int32_t z) nogil:
    cdef int64_t h = x
    h = h * 73856093
    h ^= y * 19349663
    h ^= z * 83492791
    return h


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def create_efficient_mapping_cy(
    cnp.ndarray[int32_t, ndim=2] voxels,
    cnp.ndarray[int64_t, ndim=1] atom_indices
):
    cdef:
        int n_atoms = voxels.shape[0]
        int i
        int32_t x, y, z
        int64_t h
        dict hash_to_atoms = {}
        dict hash_to_coords = {}

    for i in range(n_atoms):
        x = voxels[i, 0]
        y = voxels[i, 1]
        z = voxels[i, 2]

        h = hash_voxel(x, y, z)

        # append index to Python list
        if h not in hash_to_atoms:
            hash_to_atoms[h] = [i]
            hash_to_coords[h] = (x, y, z)
        else:
            hash_to_atoms[h].append(i)

    # convert python lists to numpy arrays
    voxel_to_atoms = {}
    for h, coords in hash_to_coords.items():
        arr = np.array(hash_to_atoms[h], dtype=np.int32)
        voxel_to_atoms[coords] = arr

    return {
        "atom_voxels": voxels.copy(),
        "voxel_to_atoms": voxel_to_atoms,
        "atom_indices": atom_indices
    }


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def voxels2atomgroup_cy(
    cnp.ndarray[int32_t, ndim=2] voxels_query,
    dict voxel_to_atoms,
    cnp.ndarray[int64_t, ndim=1] atom_indices
):
    cdef:
        int n_query = voxels_query.shape[0]
        int i, j
        int32_t x, y, z
        tuple key
        list out_positions = []
        cnp.ndarray[int32_t, ndim=1] pos_arr

    for i in range(n_query):
        x = voxels_query[i, 0]
        y = voxels_query[i, 1]
        z = voxels_query[i, 2]

        key = (x, y, z)

        if key in voxel_to_atoms:
            pos_arr = voxel_to_atoms[key]
            for j in range(pos_arr.shape[0]):
                out_positions.append(pos_arr[j])

    if not out_positions:
        return np.array([], dtype=np.int64)

    pos_arr = np.array(out_positions, dtype=np.int32)
    return atom_indices[pos_arr]