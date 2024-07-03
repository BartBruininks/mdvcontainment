import numpy as np
cimport numpy as cnp

def pbc_wrap(cnp.ndarray[cnp.int32_t, ndim=1] positions, cnp.ndarray[cnp.float64_t, ndim=2] nbox):
    """
    Returns the divider and the indices, wrapped into the triclinic box.
    """
    cdef int i
    cdef int n = positions.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] div = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] mod = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wrapped = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        div[i], mod[i] = divmod(positions[i], nbox[i, i])
    
    wrapped = np.mod(mod - np.dot(div, nbox), nbox.diagonal())
    return np.round(div).astype(np.int32), np.round(wrapped).astype(np.int32)

def find_boundary_voxels(cnp.ndarray[cnp.int32_t, ndim=3] labeled_grid):
    cdef cnp.ndarray[cnp.int32_t, ndim=3] grid = labeled_grid
    
    # Find boundary voxels at x=0.
    relevant_voxels_x_indices = np.nonzero(grid[0, :, :])
    relevant_voxels_x = np.column_stack((np.zeros(relevant_voxels_x_indices[0].shape, dtype=np.int32), relevant_voxels_x_indices[0], relevant_voxels_x_indices[1]))

    # Find boundary voxels at y=0.
    relevant_voxels_y_indices = np.nonzero(grid[:, 0, :])
    relevant_voxels_y = np.column_stack((relevant_voxels_y_indices[0], np.zeros(relevant_voxels_y_indices[0].shape, dtype=np.int32), relevant_voxels_y_indices[1]))

    # Find boundary voxels at z=0.
    relevant_voxels_z_indices = np.nonzero(grid[:, :, 0])
    relevant_voxels_z = np.column_stack((relevant_voxels_z_indices[0], relevant_voxels_z_indices[1], np.zeros(relevant_voxels_z_indices[0].shape, dtype=np.int32)))

    return relevant_voxels_x, relevant_voxels_y, relevant_voxels_z

def find_bridges(labeled_grid, nbox):
    """
    Returns which labels are in contact with each other and the accompanying shift, considering periodic boundary conditions.
    """
    cdef int shifts[9][3]
    cdef int x_shifts[9][3]
    cdef int y_shifts[9][3]
    cdef int z_shifts[9][3]
    # dim 0 shifts (x)
    x_shifts[0] = (-1, -1, -1)
    x_shifts[1] = (-1, -1,  0)
    x_shifts[2] = (-1, -1,  1)
    x_shifts[3] = (-1,  0, -1)
    x_shifts[4] = (-1,  0,  0)
    x_shifts[5] = (-1,  0,  1)
    x_shifts[6] = (-1,  1, -1)
    x_shifts[7] = (-1,  1,  0)
    x_shifts[8] = (-1,  1,  1)
    #  dim 1 shifts (y)
    y_shifts[0] = (-1, -1, -1)
    y_shifts[1] = (-1, -1,  0)
    y_shifts[2] = (-1, -1,  1)
    y_shifts[3] = ( 0, -1, -1)
    y_shifts[4] = ( 0, -1,  0)
    y_shifts[5] = ( 0, -1,  1)
    y_shifts[6] = ( 1, -1, -1)
    y_shifts[7] = ( 1, -1,  0)
    y_shifts[8] = ( 1, -1,  1)
    # dim 2 shifts (z)
    z_shifts[0] = (-1, -1, -1)
    z_shifts[1] = (-1,  0, -1)
    z_shifts[2] = (-1,  1, -1)
    z_shifts[3] = ( 0, -1, -1)
    z_shifts[4] = ( 0,  0, -1)
    z_shifts[5] = ( 0,  1, -1)
    z_shifts[6] = ( 1, -1, -1)
    z_shifts[7] = ( 1,  0, -1)
    z_shifts[8] = ( 1,  1, -1)
    
    cdef set contacts = set()
    cdef int x_max = labeled_grid.shape[0]
    cdef int y_max = labeled_grid.shape[1]
    cdef int z_max = labeled_grid.shape[2]

    cdef int x, y, z, shift_x, shift_y, shift_z, nx, ny, nz
    cdef int divx, divy, divz
    cdef int current_label, neighbor_label
    cdef cnp.ndarray[cnp.int_t, ndim=1] positions
    cdef cnp.ndarray[cnp.int_t, ndim=2] relevant_voxels

    relevant_voxels_xyz = find_boundary_voxels(labeled_grid)

    # Check the boundaries and apply periodic conditions.
    for idx, relevant_voxels in enumerate(relevant_voxels_xyz):
        for voxel in relevant_voxels:
            x, y, z = voxel
            current_label = labeled_grid[x, y, z]
            # Set the correct shifts for the current dimension.
            if idx == 0:
                shifts = x_shifts
            elif idx == 1:
                shifts = y_shifts
            else:
                shifts = z_shifts
            
            for shift in shifts:
                # Add the local shift to the boundary voxel.
                nx, ny, nz = x + shift[0], y + shift[1], z + shift[2]
                # Get the wrapped indices for the neighbor and the corresponding box shifts.
                div, (wrapped_nx, wrapped_ny, wrapped_nz) = pbc_wrap(np.array([nx, ny, nz], dtype=np.int32), nbox)
                # Add the contact if at least one boundary is crossed.
                if not np.all(div == 0):
                    neighbor_label = labeled_grid[wrapped_nx, wrapped_ny, wrapped_nz]
                    contacts.add((current_label, neighbor_label, div[0], div[1], div[2]))


    return np.array(sorted(contacts))

