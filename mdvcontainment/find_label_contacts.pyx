# find_label_contacts.pyx
cimport cython
cimport numpy as cnp

import numpy as np

def find_label_contacts(cnp.ndarray[cnp.int32_t, ndim=3] labeled_grid):
    """Find which labels are in contact with each other without periodic boundary conditions."""
    cdef cnp.int32_t[:, :, :] labeled_grid_view = labeled_grid
    cdef cnp.int32_t shifts[13][3]

    # Initialize the shifts array
    shifts[0]  = (-1, -1, -1)
    shifts[1]  = (-1, -1,  0)
    shifts[2]  = (-1, -1,  1)
    shifts[3]  = (-1,  0, -1)
    shifts[4]  = (-1,  0,  0)
    shifts[5]  = (-1,  0,  1)
    shifts[6]  = (-1,  1, -1)
    shifts[7]  = (-1,  1,  0)
    shifts[8]  = (-1,  1,  1)
    shifts[9]  = ( 0, -1, -1)
    shifts[10] = ( 0, -1,  0)
    shifts[11] = ( 0, -1,  1)
    shifts[12] = ( 0,  0, -1)

    # Initate the set of all contacts to make them unique.
    cdef set contacts = set()
    
    # Set bounds. 
    cdef int x_max = labeled_grid.shape[0]
    cdef int y_max = labeled_grid.shape[1]
    cdef int z_max = labeled_grid.shape[2]

    cdef int x, y, z, shift_x, shift_y, shift_z, nx, ny, nz
    cdef int current_label, neighbor_label

    # Iteratare over every voxel.
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                current_label = labeled_grid_view[x, y, z]
                if current_label == 0:
                    continue
                # Check the neighbors of the voxel.
                for shift_x, shift_y, shift_z in shifts:
                    nx = x + shift_x
                    ny = y + shift_y
                    nz = z + shift_z
                    # Only check in bound neighbors.
                    if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                        neighbor_label = labeled_grid_view[nx, ny, nz]
                        # Only add non-zero and non-self neighbors.
                        if neighbor_label != 0 and neighbor_label != current_label:
                            contacts.add(tuple(sorted((current_label, neighbor_label))))

    return np.array(sorted(contacts))

