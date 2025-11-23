import numpy as np
cimport numpy as cnp

def find_bridges(cnp.ndarray[cnp.int32_t, ndim=3] labeled_grid):
    """
    Returns the bridges between the segments over the boundaries.
    """
    cdef int x_max = labeled_grid.shape[0]
    cdef int y_max = labeled_grid.shape[1]
    cdef int z_max = labeled_grid.shape[2]
    
    cdef set contacts = set()
    cdef int x, y, z, nx, ny, nz
    cdef int shift_x, shift_y, shift_z
    cdef cnp.int32_t current_label, neighbor_label
    cdef int dx, dy, dz
    cdef bint is_boundary_x, is_boundary_y, is_boundary_z
    cdef bint crosses_boundary
    
    # Single pass through all voxels, only process boundary voxels
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                current_label = labeled_grid[x, y, z]
                if current_label == 0:
                    continue
                
                # Determine if this voxel is on any boundary
                is_boundary_x = (x == 0 or x == x_max - 1)
                is_boundary_y = (y == 0 or y == y_max - 1)
                is_boundary_z = (z == 0 or z == z_max - 1)
                
                # Skip if not on any boundary
                if not (is_boundary_x or is_boundary_y or is_boundary_z):
                    continue
                
                # Check all 26 neighbors efficiently
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            
                            nx = x + dx
                            ny = y + dy  
                            nz = z + dz
                            
                            # Fast boundary checking and wrapping
                            shift_x = shift_y = shift_z = 0
                            crosses_boundary = False
                            
                            if nx < 0:
                                nx = x_max - 1
                                shift_x = -1
                                crosses_boundary = True
                            elif nx >= x_max:
                                nx = 0
                                shift_x = 1
                                crosses_boundary = True
                            
                            if ny < 0:
                                ny = y_max - 1
                                shift_y = -1
                                crosses_boundary = True
                            elif ny >= y_max:
                                ny = 0
                                shift_y = 1
                                crosses_boundary = True
                                
                            if nz < 0:
                                nz = z_max - 1
                                shift_z = -1
                                crosses_boundary = True
                            elif nz >= z_max:
                                nz = 0
                                shift_z = 1
                                crosses_boundary = True
                            
                            # Only process if we actually crossed a boundary
                            if crosses_boundary:
                                neighbor_label = labeled_grid[nx, ny, nz]
                                if neighbor_label != 0:
                                    # Order labels to avoid duplicates
                                    if current_label <= neighbor_label:
                                        contacts.add((current_label, neighbor_label, shift_x, shift_y, shift_z))
                                    else:
                                        contacts.add((neighbor_label, current_label, -shift_x, -shift_y, -shift_z))
    
    return np.array(sorted(contacts), dtype=np.int32)

