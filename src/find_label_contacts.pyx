cimport cython
cimport numpy as cnp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def find_label_contacts(cnp.int32_t[:, :, :] labeled_grid):
    """Find which labels are in contact with each other without periodic boundary conditions."""
    
    # Pre-compute shifts as a C array for faster access
    cdef cnp.int32_t shifts[13][3]
    shifts[0]  = [-1, -1, -1]
    shifts[1]  = [-1, -1,  0]
    shifts[2]  = [-1, -1,  1]
    shifts[3]  = [-1,  0, -1]
    shifts[4]  = [-1,  0,  0]
    shifts[5]  = [-1,  0,  1]
    shifts[6]  = [-1,  1, -1]
    shifts[7]  = [-1,  1,  0]
    shifts[8]  = [-1,  1,  1]
    shifts[9]  = [ 0, -1, -1]
    shifts[10] = [ 0, -1,  0]
    shifts[11] = [ 0, -1,  1]
    shifts[12] = [ 0,  0, -1]
    
    # Use Python set but minimize Python object creation
    cdef set contacts = set()
    
    # Cache dimensions
    cdef cnp.int32_t x_max = labeled_grid.shape[0]
    cdef cnp.int32_t y_max = labeled_grid.shape[1] 
    cdef cnp.int32_t z_max = labeled_grid.shape[2]
    
    # Loop variables
    cdef cnp.int32_t x, y, z, i, nx, ny, nz
    cdef cnp.int32_t current_label, neighbor_label
    cdef cnp.int32_t label1, label2
    cdef tuple contact_tuple
    
    # Main loops - keep Python objects minimal
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                current_label = labeled_grid[x, y, z]
                if current_label == 0:
                    continue
                
                # Check all 13 neighbors
                for i in range(13):
                    nx = x + shifts[i][0]
                    ny = y + shifts[i][1]
                    nz = z + shifts[i][2]
                    
                    # Bounds checking
                    if nx >= 0 and nx < x_max and ny >= 0 and ny < y_max and nz >= 0 and nz < z_max:
                        neighbor_label = labeled_grid[nx, ny, nz]
                        
                        # Only add contacts between different non-zero labels
                        if neighbor_label != 0 and neighbor_label != current_label:
                            # Ensure consistent ordering (smaller label first)
                            if current_label < neighbor_label:
                                label1 = current_label
                                label2 = neighbor_label
                            else:
                                label1 = neighbor_label
                                label2 = current_label
                            
                            # Create tuple once and add to set
                            contact_tuple = (label1, label2)
                            contacts.add(contact_tuple)
    
    # Convert to sorted numpy array
    if contacts:
        contact_list = sorted(contacts)
        return np.array(contact_list, dtype=np.int32)
    else:
        return np.empty((0, 2), dtype=np.int32)

