# These functions are not used, but show what the intent is for the
#  cython codes.

# The triple for loop appraoch is the most intuitive, but is rather slow in pure python. This is
#  written in cython to make it faster (25x). But could be rewritten in any faster form.
def find_contacting_labels_forloop(labeled_grid):
    """Find which labels are in contact with each other without periodic boundary conditions."""
    shifts = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
        (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
        ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
        ( 0,  0, -1),
    ]

    contacts = set()
    x_max, y_max, z_max = labeled_grid.shape

    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                current_label = labeled_grid[x, y, z]
                if current_label == 0:
                    continue
                for shift in shifts:
                    nx, ny, nz = x + shift[0], y + shift[1], z + shift[2]
                    if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                        neighbor_label = labeled_grid[nx, ny, nz]
                        if neighbor_label != 0 and neighbor_label != current_label:
                            contacts.add(tuple(sorted((current_label, neighbor_label))))

    return np.array(sorted(contacts))

# These functions have been written to cython to make them faster.
def pbc_wrap(positions, nbox):
    """
    Returns the divider and the indices, wrapped into the triclinic box.
    """
    div, mod = np.divmod(positions, nbox.diagonal())
    return div, np.mod(mod - div @ nbox, nbox.diagonal())


def find_boundary_voxels(labeled_grid):
    # Find boundary voxels at x=0.
    relevant_voxels_x_indices = np.nonzero(labeled_grid[0, :, :])
    # Stack zeros at the x position.
    relevant_voxels_x = np.column_stack((np.zeros(relevant_voxels_x_indices[0].shape, dtype=int), relevant_voxels_x_indices[0], relevant_voxels_x_indices[1]))

    # Find boundary voxels at y=0.
    relevant_voxels_y_indices = np.nonzero(labeled_grid[:, 0, :])
    # Stack zeros at the y position.
    relevant_voxels_y = np.column_stack((relevant_voxels_y_indices[0], np.zeros(relevant_voxels_y_indices[0].shape, dtype=int), relevant_voxels_y_indices[1]))

    # Find boundary voxels at z=0.
    relevant_voxels_z_indices = np.nonzero(labeled_grid[:, :, 0])
    # Stack zeros at the z positoin.
    relevant_voxels_z = np.column_stack((relevant_voxels_z_indices[0], relevant_voxels_z_indices[1], np.zeros(relevant_voxels_z_indices[0].shape, dtype=int)))

    return np.vstack([relevant_voxels_x, relevant_voxels_y, relevant_voxels_z])


def find_bridges(labeled_grid, nbox):
    """
    Returns which labels are in contact with each other and the accompanying shift, considering periodic boundary conditions.
    """
    shifts = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1,  0, -1), (-1,  0, 0), (-1,  0, 1),
        (-1,  1, -1), (-1,  1, 0), (-1,  1, 1),
        ( 0, -1, -1), ( 0, -1, 0), ( 0, -1, 1),
        ( 0,  0, -1),
    ]

    contacts = set()
    x_max, y_max, z_max = labeled_grid.shape

    # Get all nonzero voxels with a 0 in their index (onesided oundary; x, y, and z).
    relevant_voxels = find_boundary_voxels(labeled_grid)

    # Check the boundaries and apply periodic conditions.
    for x, y, z in relevant_voxels:
        current_label = labeled_grid[x, y, z]
        for shift in shifts:
            # Add the local shift to the boundary voxel.
            nx, ny, nz = x + shift[0], y + shift[1], z + shift[2]
            # Get the wrapped indices for the neighbor and the corresponding box shifts.
            div, (wrapped_nx, wrapped_ny, wrapped_nz) = pbc_wrap(np.array([nx, ny, nz]), nbox)
            # Add the contact if at least one boundary is crossed.
            if not np.all(div == 0):
                neighbor_label = labeled_grid[wrapped_nx, wrapped_ny, wrapped_nz]
                contacts.add((current_label, neighbor_label, div[0], div[1], div[2]))

    return np.array(sorted(contacts))
