"""
Labeling of connected components.
"""

# Python External
import numpy as np
from scipy.ndimage import label


def label_3d_grid(grid):
    """Label the 3D boolean grid considering 26 neighbors connectivity."""
    # Define the structure for 26 neighbors connectivity
    structure = np.ones((3, 3, 3), dtype=int)

    # Label connected components of True values
    true_labels, num_true_components = label(grid, structure=structure)

    # Label connected components of False values
    false_labels, num_false_components = label(~grid, structure=structure)
    false_labels *= -1  # Make false labels negative

    # Combine the labels
    labeled_grid = np.where(grid, true_labels, false_labels)

    return labeled_grid

def create_components_grid(nonp_labeled_grid, component2labels):
    """
    Returns the components grid, which is a relabeling of the
    labeled grid as a copy.

    Returns
    -------
    (3darray, int32) components_grid
    """
    # Create the components_grid (by relabeling the nonp_labeled_grid in a copy)
    # Create a copy of the input array
    components_grid = np.zeros_like(nonp_labeled_grid)

    # Iterate through the mapping dictionary and replace values
    for final_val, current_vals in component2labels.items():
        for val in current_vals:
            components_grid[nonp_labeled_grid == val] = final_val
    return components_grid
