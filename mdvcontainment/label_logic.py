"""
Labeling of connected components.
"""

# Python External
import numpy as np
import numpy.typing as npt
from typing import List, Dict
from scipy.ndimage import label


def label_3d_grid(grid: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Label the 3D boolean grid considering 26 neighbors connectivity."""
    # Define the structure for 26 neighbors connectivity
    structure: npt.NDArray[np.int_] = np.ones((3, 3, 3), dtype=int)

    # Label connected components of True values (known typing error in label)
    true_labels, num_true_components = label(grid, structure=structure) # type: ignore[misc]

    # Label connected components of False values (known typing error in label)
    false_labels, num_false_components = label(~grid, structure=structure) # type: ignore[misc]
    false_labels *= -1  # Make false labels negative

    # Combine the labels
    labeled_grid = np.where(grid, true_labels, false_labels)

    return labeled_grid


def create_components_grid(nonp_labeled_grid: npt.NDArray[np.int32], 
                           component2labels: Dict[int, List[int]],
                           ) -> npt.NDArray[np.int32]:
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
