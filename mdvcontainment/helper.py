# Python External
import numpy as np


def convert_bounding_boxed_universe(universe, trim=20):
    """
    Translate the universe.atoms.positions to start from (0,0,0)
    and adds the bounding box as PBC under universe.dimension.

    The operations are in place.

    This is useful to process slices from larger wholes
    and turn them into valid slab universes.
    """
    u = universe
    # Make sure our slab is placed in a bounding box.
    min_position = np.min(u.atoms.positions, axis=0)
    max_position = np.max(u.atoms.positions, axis=0) - min_position
    # Find slice dimension
    sorted_dims = np.argsort(u.dimensions) # smallest is last
    trim_array = np.zeros(3)
    trim_array[sorted_dims:2]  = trim
    shift = min_position - np.array(trim_array)
    u.atoms.positions -= shift
    
    # Set the correct bounding box with a little bit of trim
    u.dimensions = max_position + np.array(trim_array*2)
    return u
