import unittest
import numpy as np
from .find_bridges import find_bridges


class TestPeriodicContacts(unittest.TestCase):
    
    def test_simple_case(self):
        labeled_grid = np.array([
            [[1, 0, 2], [0, 0, 0], [3, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[4, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        expected_contacts = np.array([
            [ 1,  2,  0,  0, -1],
            [ 1,  3,  0, -1,  0],
            [ 1,  4, -1,  0,  0],
            [ 2,  3,  0, -1,  1],
            [ 2,  4, -1,  0,  1],
            [ 3,  4, -1,  1,  0],
        ])


        contacts = find_bridges(labeled_grid)
        np.testing.assert_array_equal(contacts, expected_contacts)

    def test_triclinic_case(self):
        labeled_grid = np.zeros((4,4,4))
        labeled_grid[0,0,0] = 1
        labeled_grid[-1,-1,-1] = 2
        labeled_grid[-1,-1,-2] = 3

        expected_contacts = np.array([
            (1, 3, -1, -1, 0),
        ])

        contacts = find_bridges(labeled_grid)
        np.testing.assert_array_equal(contacts, expected_contacts)

if __name__ == "__main__":
    unittest.main()

