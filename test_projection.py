import unittest
import numpy as np
from Ex_1 import project_plane
class TestProjection(unittest.TestCase):

    def test_rectangle_inside_constant_volume(self):
        M = np.full((5, 5, 5), 7)

        a = [1, 1, 2]
        a_second = [3, 1, 2]
        b = [1, 3, 2]
        c = [3, 3, 2]

        E = project_plane(M, a, b, c, a_second)

        self.assertEqual(E.shape, (2, 3))
        np.testing.assert_array_equal(
            E,
            np.full((2, 3), 7)
        )

    def test_rectangle_outside_volume(self):
            M = np.full((5, 5, 5), 7)

            a = [1, 1, 10]
            a_second = [3, 1, 10]
            b = [1, 3, 10]
            c = [3, 3, 10]

            E = project_plane(M, a, b, c, a_second)

            np.testing.assert_array_equal(
                E,
                np.zeros((2, 3))
            )
    def test_rectangle_crossing_boundary(self):
        M = np.full((5, 5, 5), 7)

        a = [-1, 1, 2]
        a_second = [1, 1, 2]
        b = [-1, 3, 2]
        c = [1, 3, 2]

        E = project_plane(M, a, b, c, a_second)

        expected = np.array([
            [0, 7, 7],
            [0, 7, 7]
        ])

        np.testing.assert_array_equal(E, expected)

    def test_rectangle_in_varying_volume(self):
        M = np.zeros((5, 5, 5), dtype=int)

        for x in range(5):
            for y in range(5):
                for z in range(5):
                    M[x, y, z] = x + 10*y + 100*z

        a = [1, 1, 2]
        a_second = [3, 1, 2]
        b = [1, 3, 2]
        c = [3, 3, 2]

        E = project_plane(M, a, b, c, a_second)

        expected = np.array([
            [221, 222, 223],
            [231, 232, 233]
        ])

        np.testing.assert_array_equal(E, expected)

    def test_zero_length_side(self):
        M = np.full((5, 5, 5), 7)

        a = [1, 1, 2]
        b = [1, 1, 2]
        c = [3, 3, 2]

        with self.assertRaises(ValueError):
            project_plane(M, a, b, c)



    def test_unequal_side_lengths(self):
        M = np.full((5, 5, 5), 7)

        a = [1, 1, 2]
        b = [1, 3, 2]
        c = [4, 1, 2]

        with self.assertRaises(ValueError):
            project_plane(M, a, b, c)

    def test_triangle_inside_constant_volume(self):
        M = np.full((8, 8, 8), 7)

        a = [3, 0, 2]
        b = [0, 4, 2]
        c = [6, 4, 2]

        E = project_plane(M, a, b, c)

        self.assertEqual(E.shape, (5, 7))
        self.assertTrue(np.any(E == 7))
        self.assertTrue(np.any(E == 0))
        self.assertTrue(np.all((E == 0) | (E == 7)))

        filled_per_row = np.count_nonzero(E, axis=1)
        self.assertGreater(filled_per_row[-1], filled_per_row[0])




if __name__ == "__main__":
    unittest.main()
