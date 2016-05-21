try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np

from math3.funcs import rect


class test_rectangle(unittest.TestCase):
    def test_import(self):
        import math3
        math3.funcs.rectfunc

    def test_create(self):
        result = rect.create()
        np.testing.assert_almost_equal(result, [[0,0],[1,1]], decimal=5)

    def test_create_dtype(self):
        result = rect.create(dtype=np.float)
        np.testing.assert_almost_equal(result, [[0.,0.],[1.,1.]], decimal=5)

    def test_create_zeros(self):
        result = rect.create_zeros()
        np.testing.assert_almost_equal(result, [[0,0],[0,0]], decimal=5)

    def test_create_from_bounds(self):
        result = rect.create_from_bounds(-1, 1, -2, 2)
        np.testing.assert_almost_equal(result, [[-1,-2],[2,4]], decimal=5)

    def test_bounds(self):
        rect = rect.create_from_bounds(-1, 1, -2, 2)
        result = rect.bounds(rect)
        np.testing.assert_almost_equal(result, (-1,1,-2,2), decimal=5)

    def test_scale_by_vector(self):
        result = rect.scale_by_vector([[-1., -2.], [2., 4.]], [2., 3.])
        np.testing.assert_almost_equal(result, [[-2.,-6.],[4.,12.]], decimal=5)

    def test_scale_by_vector3(self):
        result = rect.scale_by_vector([[-1., -2.], [2., 4.]], [2., 3., 4.])
        np.testing.assert_almost_equal(result, [[-2.,-6.],[4.,12.]], decimal=5)

    def test_right(self):
        result = rect.right([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 4., decimal=5)

    def test_right_negative(self):
        result = rect.right([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 1., decimal=5)

    def test_left(self):
        result = rect.left([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 1., decimal=5)

    def test_left_negative(self):
        result = rect.left([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, -2., decimal=5)

    def test_top(self):
        result = rect.top([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 6., decimal=5)

    def test_top_negative(self):
        result = rect.top([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 2., decimal=5)

    def test_bottom(self):
        result = rect.bottom([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 2., decimal=5)

    def test_bottom_negative(self):
        result = rect.bottom([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, -2., decimal=5)

    def test_x(self):
        result = rect.x([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 1., decimal=5)

    def test_x_negative(self):
        result = rect.x([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 1., decimal=5)

    def test_y(self):
        result = rect.y([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 2., decimal=5)

    def test_y_negative(self):
        result = rect.y([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 2., decimal=5)

    def test_width(self):
        result = rect.width([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 3., decimal=5)

    def test_width_negative(self):
        result = rect.width([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, -3., decimal=5)

    def test_height(self):
        result = rect.height([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 4., decimal=5)

    def test_height_negative(self):
        result = rect.height([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, -4., decimal=5)

    def test_abs_height(self):
        result = rect.abs_height([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 4., decimal=5)

    def test_abs_height_negative(self):
        result = rect.abs_height([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 4., decimal=5)

    def test_abs_width(self):
        result = rect.abs_width([[1., 2.], [3., 4.]])
        np.testing.assert_almost_equal(result, 3., decimal=5)

    def test_abs_width_negative(self):
        result = rect.abs_width([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, 3., decimal=5)

    def test_position(self):
        result = rect.position([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, [1.,2.], decimal=5)

    def test_size(self):
        result = rect.size([[1., 2.], [-3., -4.]])
        np.testing.assert_almost_equal(result, [-3.,-4.], decimal=5)

    
if __name__ == '__main__':
    unittest.main()
