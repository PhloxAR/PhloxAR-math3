try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from math3 import sphere


class test_sphere(unittest.TestCase):
    def test_import(self):
        import math3
        math3.sphere
        from math3 import sphere

    def test_create_from_points(self):
        # the biggest should be 5,5,5
        result = sphere.create_from_points([
            [ 0.0, 0.0, 0.0 ],
            [ 5.0, 5.0, 5.0 ],
            [ 0.0, 0.0, 5.0 ],
            [-5.0, 0.0, 0.0 ],
        ])
        # centred around 0,0,0
        # with MAX LENGTH as radius
        np.testing.assert_almost_equal(result, [0.,0.,0., 8.66025], decimal=5)

    def test_position(self):
        result = sphere.position([1.,2.,3.,4.])
        np.testing.assert_almost_equal(result, [1.,2.,3.], decimal=5)

    def test_radius(self):
        result = sphere.radius([1.,2.,3.,4.])
        np.testing.assert_almost_equal(result, 4., decimal=5)


if __name__ == '__main__':
    unittest.main()
