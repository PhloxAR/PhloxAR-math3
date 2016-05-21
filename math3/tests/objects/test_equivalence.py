from __future__ import absolute_import, division, print_function
try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
from math3 import Quaternion, Matrix4, Matrix3, Vector3, Vector4, euler


class test_matrix_quaternion(unittest.TestCase):
    def test_m44_q_equivalence(self):
        """Test for equivalance of matrix and quaternion rotations.

        Create a matrix and quaternion, rotate each by the same values
        then convert matrix<->quaternion and check the results are the same.
        """
        m = Matrix4.from_x_rotation(np.pi / 2.)
        mq = Quaternion.from_matrix(m)

        q = Quaternion.from_x_rotation(np.pi / 2.)
        qm = Matrix4.from_quaternion(q)

        self.assertTrue(np.allclose(np.dot([1., 0., 0., 1.], m), [1., 0., 0., 1.]))
        self.assertTrue(np.allclose(np.dot([1., 0., 0., 1.], qm), [1., 0., 0., 1.]))

        self.assertTrue(np.allclose(q * Vector4([1., 0., 0., 1.]), [1., 0., 0., 1.]))
        self.assertTrue(np.allclose(mq * Vector4([1., 0., 0., 1.]), [1., 0., 0., 1.]))

        np.testing.assert_almost_equal(q, mq, decimal=5)
        np.testing.assert_almost_equal(m, qm, decimal=5)

    def test_euler_equivalence(self):
        eulers = euler.create_from_x_rotation(np.pi / 2.)
        m = Matrix3.from_x_rotation(np.pi / 2.)
        q = Quaternion.from_x_rotation(np.pi / 2.)
        qm = Matrix3.from_quaternion(q)
        em = Matrix3.from_eulers(eulers)
        self.assertTrue(np.allclose(qm, m))
        self.assertTrue(np.allclose(qm, em))
        self.assertTrue(np.allclose(m, em))


if __name__ == '__main__':
    unittest.main()
