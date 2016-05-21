try:
    import unittest2 as unittest
except:
    import unittest
from math3.funcs import trig


class test_trig(unittest.TestCase):
    def test_import(self):
        import math3
        math3.funcs.trigfunc

    def test_aspec_ratio(self):
        self.assertEqual(trig.aspect_ratio(1920, 1080), 1920. / 1080.)

    @unittest.skip('Need a test here')
    def test_calculate_fov(self):
        pass

    @unittest.skip('Need a test here')
    def test_calculate_zoom(self):
        pass

    @unittest.skip('Need a test here')
    def test_calculate_height(self):
        pass

    @unittest.skip('Need a test here')
    def test_calculate_plane_size(self):
        pass

if __name__ == '__main__':
    unittest.main()
