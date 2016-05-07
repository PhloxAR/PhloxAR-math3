try:
    import unittest2 as unittest
except:
    import unittest
from math3 import integer

class test_integer(unittest.TestCase):
    def test_import(self):
        import math3
        math3.integer
        from math3 import integer

    def test_count_bits(self):
        i = 0b010101
        self.assertEqual(integer.count_bits(i), 3)

if __name__ == '__main__':
    unittest.main()
