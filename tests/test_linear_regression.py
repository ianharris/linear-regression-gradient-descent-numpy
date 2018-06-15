from lr import cost
from lr import hypothesis

import numpy as np
import unittest

class LinearRegressionTest(unittest.TestCase):

    def test_cost(self):
        
        c = cost(np.matrix([[1], [1], [2]]), \
            np.matrix([[1, 1], [2, 2], [3, 3]]), \
            np.matrix([[1],[1],[1]]))

        self.assertEqual(c, 63)

    def test_hypothesis(self):

        h = hypothesis(np.matrix([[1], [1], [2]]), np.matrix([[1, 1], [2, 2], [3, 3]]))

        np.testing.assert_array_equal(h, np.matrix([[4], [7], [10]]))
