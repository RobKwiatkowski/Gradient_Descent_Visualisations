import numpy as np
from unittest import TestCase

import Batch_GD


class TestMe(TestCase):

    def test_learning_rate_zero(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        result = Batch_GD.batch_gradient_descent(x, y, 2, 10, 0, 300, 0.001)
        self.assertFalse(result)

    def test_empty_x(self):
        x = []
        y = [1, 2]
        result = Batch_GD.batch_gradient_descent(x, y, 0.1, 10, 0, 300, 0.001)
        self.assertFalse(result)

    def test_empty_y(self):
        x = [1, 2]
        y = []
        result = Batch_GD.batch_gradient_descent(x, y, 0.1, 10, 0, 300, 0.001)
        self.assertFalse(result)

    def test_xy_not_match(self):
        x = [1, 2]
        y = [1, 2, 3, 4]
        result = Batch_GD.batch_gradient_descent(x, y, 0.05, 10, 0, 300, 0.001)
        self.assertFalse(result)

    def test_normal_case(self):

        rng = np.random.RandomState(1)
        x = 5 * rng.rand(100, 1)
        y = 6 + 2 * x
        result = Batch_GD.batch_gradient_descent(x, y, 0.05, 0, 0, 2000, 0.001)
        a0 = result[0][0][0]

        self.assertAlmostEqual(a0, 6, places=1)
