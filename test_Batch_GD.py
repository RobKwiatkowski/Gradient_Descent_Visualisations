import numpy as np
from unittest import TestCase

import Batch_GD


class TestMe(TestCase):

    def test_learning_rate_zero(self):
        rng = np.random.RandomState(1)
        x = 5 * rng.rand(100, 1)
        y = 6 + 2 * x + np.random.randn(100, 1)
        result = Batch_GD.batch_gradient_descent(x, y, 2, 10, 0, 300, 0.001)

        self.assertFalse(result)
