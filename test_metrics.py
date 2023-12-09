import unittest

import numpy as np

from metrics import ap10

class TestAveragePrecisionAt10(unittest.TestCase):


    def test(self):
        grand_truth = np.array([2, 10, 8])
        predictions = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 3個目
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 正解なし
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],  # 9個目
        ])
        expected = np.array([1/3, 0., 1/9])
        output = ap10(grand_truth, predictions)
        self.assertTrue(np.allclose(expected, output), (expected, output))


if __name__ == '__main__':
    unittest.main()