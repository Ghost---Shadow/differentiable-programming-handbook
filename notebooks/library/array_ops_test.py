import unittest
from unittest import TestCase

import tensorflow as tf
import numpy as np
import numpy.testing as npt

from .array_ops import tensor_lookup_2d


class TestTensorLookup2D(TestCase):

    # python3 -m unittest notebooks.library.array_ops_test.TestTensorLookup2D.test_lookup1
    def test_lookup1(self):
        arr = tf.Variable([
            [[1, 1], [1, 11], [1, 111]],
            [[2, 2], [2, 22], [2, 222]],
            [[3, 3], [3, 33], [3, 333]]
        ], dtype=tf.float32)
        x_index = tf.Variable(tf.one_hot(1, 3), dtype=tf.float32)
        y_index = tf.Variable(tf.one_hot(2, 3), dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            element = tensor_lookup_2d(arr, x_index, y_index)

        d_element_arr = tape.gradient(element, arr)
        d_element_x_index = tape.gradient(element, x_index)
        d_element_y_index = tape.gradient(element, y_index)

        self.assertEqual(element.shape, (2,))
        self.assertEqual(d_element_arr.shape, (3, 3, 2))
        self.assertEqual(d_element_x_index.shape, (3, ))
        self.assertEqual(d_element_y_index.shape, (3, ))

        npt.assert_almost_equal(element, [2, 222])


if __name__ == '__main__':
    unittest.main()
