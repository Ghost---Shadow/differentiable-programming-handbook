import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .vector_math import shift_left_one_hot, dot


class TestVectorMath(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")

    def test_shift_left_one_hot(self):
        vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, requires_grad=True)
        result = shift_left_one_hot(vec, shift=-1)

        # Check shape
        self.assertEqual(result.shape, (1, 3))

        # Test gradients
        result.sum().backward()
        # Shift operation should propagate gradients through permutation matrix
        self.assertEqual(vec.grad.shape, vec.shape)
        expected_grad = torch.tensor(
            [1.0, 1.0, 1.0], device=self.device
        )  # All positions contribute
        npt.assert_almost_equal(
            vec.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )

    def test_shift_left_one_hot_different_shift(self):
        vec = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        result1 = shift_left_one_hot(vec, shift=-1)
        result2 = shift_left_one_hot(vec, shift=1)

        # Results should be different for different shifts
        self.assertFalse(torch.allclose(result1, result2))

    def test_dot(self):
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, requires_grad=True
        )
        y = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=self.device, requires_grad=True
        )

        result = dot(x, y)

        # Check expected values: [1*1 + 2*1 + 3*1, 4*2 + 5*2 + 6*2] = [6, 30]
        expected = torch.tensor([6.0, 30.0], device=self.device)
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected.cpu().numpy())

        # Test gradients
        result.sum().backward()
        # Dot product gradients: dx = y, dy = x
        expected_x_grad = y.clone().detach()
        expected_y_grad = x.clone().detach()
        npt.assert_almost_equal(
            x.grad.detach().cpu().numpy(), expected_x_grad.cpu().numpy()
        )
        npt.assert_almost_equal(
            y.grad.detach().cpu().numpy(), expected_y_grad.cpu().numpy()
        )

    def test_dot_single_vector(self):
        x = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        y = torch.tensor([4.0, 5.0, 6.0], device=self.device, requires_grad=True)

        result = dot(x, y)

        # Check expected value: 1*4 + 2*5 + 3*6 = 32
        expected = 32.0
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected)

        # Test gradients
        result.backward()
        # Single vector dot product gradients
        expected_x_grad = y.clone().detach()  # [4, 5, 6]
        expected_y_grad = x.clone().detach()  # [1, 2, 3]
        npt.assert_almost_equal(
            x.grad.detach().cpu().numpy(), expected_x_grad.cpu().numpy()
        )
        npt.assert_almost_equal(
            y.grad.detach().cpu().numpy(), expected_y_grad.cpu().numpy()
        )

    def test_gpu_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        vec = torch.tensor([1.0, 0.0, 0.0], device=device, requires_grad=True)
        result = shift_left_one_hot(vec)

        self.assertTrue(result.is_cuda)

        result.sum().backward()
        self.assertTrue(vec.grad.is_cuda)
        # Verify GPU gradients match CPU test
        expected_grad = torch.tensor([1.0, 1.0, 1.0], device=device)
        npt.assert_almost_equal(
            vec.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )


if __name__ == "__main__":
    unittest.main()
