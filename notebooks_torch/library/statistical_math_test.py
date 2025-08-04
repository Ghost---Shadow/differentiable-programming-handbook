import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .statistical_math import to_prob_dist_all, cross_entropy, entropy


class TestStatisticalMath(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")

    def test_to_prob_dist_all(self):
        v = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, requires_grad=True
        )
        result = to_prob_dist_all(v)

        # Check that each row sums to 1
        row_sums = torch.sum(result, dim=-1)
        npt.assert_almost_equal(row_sums.detach().cpu().numpy(), [1.0, 1.0], decimal=5)

        # Test gradients
        result.sum().backward()
        # Probability distribution gradients should sum to zero (normalization constraint)
        self.assertEqual(v.grad.shape, v.shape)
        row_grad_sums = torch.sum(v.grad, dim=-1)
        npt.assert_almost_equal(row_grad_sums.detach().cpu().numpy(), [0.0, 0.0], decimal=5)

    def test_cross_entropy(self):
        x = torch.tensor(
            [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], device=self.device, requires_grad=True
        )
        y = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], device=self.device)

        result = cross_entropy(x, y)

        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.all(result >= 0))  # Cross-entropy should be non-negative

        # Test gradients
        result.sum().backward()
        # Cross-entropy gradients should be negative of normalized targets
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(torch.any(x.grad != 0))

    def test_entropy(self):
        x = torch.tensor(
            [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], device=self.device, requires_grad=True
        )

        result = entropy(x)

        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.all(result >= 0))  # Entropy should be non-negative

        # Test gradients
        result.sum().backward()
        # Entropy gradients (self cross-entropy)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(torch.any(x.grad != 0))

    def test_gpu_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        v = torch.tensor([[1.0, 2.0, 3.0]], device=device, requires_grad=True)
        result = to_prob_dist_all(v)

        self.assertTrue(result.is_cuda)

        result.sum().backward()
        self.assertTrue(v.grad.is_cuda)
        # Verify GPU gradients have correct properties
        self.assertEqual(v.grad.shape, v.shape)
        grad_sum = torch.sum(v.grad, dim=-1)
        npt.assert_almost_equal(grad_sum.detach().cpu().numpy(), [0.0], decimal=5)


if __name__ == "__main__":
    unittest.main()
