import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .loss import bistable_loss, permute_matrix_loss


class TestLoss(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")

    def test_bistable_loss(self):
        x = torch.tensor([0.0, 0.5, 1.0], device=self.device, requires_grad=True)
        loss = bistable_loss(x)

        # Test expected values
        expected = torch.tensor([0.0, 0.0625, 0.0], device=self.device)
        npt.assert_almost_equal(
            loss.detach().cpu().numpy(), expected.cpu().numpy(), decimal=4
        )

        # Test gradients
        loss.sum().backward()
        # Bistable loss gradient: d/dx[x^2(x-1)^2] = 2x(x-1)^2 + x^2*2(x-1) = 2x(x-1)(2x-1)
        # For x=[0, 0.5, 1]: gradients are [0, 0, 0] (critical points)
        expected_grad = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        npt.assert_almost_equal(
            x.grad.detach().cpu().numpy(), expected_grad.cpu().numpy(), decimal=4
        )

    def test_permute_matrix_loss(self):
        # Test with identity matrix (should have low loss)
        P = torch.eye(3, device=self.device, requires_grad=True)
        loss = permute_matrix_loss(P)

        # Identity should have relatively low loss
        self.assertLessEqual(loss.item(), 1.0)

        # Test gradients
        loss.backward()
        # For identity matrix, gradients should be relatively small
        self.assertEqual(P.grad.shape, (3, 3))
        self.assertTrue(torch.all(torch.abs(P.grad) < 1.0))  # Should be small gradients

    def test_permute_matrix_loss_with_cycle(self):
        # Test with cycle parameters
        P = torch.eye(2, device=self.device, requires_grad=True)
        loss = permute_matrix_loss(P, cycle_length=2, cycle_weight=0.5)

        # Test gradients
        loss.backward()
        # With cycle weight, gradients should still be reasonable
        self.assertEqual(P.grad.shape, (2, 2))
        self.assertTrue(
            torch.all(torch.abs(P.grad) < 5.0)
        )  # Should be reasonable gradients

    def test_gpu_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        x = torch.tensor([0.0, 0.5, 1.0], device=device, requires_grad=True)
        loss = bistable_loss(x)

        self.assertTrue(loss.is_cuda)

        loss.sum().backward()
        self.assertTrue(x.grad.is_cuda)
        # Verify GPU gradients match CPU test
        expected_grad = torch.tensor([0.0, 0.0, 0.0], device=device)
        npt.assert_almost_equal(
            x.grad.detach().cpu().numpy(), expected_grad.cpu().numpy(), decimal=4
        )


if __name__ == "__main__":
    unittest.main()
