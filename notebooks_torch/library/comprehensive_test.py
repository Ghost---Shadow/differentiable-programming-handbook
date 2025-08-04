import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .array_ops import (
    assign_index,
    assign_index_vectored,
    naive_lookup,
    linear_lookup,
    superposition_lookup_vectored,
    asymmetrical_vectored_lookup,
    bandwidthify,
    bulk_bandwidthify,
    superposition_lookup,
    residual_lookup,
    match_shapes,
    broadcast_multiply,
    tensor_lookup_2d,
    tensor_write_2d,
)


class TestComprehensiveArrayOps(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running comprehensive tests on device: {self.device}")

    def test_assign_index(self):
        arr = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        element = torch.tensor(9.0, device=self.device)
        result = assign_index(arr, 1, element)

        expected = torch.tensor([1.0, 9.0, 3.0], device=self.device)
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected.cpu().numpy())

        result.sum().backward()
        expected_grad = torch.tensor(
            [1.0, 0.0, 1.0], device=self.device
        )  # Only unmasked elements get gradient
        npt.assert_almost_equal(
            arr.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )

    def test_assign_index_vectored(self):
        arr = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        index = torch.tensor([0.0, 1.0, 0.0], device=self.device)  # one-hot for index 1
        element = torch.tensor(9.0, device=self.device)
        result = assign_index_vectored(arr, index, element)

        expected = torch.tensor([1.0, 9.0, 3.0], device=self.device)
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected.cpu().numpy())

    def test_naive_lookup(self):
        arr = torch.tensor([10.0, 20.0, 30.0], device=self.device)
        index = torch.tensor(1.2, device=self.device)  # Should round to 1
        result = naive_lookup(arr, index)

        expected = 20.0
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected)

    def test_linear_lookup(self):
        arr = torch.tensor([10.0, 20.0, 30.0], device=self.device, requires_grad=True)
        index = torch.tensor(0.5, device=self.device)  # Between 0 and 1
        result = linear_lookup(arr, index)

        # Should interpolate between arr[0] and arr[1]
        expected = 15.0  # 0.5 * 10 + 0.5 * 20
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected)

        result.backward()
        expected_grad = torch.tensor(
            [0.5, 0.5, 0.0], device=self.device
        )  # Linear interpolation gradients
        npt.assert_almost_equal(
            arr.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )

    def test_superposition_lookup_vectored(self):
        arr = torch.tensor([10.0, 20.0, 30.0], device=self.device, requires_grad=True)
        indices = torch.tensor([0.5, 0.3, 0.2], device=self.device)
        result = superposition_lookup_vectored(arr, indices)

        # Should be a weighted sum
        expected = 0.5 * 10.0 + 0.3 * 20.0 + 0.2 * 30.0
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected)

        result.backward()
        expected_grad = torch.tensor(
            [0.5, 0.3, 0.2], device=self.device
        )  # Gradient equals weights
        npt.assert_almost_equal(
            arr.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )

    def test_asymmetrical_vectored_lookup(self):
        v = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, requires_grad=True
        )
        k = torch.tensor(
            [[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], device=self.device, requires_grad=True
        )
        result = asymmetrical_vectored_lookup(v, k)

        self.assertEqual(result.shape, (2,))

        result.sum().backward()
        # For asymmetrical lookup, gradients depend on the specific backward implementation
        # Just verify they have the correct shapes and are non-zero
        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertTrue(torch.any(v.grad != 0))
        self.assertTrue(torch.any(k.grad != 0))

    def test_bandwidthify(self):
        index = torch.tensor(1.5, device=self.device)
        bandwidth = 4
        result = bandwidthify(index, bandwidth)

        self.assertEqual(result.shape, (4,))
        # Should sum to 1 (probability distribution)
        npt.assert_almost_equal(torch.sum(result).detach().cpu().numpy(), 1.0)

    def test_bulk_bandwidthify(self):
        indices = torch.tensor([0.5, 1.5, 2.5], device=self.device)
        bandwidth = 4
        result = bulk_bandwidthify(indices, bandwidth)

        self.assertEqual(result.shape, (3, 4))

    def test_superposition_lookup(self):
        arr = torch.tensor([10.0, 20.0, 30.0], device=self.device, requires_grad=True)
        index = torch.tensor(1.5, device=self.device)
        result = superposition_lookup(arr, index)

        result.backward()
        # Superposition lookup involves bandwidthify which creates a distribution
        # Just verify gradient exists and has correct shape
        self.assertEqual(arr.grad.shape, arr.shape)
        self.assertTrue(torch.any(arr.grad != 0))

    def test_residual_lookup(self):
        arr = torch.tensor([10.0, 20.0, 30.0], device=self.device)
        index = torch.tensor(1.3, device=self.device)
        result, residue = residual_lookup(arr, index)

        expected_result = 20.0  # arr[1]
        expected_residue = 0.3  # 1.3 - 1
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected_result)
        npt.assert_almost_equal(residue.detach().cpu().numpy(), expected_residue)

    def test_match_shapes(self):
        x = torch.tensor([[1, 2]], device=self.device)  # Shape: (1, 2)
        y = torch.tensor([3, 4], device=self.device)  # Shape: (2,)

        high, low = match_shapes(x, y)

        # After matching, both should be broadcastable
        result = high * low
        self.assertEqual(result.shape, (1, 2))

    def test_broadcast_multiply(self):
        x = torch.tensor([[1, 2]], device=self.device)
        y = torch.tensor([3, 4], device=self.device)

        result = broadcast_multiply(x, y)
        expected = torch.tensor([[3, 8]], device=self.device)
        npt.assert_almost_equal(result.detach().cpu().numpy(), expected.cpu().numpy())

    def test_tensor_write_2d(self):
        arr = torch.zeros((2, 2, 3), device=self.device, requires_grad=True)
        element = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        x_index = torch.tensor([0.0, 1.0], device=self.device)  # one-hot for x=1
        y_index = torch.tensor([1.0, 0.0], device=self.device)  # one-hot for y=0

        result = tensor_write_2d(arr, element, x_index, y_index)

        self.assertEqual(result.shape, (2, 2, 3))

        result.sum().backward()
        # Tensor write operation - gradient should flow to all elements
        self.assertEqual(arr.grad.shape, arr.shape)
        self.assertTrue(torch.any(arr.grad != 0))

    def test_gpu_compatibility_comprehensive(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")

        # Test multiple operations on GPU
        arr = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        element = torch.tensor(9.0, device=device)
        result = assign_index(arr, 1, element)

        self.assertTrue(result.is_cuda)

        result.sum().backward()
        self.assertTrue(arr.grad.is_cuda)
        # Verify gradient values match CPU test
        expected_grad = torch.tensor(
            [1.0, 0.0, 1.0], device=device
        )  # Only unmasked elements get gradient
        npt.assert_almost_equal(
            arr.grad.detach().cpu().numpy(), expected_grad.cpu().numpy()
        )


if __name__ == "__main__":
    unittest.main()
