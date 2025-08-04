import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .array_ops import tensor_lookup_2d


class TestTensorLookup2D(TestCase):
    def setUp(self):
        # Check if CUDA is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")

    # python3 -m unittest notebooks_torch.library.array_ops_test.TestTensorLookup2D.test_lookup1
    def test_lookup1(self):
        arr = torch.tensor(
            [
                [[1, 1], [1, 11], [1, 111]],
                [[2, 2], [2, 22], [2, 222]],
                [[3, 3], [3, 33], [3, 333]],
            ],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        x_index = (
            torch.nn.functional.one_hot(torch.tensor(1), 3).float().to(self.device)
        )
        x_index.requires_grad_(True)

        y_index = (
            torch.nn.functional.one_hot(torch.tensor(2), 3).float().to(self.device)
        )
        y_index.requires_grad_(True)

        element = tensor_lookup_2d(arr, x_index, y_index)

        # Compute gradients
        element.sum().backward()
        d_element_arr = arr.grad
        d_element_x_index = x_index.grad
        d_element_y_index = y_index.grad

        self.assertEqual(element.shape, (2,))
        self.assertEqual(d_element_arr.shape, (3, 3, 2))
        self.assertEqual(d_element_x_index.shape, (3,))
        self.assertEqual(d_element_y_index.shape, (3,))

        npt.assert_almost_equal(element.detach().cpu().numpy(), [2, 222])
        
        # Snapshot test for gradients
        expected_arr_grad = torch.zeros((3, 3, 2), device=self.device)
        expected_arr_grad[1, 2, :] = 1.0  # Gradient flows to the selected element
        npt.assert_almost_equal(d_element_arr.detach().cpu().numpy(), expected_arr_grad.cpu().numpy())
        
        expected_x_index_grad = torch.tensor([0.0, 224.0, 0.0], device=self.device)  # Sum of selected row
        npt.assert_almost_equal(d_element_x_index.detach().cpu().numpy(), expected_x_index_grad.cpu().numpy())
        
        expected_y_index_grad = torch.tensor([0.0, 0.0, 224.0], device=self.device)  # Sum of selected column
        npt.assert_almost_equal(d_element_y_index.detach().cpu().numpy(), expected_y_index_grad.cpu().numpy())

    def test_gpu_compatibility(self):
        """Test that operations work on GPU if available"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")

        arr = torch.tensor(
            [
                [[1, 1], [1, 11], [1, 111]],
                [[2, 2], [2, 22], [2, 222]],
                [[3, 3], [3, 33], [3, 333]],
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        x_index = torch.nn.functional.one_hot(torch.tensor(1, device=device), 3).float()
        x_index.requires_grad_(True)

        y_index = torch.nn.functional.one_hot(torch.tensor(2, device=device), 3).float()
        y_index.requires_grad_(True)

        element = tensor_lookup_2d(arr, x_index, y_index)

        # Verify tensors are on GPU
        self.assertTrue(element.is_cuda)
        self.assertTrue(arr.is_cuda)
        self.assertTrue(x_index.is_cuda)
        self.assertTrue(y_index.is_cuda)

        # Compute gradients
        element.sum().backward()

        # Verify gradients are on GPU
        self.assertTrue(arr.grad.is_cuda)
        self.assertTrue(x_index.grad.is_cuda)
        self.assertTrue(y_index.grad.is_cuda)

        npt.assert_almost_equal(element.detach().cpu().numpy(), [2, 222])
        
        # Snapshot test for GPU gradients - same values as CPU test
        expected_arr_grad = torch.zeros((3, 3, 2), device=device)
        expected_arr_grad[1, 2, :] = 1.0
        npt.assert_almost_equal(arr.grad.detach().cpu().numpy(), expected_arr_grad.cpu().numpy())
        
        expected_x_index_grad = torch.tensor([0.0, 224.0, 0.0], device=device)
        npt.assert_almost_equal(x_index.grad.detach().cpu().numpy(), expected_x_index_grad.cpu().numpy())
        
        expected_y_index_grad = torch.tensor([0.0, 0.0, 224.0], device=device)
        npt.assert_almost_equal(y_index.grad.detach().cpu().numpy(), expected_y_index_grad.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
