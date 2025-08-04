import unittest
from unittest import TestCase

import torch
import numpy as np
import numpy.testing as npt

from .stacks import new_stack, new_stack_from_buffer, stack_push, stack_pop, stack_peek


class TestStacks(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {self.device}")

    def test_new_stack(self):
        stack_shape = (5, 3)
        stack = new_stack(stack_shape, device=self.device)
        buffer, index = stack

        self.assertEqual(buffer.shape, stack_shape)
        self.assertEqual(index.shape, (5,))
        self.assertEqual(buffer.device.type, self.device.type)
        self.assertEqual(index.device.type, self.device.type)

    def test_new_stack_from_buffer(self):
        buffer = torch.randn(4, 2, device=self.device)
        stack = new_stack_from_buffer(buffer)
        new_buffer, index = stack

        self.assertEqual(new_buffer.shape, buffer.shape)
        self.assertEqual(index.shape, (4,))
        self.assertTrue(torch.equal(new_buffer, buffer))

    def test_stack_push(self):
        stack = new_stack((3, 2), device=self.device)
        element = torch.tensor([1.0, 2.0], device=self.device)

        pushed_stack = stack_push(stack, element)
        buffer, index = pushed_stack

        self.assertEqual(buffer.shape, (3, 2))
        self.assertEqual(index.shape, (3,))

    def test_stack_pop(self):
        stack = new_stack((3, 2), device=self.device)
        element = torch.tensor([1.0, 2.0], device=self.device)

        # Push then pop
        pushed_stack = stack_push(stack, element)
        popped_stack, popped_element = stack_pop(pushed_stack)

        self.assertEqual(popped_element.shape, (2,))

    def test_stack_peek(self):
        stack = new_stack((3, 2), device=self.device)
        element = torch.tensor([1.0, 2.0], device=self.device)

        # Push then peek
        pushed_stack = stack_push(stack, element)
        peeked_element = stack_peek(pushed_stack)

        self.assertEqual(peeked_element.shape, (2,))

    def test_learnable_stack(self):
        stack = new_stack((3, 2), is_learnable=True, device=self.device)
        buffer, index = stack

        self.assertIsInstance(buffer, torch.nn.Parameter)
        self.assertIsInstance(index, torch.nn.Parameter)

    def test_gpu_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        stack = new_stack((3, 2), device=device)
        element = torch.tensor([1.0, 2.0], device=device)

        buffer, index = stack
        self.assertTrue(buffer.is_cuda)
        self.assertTrue(index.is_cuda)

        pushed_stack = stack_push(stack, element)
        new_buffer, new_index = pushed_stack
        self.assertTrue(new_buffer.is_cuda)
        self.assertTrue(new_index.is_cuda)


if __name__ == "__main__":
    unittest.main()
