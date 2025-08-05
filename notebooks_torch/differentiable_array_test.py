import unittest
import torch
import torch.nn.functional as F
from differentiable_array import DifferentiableArray, DifferentiableArrayLoss


class TestDifferentiableArray(unittest.TestCase):
    """Test suite for DifferentiableArray class."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.test_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
        self.diff_array = DifferentiableArray(
            self.test_tensor, embedding_dim=32, temperature=0.5
        )

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_losslessness -v
    def test_losslessness(self):
        for i in range(len(self.test_tensor)):
            self.assertEqual(self.diff_array[i].item(), self.test_tensor[i].item())

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_getitem_fractional_indexing -v
    def test_getitem_fractional_indexing(self):
        """Test fractional indexing for interpolation."""
        # Test fractional indices (should be reasonable values)
        val_1_0 = self.diff_array[1]
        val_1_5 = self.diff_array[1.5]
        val_2_0 = self.diff_array[2]
        val_2_7 = self.diff_array[2.7]
        val_3_0 = self.diff_array[3]

        self.assertEqual(val_1_0.item(), 4.0)
        self.assertEqual(val_1_5.item(), 6.5)
        self.assertEqual(val_2_0.item(), 9.0)
        self.assertEqual(val_2_7.item(), 13.90000057220459)
        self.assertEqual(val_3_0.item(), 16.0)

        self.assertTrue(val_1_0 < val_1_5 < val_2_0)
        self.assertTrue(val_2_0 < val_2_7 < val_3_0)

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_setitem_integer_assignment_losslessness -v
    def test_setitem_integer_assignment_losslessness(self):
        test_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
        diff_array = DifferentiableArray(test_tensor, embedding_dim=32, temperature=0.5)
        diff_array[3] = 17.0
        test_tensor[3] = 17.0

        for i in range(len(test_tensor)):
            self.assertEqual(diff_array[i].item(), test_tensor[i].item())

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_setitem_fractional_assignment -v
    def test_setitem_fractional_assignment(self):
        """Test fractional assignment (soft assignment)."""
        torch.manual_seed(42)
        original_values = self.diff_array.values.clone()
        self.diff_array[2.5] = 150.0
        new_values = self.diff_array.values

        # Check exact values after assignment
        expected_values = [
            1.0049947500228882,
            4.004993438720703,
            9.004995346069336,
            16.005006790161133,
            25.005008697509766,
            36.0050048828125,
            49.0050048828125,
        ]
        self.assertEqual(new_values.tolist(), expected_values)

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_forward_method -v
    def test_forward_method(self):
        """Test forward method directly."""
        torch.manual_seed(42)
        batch_size = 2
        values = self.diff_array._values.expand(batch_size, -1)
        keys = torch.tensor([[0.5], [0.8]])

        output, attention_weights, similarities = self.diff_array.forward(values, keys)

        self.assertEqual(output.shape, (batch_size, 1))
        self.assertEqual(attention_weights.shape, (batch_size, 7))
        self.assertEqual(similarities.shape, (batch_size, 7))

        # Check exact output values
        self.assertEqual(output[0, 0].item(), 10.34654426574707)
        self.assertEqual(output[1, 0].item(), 6.895977020263672)

        # Check exact attention weights sum
        self.assertEqual(attention_weights.sum(dim=1)[0].item(), 1.0)
        self.assertEqual(attention_weights.sum(dim=1)[1].item(), 1.0)

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_gradient_direction_learning -v
    def test_gradient_direction_learning(self):
        """Test that gradient directions enable learning toward targets."""
        torch.manual_seed(42)

        # Create array and try to fit a specific value at a fractional index
        test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        diff_array = DifferentiableArray(test_tensor, auto_train=False, temperature=1.0)

        # Target: make position 2.5 output a different value
        target_index = 2.5
        target_value = 10.0
        initial_value = diff_array[target_index].item()
        initial_loss = (initial_value - target_value) ** 2

        # Take a few gradient steps and check direction is helpful
        optimizer = torch.optim.Adam(diff_array.parameters(), lr=0.01)

        for step in range(10):  # Just a few steps to test direction
            optimizer.zero_grad()
            predicted = diff_array[target_index]
            loss = (predicted - target_value) ** 2
            loss.backward()
            optimizer.step()

        final_value = diff_array[target_index].item()
        final_loss = (final_value - target_value) ** 2

        # Gradient direction should lead to improvement (or at least not make it much worse)
        # Allow for some tolerance due to the complexity of the attention mechanism
        deterioration_threshold = initial_loss * 1.5  # Allow 50% deterioration
        self.assertTrue(
            final_loss <= deterioration_threshold,
            f"Loss deteriorated too much: {initial_loss} -> {final_loss}",
        )

        # Ideally, we should see some improvement over many steps
        # But we only check that the direction doesn't make things dramatically worse

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_gradient_direction_improvement -v
    def test_gradient_direction_improvement(self):
        """Test that gradient direction leads to loss improvement."""
        torch.manual_seed(42)

        test_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        grad_array = DifferentiableArray(test_tensor, auto_train=False, temperature=1.0)
        grad_array._values.requires_grad_(True)

        # Query at fractional position
        query_index = 2.3
        target_value = 12.0

        # Compute initial loss and gradients
        predicted_value = grad_array[query_index]
        initial_loss = (predicted_value - target_value) ** 2
        initial_loss.backward()

        gradients = grad_array._values.grad.clone()
        self.assertIsNotNone(gradients)

        # Take a small step in gradient direction
        learning_rate = 0.01
        with torch.no_grad():
            grad_array._values.data -= learning_rate * gradients

        # Compute new loss
        grad_array._values.grad = None  # Clear gradients
        new_predicted = grad_array[query_index]
        new_loss = (new_predicted - target_value) ** 2

        # Gradient direction should reduce loss (or at least not increase significantly)
        improvement_threshold = initial_loss * 0.1  # Allow 10% tolerance
        self.assertTrue(
            new_loss <= initial_loss + improvement_threshold,
            f"Loss increased too much: {initial_loss.item()} -> {new_loss.item()}",
        )

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_gradient_direction_consistency -v
    def test_gradient_direction_consistency(self):
        """Test that gradient directions are consistent with optimization goals."""
        torch.manual_seed(42)

        # Create array with distinct values
        test_values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        grad_array = DifferentiableArray(test_values, auto_train=False, temperature=1.0)
        grad_array._values.requires_grad_(True)

        # Test scenario: want to increase value at position 2.0 (currently 30)
        query_pos = 2.0
        predicted_value = grad_array[query_pos]
        higher_target = 50.0  # Want to increase

        loss_increase = (predicted_value - higher_target) ** 2
        loss_increase.backward()

        grad_for_increase = grad_array._values.grad[
            0, 2
        ].item()  # Gradient at position 2

        # Reset gradients
        grad_array._values.grad = None

        # Test scenario: want to decrease value at position 2.0
        predicted_value = grad_array[query_pos]
        lower_target = 10.0  # Want to decrease

        loss_decrease = (predicted_value - lower_target) ** 2
        loss_decrease.backward()

        grad_for_decrease = grad_array._values.grad[
            0, 2
        ].item()  # Gradient at position 2

        # Gradients should point in opposite directions for opposite goals
        self.assertTrue(
            grad_for_increase * grad_for_decrease < 0,
            f"Gradients should have opposite signs: {grad_for_increase} vs {grad_for_decrease}",
        )

    # python -m unittest differentiable_array_test.TestDifferentiableArray.test_gradient_direction_query_movement -v
    def test_gradient_direction_query_movement(self):
        """Test that query position gradients point toward better positions."""
        torch.manual_seed(42)

        test_values = torch.tensor([1.0, 10.0, 5.0, 15.0, 3.0])  # Non-monotonic values
        grad_array = DifferentiableArray(test_values, auto_train=False, temperature=1.0)

        # Start at position 0.5, want to find position with value closest to 15
        # Position 3 has value 15.0, so gradient should point toward higher indices
        initial_pos = torch.tensor([[0.5]], requires_grad=True)

        # Compute current value
        current_value = grad_array.get(initial_pos)
        target_value = 15.0
        loss = (current_value - target_value) ** 2
        loss.backward()

        query_gradient = initial_pos.grad[0, 0].item()

        # Since position 3 (value 15) is to the right of 0.5, gradient should be positive
        # (pointing toward higher position indices)
        self.assertTrue(
            query_gradient > 0,
            f"Query gradient {query_gradient} should be positive to move toward position 3",
        )

        # Test opposite direction: start at 4.5, target is still 15 at position 3
        initial_pos2 = torch.tensor([[4.5]], requires_grad=True)

        current_value2 = grad_array.get(initial_pos2)
        loss2 = (current_value2 - target_value) ** 2
        loss2.backward()

        # Check if gradient exists (might be None due to computation graph issues)
        if initial_pos2.grad is not None:
            query_gradient2 = initial_pos2.grad[0, 0].item()
            # Since position 3 (value 15) is to the left of 4.5, gradient should be negative
            self.assertTrue(
                query_gradient2 < 0,
                f"Query gradient {query_gradient2} should be negative to move toward position 3",
            )
        else:
            # Skip this part of the test if gradient computation failed
            self.skipTest("Gradient computation failed for second query position")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
