import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AssignIndex(nn.Module):
    """Assigns a value to a specific index in a tensor using differentiable masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self, arr: torch.Tensor, index: int, element: torch.Tensor
    ) -> torch.Tensor:
        device = arr.device
        arr_shape = arr.shape

        pos_mask = torch.eye(arr_shape[0], device=device)[index]
        neg_mask = 1 - pos_mask

        # Handle different tensor dimensions properly
        if len(arr_shape) > 1:
            pos_mask = pos_mask.view(-1, *([1] * (len(arr_shape) - 1)))
            neg_mask = neg_mask.view(-1, *([1] * (len(arr_shape) - 1)))

            # Expand element to match the inner dimensions of arr
            if element.dim() == 0:
                element = element.expand(arr_shape[1:])
            tiled_element = element.unsqueeze(0).expand(arr_shape)
        else:
            # For 1D arrays, element should be scalar
            tiled_element = element.expand(arr_shape)

        result = arr * neg_mask + tiled_element * pos_mask
        return result


class AssignIndexVectored(nn.Module):
    """Assigns a value using a vectorized index (one-hot or soft assignment)."""

    def __init__(self):
        super().__init__()

    def forward(
        self, arr: torch.Tensor, index: torch.Tensor, element: torch.Tensor
    ) -> torch.Tensor:
        device = arr.device
        arr_shape = arr.shape

        pos_mask = index
        neg_mask = 1 - pos_mask

        # Handle different tensor dimensions properly
        if len(arr_shape) > 1:
            pos_mask = pos_mask.view(-1, *([1] * (len(arr_shape) - 1)))
            neg_mask = neg_mask.view(-1, *([1] * (len(arr_shape) - 1)))

            # Expand element to match the inner dimensions of arr
            if element.dim() == 0:
                element = element.expand(arr_shape[1:])
            tiled_element = element.unsqueeze(0).expand(arr_shape)
        else:
            # For 1D arrays, element should be scalar
            tiled_element = element.expand(arr_shape)

        result = arr * neg_mask + tiled_element * pos_mask
        return result


class NaiveLookup(nn.Module):
    """Simple index-based lookup with rounding."""

    def __init__(self):
        super().__init__()

    def forward(self, arr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index = torch.round(index).long()
        return arr[index]


class LinearLookup(nn.Module):
    """Linear interpolation between adjacent array elements."""

    def __init__(self):
        super().__init__()

    def forward(self, arr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        t1 = torch.floor(index)
        t2 = torch.ceil(index)

        # Handle division by zero case
        denominator = t2 - t1
        t = torch.where(
            denominator != 0, (index - t1) / denominator, torch.zeros_like(index)
        )

        i1 = t1.long()
        i2 = t2.long()

        # Linear interpolation
        result = t * arr[i1] + (1 - t) * arr[i2]
        return result


class SuperpositionLookup(nn.Module):
    """Soft lookup using weighted combination of array elements."""

    def __init__(self):
        super().__init__()

    def forward(self, arr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if arr.dim() == 1:
            arr = arr.unsqueeze(-1)
        indices = indices.unsqueeze(-1)
        result = arr * indices
        return torch.sum(result, dim=0)


class AsymmetricalVectoredLookup(torch.autograd.Function):
    """Custom autograd function for asymmetrical vectored lookup."""

    @staticmethod
    def forward(ctx, v, k):
        # Pick the value at the most likely index, non-differentiably
        b_idx = torch.argmax(k, dim=-1)
        idx_len = b_idx.shape[0]
        a_idx = torch.arange(idx_len, dtype=torch.long, device=v.device)
        forward_result = v[a_idx, b_idx]

        ctx.save_for_backward(v, k, forward_result)
        return forward_result

    @staticmethod
    def backward(ctx, upstream_grads):
        v, k, forward_result = ctx.saved_tensors
        k_shape = k.shape

        # Estimate the target scalar which we want to look up
        target = forward_result - upstream_grads
        target = target.unsqueeze(-1)

        # Find the index of element in the array which is closest to target
        diff_vector = torch.square(v - target)
        d_idx = torch.argmin(diff_vector, dim=-1)

        # Create a vector which is 1 everywhere except the idx
        # of the target, where it is -1
        ones = torch.ones(k_shape, device=k.device)
        eyes = F.one_hot(d_idx, k_shape[-1]).float()
        k_grad = -(2 * eyes - ones)

        # d/dv (v . k) = k
        v_grad = k

        upstream_grads = upstream_grads.unsqueeze(-1)
        return upstream_grads * v_grad, torch.abs(upstream_grads) * k_grad


class AsymmetricalLookup(nn.Module):
    """Asymmetrical vectored lookup module."""

    def __init__(self):
        super().__init__()

    def forward(self, v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return AsymmetricalVectoredLookup.apply(v, k)


class Bandwidthify(nn.Module):
    """Converts a continuous index to a probability distribution over discrete indices."""

    def __init__(self, bandwidth: int):
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        device = index.device
        t1 = torch.floor(index)
        t2 = torch.ceil(index)

        # Handle division by zero case
        denominator = t2 - t1
        t = torch.where(
            denominator != 0, (index - t1) / denominator, torch.zeros_like(index)
        )

        i1 = t1.long()
        i2 = t2.long()

        # Prevent array out of bounds
        i1 = torch.clamp(i1, 0, self.bandwidth - 1)
        i2 = torch.clamp(i2, 0, self.bandwidth - 1)
        t = torch.clamp(t, 0, 1)

        # Linear interpolation
        eye = torch.eye(self.bandwidth, device=device)
        result = t * eye[i1] + (1 - t) * eye[i2]
        return result


class BulkBandwidthify(nn.Module):
    """Applies bandwidthify to multiple indices."""

    def __init__(self, bandwidth: int):
        super().__init__()
        self.bandwidth = bandwidth
        self.bandwidthify = Bandwidthify(bandwidth)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        device = indices.device
        num_indices = indices.shape[0]

        result = torch.zeros(
            (num_indices, self.bandwidth), dtype=torch.float32, device=device
        )

        for i, index in enumerate(indices):
            b_index = self.bandwidthify(index)
            result[i] += b_index

        return result


class SuperpositionArrayLookup(nn.Module):
    """Combines bandwidthify with superposition lookup for continuous indexing."""

    def __init__(self):
        super().__init__()
        self.superposition_lookup = SuperpositionLookup()

    def forward(self, arr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        bandwidth = arr.shape[0]
        bandwidthify = Bandwidthify(bandwidth)
        vectored_index = bandwidthify(index)
        result = self.superposition_lookup(arr, vectored_index)
        return result


class ResidualLookup(nn.Module):
    """Lookup with residual computation."""

    def __init__(self):
        super().__init__()

    def forward(
        self, arr: torch.Tensor, index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        i = torch.round(index)
        residue = index - i
        i = i.long()

        result = arr[i]
        return result, residue


class ShapeMatcher(nn.Module):
    """Handles tensor shape matching for broadcasting."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use PyTorch's built-in broadcasting rules
        try:
            # Just expand both to a common shape using broadcasting
            result_shape = torch.broadcast_shapes(x.shape, y.shape)
            x_expanded = x.expand(result_shape)
            y_expanded = y.expand(result_shape)
            return x_expanded, y_expanded
        except RuntimeError:
            # If broadcasting fails, try manual alignment as in original TF code
            if x.dim() > y.dim():
                high, low = x, y
            else:
                high, low = y, x

            l_rank, l_shape = low.dim(), low.shape
            h_rank, h_shape = high.dim(), high.shape

            # Try to match from the right (trailing dimensions)
            if l_rank > 0 and h_rank >= l_rank:
                common_shape = h_shape[-l_rank:]
                if list(common_shape) == list(l_shape):
                    # Reshape low tensor to match high tensor's dimensions
                    padding = [1] * (h_rank - l_rank)
                    new_shape = padding + list(l_shape)
                    low = low.reshape(new_shape)
                    return (high, low) if x.dim() > y.dim() else (low, high)

            raise AssertionError("No common shape to broadcast")


class BroadcastMultiply(nn.Module):
    """Broadcasting multiplication operation."""

    def __init__(self):
        super().__init__()
        self.shape_matcher = ShapeMatcher()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.shape_matcher(x, y)
        return x * y


class TensorLookup2D(nn.Module):
    """2D tensor lookup using outer product masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self, arr: torch.Tensor, x_index: torch.Tensor, y_index: torch.Tensor
    ) -> torch.Tensor:
        # Calculate outer product
        mask = torch.outer(x_index, y_index)

        # Expand mask to match arr dimensions
        if len(arr.shape) > 2:
            # Add dimensions for the extra dimensions in arr
            for _ in range(len(arr.shape) - 2):
                mask = mask.unsqueeze(-1)
            mask = mask.expand(arr.shape)

        # Element-wise multiply
        masked_arr = mask * arr

        # Reduce max to extract the cell
        element = torch.max(masked_arr, dim=0)[0]
        element = torch.max(element, dim=0)[0]
        return element


class TensorWrite2D(nn.Module):
    """2D tensor write operation using outer product masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        arr: torch.Tensor,
        element: torch.Tensor,
        x_index: torch.Tensor,
        y_index: torch.Tensor,
    ) -> torch.Tensor:
        device = arr.device
        arr_shape = arr.shape
        mask = torch.outer(x_index, y_index)

        # Expand mask to match arr dimensions
        if len(arr_shape) > 2:
            # Add dimensions for the extra dimensions in arr
            for _ in range(len(arr_shape) - 2):
                mask = mask.unsqueeze(-1)
            mask = mask.expand(arr_shape)

        # Expand element to match arr dimensions
        if element.dim() == 1 and len(arr_shape) == 3:
            element = element.reshape([1, 1, -1])
            element = element.expand([arr_shape[0], arr_shape[1], element.shape[-1]])
        elif element.dim() == 0:
            element = element.expand(arr_shape)

        result = (1.0 - mask) * arr + mask * element
        return result


# Create default instances for backward compatibility
assign_index = AssignIndex()
assign_index_vectored = AssignIndexVectored()
naive_lookup = NaiveLookup()
linear_lookup = LinearLookup()
superposition_lookup_vectored = SuperpositionLookup()
asymmetrical_vectored_lookup = AsymmetricalLookup()
residual_lookup = ResidualLookup()
broadcast_multiply = BroadcastMultiply()
tensor_lookup_2d = TensorLookup2D()
tensor_write_2d = TensorWrite2D()


def superposition_lookup(arr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Functional interface for superposition lookup."""
    module = SuperpositionArrayLookup()
    return module(arr, index)


def bandwidthify(index: torch.Tensor, bandwidth: int) -> torch.Tensor:
    """Functional interface for bandwidthify."""
    module = Bandwidthify(bandwidth)
    return module(index)


def bulk_bandwidthify(indices: torch.Tensor, bandwidth: int) -> torch.Tensor:
    """Functional interface for bulk bandwidthify."""
    module = BulkBandwidthify(bandwidth)
    return module(indices)


def match_shapes(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional interface for shape matching."""
    module = ShapeMatcher()
    return module(x, y)
