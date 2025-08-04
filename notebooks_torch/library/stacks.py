import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .array_ops import AssignIndexVectored, SuperpositionLookup


class DifferentiableStack(nn.Module):
    """A differentiable stack data structure implemented as a PyTorch module."""

    def __init__(
        self, stack_shape: Tuple[int, ...], device: Optional[torch.device] = None
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stack_shape = stack_shape
        self.device = device

        # Initialize buffer and index as parameters
        buffer = torch.zeros(stack_shape, dtype=torch.float32, device=device)
        index = F.one_hot(torch.tensor(0, device=device), stack_shape[0]).float()

        self.register_parameter("buffer", nn.Parameter(buffer))
        self.register_parameter("index", nn.Parameter(index))

        # Operations
        self.assign_op = AssignIndexVectored()
        self.lookup_op = SuperpositionLookup()

    def push(self, element: torch.Tensor) -> None:
        """Push an element onto the stack."""
        # Update buffer at current index position
        self.buffer.data = self.assign_op(self.buffer, self.index, element)
        # Shift index pointer
        self.index.data = torch.roll(self.index, shifts=1, dims=0)

    def pop(self) -> torch.Tensor:
        """Pop an element from the stack."""
        # Shift index pointer back
        self.index.data = torch.roll(self.index, shifts=-1, dims=0)
        # Get element at current position
        element = self.lookup_op(self.buffer, self.index)
        return element

    def peek(self) -> torch.Tensor:
        """Peek at the top element without removing it."""
        # Get index of top element
        peek_index = torch.roll(self.index, shifts=-1, dims=0)
        element = self.lookup_op(self.buffer, peek_index)
        return element

    def size(self) -> torch.Tensor:
        """Get the current size of the stack (differentiable)."""
        # Sum of shifted index gives current stack size
        return torch.sum(torch.roll(self.index, shifts=-1, dims=0))

    def is_empty(self) -> torch.Tensor:
        """Check if stack is empty (differentiable)."""
        return self.size() == 0

    def reset(self) -> None:
        """Reset the stack to empty state."""
        self.buffer.data.zero_()
        self.index.data.zero_()
        self.index.data[0] = 1.0


class StackFactory(nn.Module):
    """Factory for creating different types of stacks."""

    def __init__(self):
        super().__init__()

    def create_stack(
        self, stack_shape: Tuple[int, ...], device: Optional[torch.device] = None
    ) -> DifferentiableStack:
        """Create a new differentiable stack."""
        return DifferentiableStack(stack_shape, device)

    def create_from_buffer(self, buffer: torch.Tensor) -> DifferentiableStack:
        """Create a stack from an existing buffer."""
        stack = DifferentiableStack(buffer.shape, buffer.device)
        stack.buffer.data = buffer.clone()
        return stack


class MultiStack(nn.Module):
    """Multiple parallel stacks for batch processing."""

    def __init__(
        self,
        num_stacks: int,
        stack_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_stacks = num_stacks
        self.stack_shape = stack_shape
        self.device = device

        # Create multiple stacks
        self.stacks = nn.ModuleList(
            [DifferentiableStack(stack_shape, device) for _ in range(num_stacks)]
        )

    def push_batch(self, elements: torch.Tensor) -> None:
        """Push elements to multiple stacks in parallel."""
        assert elements.shape[0] == self.num_stacks
        for i, element in enumerate(elements):
            self.stacks[i].push(element)

    def pop_batch(self) -> torch.Tensor:
        """Pop elements from multiple stacks in parallel."""
        elements = []
        for stack in self.stacks:
            elements.append(stack.pop())
        return torch.stack(elements)

    def peek_batch(self) -> torch.Tensor:
        """Peek at elements from multiple stacks in parallel."""
        elements = []
        for stack in self.stacks:
            elements.append(stack.peek())
        return torch.stack(elements)


class StackAttention(nn.Module):
    """Attention mechanism over stack contents."""

    def __init__(self, stack_shape: Tuple[int, ...], attention_dim: int):
        super().__init__()
        self.stack_shape = stack_shape
        self.attention_dim = attention_dim

        # Attention weights computation
        self.attention_net = nn.Sequential(
            nn.Linear(stack_shape[-1], attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, stack: DifferentiableStack, query: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention-weighted sum over stack contents."""
        # Get all stack contents
        stack_contents = stack.buffer  # Shape: (stack_size, element_dim)

        if query is not None:
            # Use query for attention (could extend this)
            # For now, just use stack contents
            pass

        # Compute attention weights
        attention_scores = self.attention_net(stack_contents)  # Shape: (stack_size, 1)
        attention_weights = F.softmax(
            attention_scores.squeeze(-1), dim=0
        )  # Shape: (stack_size,)

        # Apply attention to get weighted sum
        attended_content = torch.sum(
            attention_weights.unsqueeze(-1) * stack_contents, dim=0
        )

        return attended_content


# Backward compatibility functions
def new_stack(
    stack_shape: Tuple[int, ...],
    is_learnable: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a new stack (backward compatibility)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = torch.zeros(stack_shape, dtype=torch.float32, device=device)
    index = F.one_hot(torch.tensor(0, device=device), stack_shape[0]).float()

    if is_learnable:
        buffer = nn.Parameter(buffer)
        index = nn.Parameter(index)

    return buffer, index


def new_stack_from_buffer(
    buffer: torch.Tensor, is_learnable: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a stack from existing buffer (backward compatibility)."""
    device = buffer.device
    stack_shape = buffer.shape
    index = F.one_hot(torch.tensor(0, device=device), stack_shape[0]).float()

    if is_learnable:
        buffer = nn.Parameter(buffer)
        index = nn.Parameter(index)

    return buffer, index


def stack_push(
    state: Tuple[torch.Tensor, torch.Tensor], element: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Push operation (backward compatibility)."""
    buffer, index = state
    assign_op = AssignIndexVectored()
    buffer = assign_op(buffer, index, element)
    index = torch.roll(index, shifts=1, dims=0)
    return buffer, index


def stack_pop(
    state: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Pop operation (backward compatibility)."""
    buffer, index = state
    index = torch.roll(index, shifts=-1, dims=0)
    lookup_op = SuperpositionLookup()
    element = lookup_op(buffer, index)
    return (buffer, index), element


def stack_peek(stack: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Peek operation (backward compatibility)."""
    buffer, index = stack
    index = torch.roll(index, shifts=-1, dims=0)
    lookup_op = SuperpositionLookup()
    element = lookup_op(buffer, index)
    return element
