import torch
import torch.nn.functional as F

from .array_ops import assign_index_vectored, superposition_lookup_vectored


def new_stack(stack_shape, is_learnable=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = torch.zeros(stack_shape, dtype=torch.float32, device=device)
    index = F.one_hot(torch.tensor(0, device=device), stack_shape[0]).float()

    if is_learnable:
        buffer = torch.nn.Parameter(buffer)
        index = torch.nn.Parameter(index)

    stack = (buffer, index)
    return stack


def new_stack_from_buffer(buffer, is_learnable=False):
    device = buffer.device
    stack_shape = buffer.shape
    index = F.one_hot(torch.tensor(0, device=device), stack_shape[0]).float()

    if is_learnable:
        buffer = torch.nn.Parameter(buffer)
        index = torch.nn.Parameter(index)

    stack = (buffer, index)
    return stack


def stack_push(state, element):
    buffer, index = state
    buffer = assign_index_vectored(buffer, index, element)
    index = torch.roll(index, shifts=1, dims=0)
    state = (buffer, index)
    return state


def stack_pop(state):
    buffer, index = state
    index = torch.roll(index, shifts=-1, dims=0)
    element = superposition_lookup_vectored(buffer, index)
    state = (buffer, index)
    return state, element


def stack_peek(stack):
    buffer, index = stack
    index = torch.roll(index, shifts=-1, dims=0)
    element = superposition_lookup_vectored(buffer, index)
    return element
