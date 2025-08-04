import torch


def shift_left_one_hot(vec, shift=-1):
    device = vec.device
    P = torch.eye(vec.shape[0], device=device)
    P = torch.roll(P, shifts=shift, dims=0)

    vec = vec.unsqueeze(0)

    return vec @ P


def dot(x, y):
    r = torch.multiply(x, y)
    return torch.sum(r, dim=-1)
