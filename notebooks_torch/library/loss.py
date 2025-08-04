import torch
import torch.nn.functional as F


def bistable_loss(x):
    a = x**2
    b = (x - 1) ** 2

    return a * b


def permute_matrix_loss(P, cycle_length=1, cycle_weight=0):
    device = P.device
    loss = torch.tensor(0.0, device=device)

    P_square = torch.square(P)
    axis_1_sum = torch.sum(P_square, dim=1)
    axis_0_sum = torch.sum(P_square, dim=0)

    # Penalize axes not adding up to one
    loss += F.mse_loss(axis_1_sum, torch.ones_like(axis_1_sum)) * axis_1_sum.numel() / 2
    loss += F.mse_loss(axis_0_sum, torch.ones_like(axis_0_sum)) * axis_0_sum.numel() / 2

    # Penalize numbers outside 0 or 1
    loss += torch.sum(bistable_loss(P))

    # Cycle loss
    Q = P
    for _ in range(cycle_length - 1):
        Q = P @ Q
    cycle_loss = F.mse_loss(Q, torch.eye(Q.shape[0], device=device)) * Q.numel() / 2
    loss += cycle_loss * cycle_weight

    return loss
