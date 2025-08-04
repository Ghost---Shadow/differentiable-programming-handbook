import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BistableLoss(nn.Module):
    """Loss function that encourages values to be either 0 or 1 (bistable)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x**2
        b = (x - 1) ** 2
        loss = a * b

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class PermuteMatrixLoss(nn.Module):
    """Loss function for permutation matrices with optional cycle constraints."""

    def __init__(
        self, cycle_length: int = 1, cycle_weight: float = 0.0, reduction: str = "mean"
    ):
        super().__init__()
        self.cycle_length = cycle_length
        self.cycle_weight = cycle_weight
        self.reduction = reduction
        self.bistable_loss = BistableLoss(reduction="sum")

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        device = P.device
        loss = torch.tensor(0.0, device=device)

        P_square = torch.square(P)
        axis_1_sum = torch.sum(P_square, dim=1)
        axis_0_sum = torch.sum(P_square, dim=0)

        # Penalize axes not adding up to one
        loss += F.mse_loss(axis_1_sum, torch.ones_like(axis_1_sum), reduction="sum") / 2
        loss += F.mse_loss(axis_0_sum, torch.ones_like(axis_0_sum), reduction="sum") / 2

        # Penalize numbers outside 0 or 1
        loss += self.bistable_loss(P)

        # Cycle loss
        if self.cycle_weight > 0 and self.cycle_length > 1:
            Q = P
            for _ in range(self.cycle_length - 1):
                Q = P @ Q
            cycle_loss = (
                F.mse_loss(Q, torch.eye(Q.shape[0], device=device), reduction="sum") / 2
            )
            loss += cycle_loss * self.cycle_weight

        if self.reduction == "mean":
            return loss / P.numel()
        elif self.reduction == "sum":
            return loss
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class DoublyStochasticLoss(nn.Module):
    """Loss function for doubly stochastic matrices (row and column sums = 1)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        device = P.device
        loss = torch.tensor(0.0, device=device)

        # Row sums should be 1
        row_sums = torch.sum(P, dim=1)
        loss += F.mse_loss(row_sums, torch.ones_like(row_sums), reduction="sum")

        # Column sums should be 1
        col_sums = torch.sum(P, dim=0)
        loss += F.mse_loss(col_sums, torch.ones_like(col_sums), reduction="sum")

        if self.reduction == "mean":
            return loss / (2 * P.shape[0])
        elif self.reduction == "sum":
            return loss
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class OrthogonalLoss(nn.Module):
    """Loss function to encourage orthogonal matrices (P @ P.T = I)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        device = P.device
        identity = torch.eye(P.shape[0], device=device)

        # P @ P.T should equal identity
        orthogonal_product = P @ P.T
        loss = F.mse_loss(orthogonal_product, identity, reduction=self.reduction)

        return loss


class SparsityLoss(nn.Module):
    """L1 sparsity loss to encourage sparse representations."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return torch.mean(torch.abs(x))
        elif self.reduction == "sum":
            return torch.sum(torch.abs(x))
        elif self.reduction == "none":
            return torch.abs(x)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class GroupSparsityLoss(nn.Module):
    """Group sparsity loss using L2,1 norm."""

    def __init__(self, group_dim: int = -1, reduction: str = "mean"):
        super().__init__()
        self.group_dim = group_dim
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm within groups, then L1 norm across groups
        group_norms = torch.norm(x, p=2, dim=self.group_dim)

        if self.reduction == "mean":
            return torch.mean(group_norms)
        elif self.reduction == "sum":
            return torch.sum(group_norms)
        elif self.reduction == "none":
            return group_norms
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


# Create default instances for backward compatibility
def bistable_loss(x: torch.Tensor) -> torch.Tensor:
    """Functional interface for bistable loss."""
    module = BistableLoss(reduction="none")
    return module(x)


def permute_matrix_loss(
    P: torch.Tensor, cycle_length: int = 1, cycle_weight: float = 0.0
) -> torch.Tensor:
    """Functional interface for permutation matrix loss."""
    module = PermuteMatrixLoss(cycle_length, cycle_weight, reduction="sum")
    return module(P)
