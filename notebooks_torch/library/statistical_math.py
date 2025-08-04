import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProbabilityDistribution(nn.Module):
    """Converts vectors to probability distributions."""

    def __init__(self, normalization: str = "l2", epsilon: float = 1e-9):
        super().__init__()
        self.normalization = normalization
        self.epsilon = epsilon

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.normalization == "l2":
            # L2 normalization with small epsilon for stability
            v2 = torch.sqrt(torch.square(v) + self.epsilon)
            m = torch.sum(v2, dim=-1, keepdim=True)
            n = torch.where(m != 0, v2 / m, torch.zeros_like(v2))
            return n
        elif self.normalization == "softmax":
            return F.softmax(v, dim=-1)
        elif self.normalization == "l1":
            abs_v = torch.abs(v) + self.epsilon
            return abs_v / torch.sum(abs_v, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")


class CrossEntropy(nn.Module):
    """Cross-entropy computation with customizable base."""

    def __init__(
        self, base: float = 2.0, epsilon: float = 1e-9, reduction: str = "mean"
    ):
        super().__init__()
        self.base = base
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_base = torch.log(torch.tensor(self.base))
        entropy = -2 * torch.mean(y * torch.log(x + self.epsilon), dim=-1) / log_base

        if self.reduction == "mean":
            return torch.mean(entropy)
        elif self.reduction == "sum":
            return torch.sum(entropy)
        elif self.reduction == "none":
            return entropy
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class Entropy(nn.Module):
    """Entropy computation (self cross-entropy)."""

    def __init__(
        self, base: float = 2.0, epsilon: float = 1e-9, reduction: str = "mean"
    ):
        super().__init__()
        self.cross_entropy = CrossEntropy(base, epsilon, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(x, x)


class KLDivergence(nn.Module):
    """Kullback-Leibler divergence."""

    def __init__(self, epsilon: float = 1e-9, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # KL(P||Q) = sum(p * log(p/q))
        kl = p * torch.log((p + self.epsilon) / (q + self.epsilon))
        kl = torch.sum(kl, dim=-1)

        if self.reduction == "mean":
            return torch.mean(kl)
        elif self.reduction == "sum":
            return torch.sum(kl)
        elif self.reduction == "none":
            return kl
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class JSdivergence(nn.Module):
    """Jensen-Shannon divergence."""

    def __init__(self, epsilon: float = 1e-9, reduction: str = "mean"):
        super().__init__()
        self.kl_div = KLDivergence(epsilon, "none")
        self.reduction = reduction

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        js = 0.5 * self.kl_div(p, m) + 0.5 * self.kl_div(q, m)

        if self.reduction == "mean":
            return torch.mean(js)
        elif self.reduction == "sum":
            return torch.sum(js)
        elif self.reduction == "none":
            return js
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MutualInformation(nn.Module):
    """Mutual information estimation."""

    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Simple mutual information estimation
        # This is a basic implementation - more sophisticated methods exist

        # Compute marginal distributions
        px = torch.mean(x, dim=0)
        py = torch.mean(y, dim=0)

        # Compute joint distribution (assuming independence as baseline)
        pxy = torch.outer(px, py)

        # Estimate actual joint from samples
        batch_size = x.shape[0]
        actual_joint = torch.zeros_like(pxy)
        for i in range(batch_size):
            actual_joint += torch.outer(x[i], y[i])
        actual_joint /= batch_size

        # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
        mi = actual_joint * torch.log(
            (actual_joint + self.epsilon) / (pxy + self.epsilon)
        )
        return torch.sum(mi)


class DistributionMatcher(nn.Module):
    """Matches one distribution to another using various metrics."""

    def __init__(self, metric: str = "kl", reduction: str = "mean"):
        super().__init__()
        self.metric = metric

        if metric == "kl":
            self.distance = KLDivergence(reduction=reduction)
        elif metric == "js":
            self.distance = JSdivergence(reduction=reduction)
        elif metric == "mse":
            self.distance = nn.MSELoss(reduction=reduction)
        elif metric == "l1":
            self.distance = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.metric in ["kl", "js"]:
            return self.distance(predicted, target)
        else:
            return self.distance(predicted, target)


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability distributions."""

    def __init__(self, initial_temperature: float = 1.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer("temperature", torch.tensor(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for differentiable discrete sampling."""

    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)


# Backward compatibility functions
def to_prob_dist_all(v: torch.Tensor) -> torch.Tensor:
    """Functional interface for probability distribution conversion."""
    module = ProbabilityDistribution()
    return module(v)


def cross_entropy(
    x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-9
) -> torch.Tensor:
    """Functional interface for cross-entropy."""
    module = CrossEntropy(epsilon=epsilon, reduction="none")
    return module(x, y)


def entropy(x: torch.Tensor) -> torch.Tensor:
    """Functional interface for entropy."""
    module = Entropy(reduction="none")
    return module(x)
