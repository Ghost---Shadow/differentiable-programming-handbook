import torch


def to_prob_dist_all(v):
    v2 = torch.sqrt(torch.square(v) + 1e-9)
    m = torch.sum(v2, dim=-1, keepdim=True)
    n = torch.where(m != 0, v2 / m, torch.zeros_like(v2))
    return n


def cross_entropy(x, y, epsilon=1e-9):
    return (
        -2
        * torch.mean(y * torch.log(x + epsilon), dim=-1)
        / torch.log(torch.tensor(2.0))
    )


def entropy(x):
    return cross_entropy(x, x)
