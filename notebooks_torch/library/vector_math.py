import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CircularShift(nn.Module):
    """Circular shift operation using permutation matrices."""

    def __init__(self, shift: int = -1):
        super().__init__()
        self.shift = shift

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        device = vec.device
        P = torch.eye(vec.shape[-1], device=device)
        P = torch.roll(P, shifts=self.shift, dims=0)

        # Handle batch dimensions
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
            result = vec @ P
            return result.squeeze(0)
        else:
            return vec @ P


class DotProduct(nn.Module):
    """Dot product operation with optional normalization."""

    def __init__(self, normalize: bool = False, dim: int = -1):
        super().__init__()
        self.normalize = normalize
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.normalize(x, p=2, dim=self.dim)
            y = F.normalize(y, p=2, dim=self.dim)

        result = torch.multiply(x, y)
        return torch.sum(result, dim=self.dim)


class CosineSimilarity(nn.Module):
    """Cosine similarity between vectors."""

    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x, y, dim=self.dim, eps=self.eps)


class VectorNorm(nn.Module):
    """Vector norm computation with various p-norms."""

    def __init__(self, p: float = 2.0, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)


class Normalize(nn.Module):
    """Vector normalization."""

    def __init__(self, p: float = 2.0, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


class OuterProduct(nn.Module):
    """Outer product between vectors."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.outer(x, y)


class VectorProjection(nn.Module):
    """Project vector a onto vector b."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # proj_b(a) = (a · b / |b|²) * b
        dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
        b_norm_sq = torch.sum(b * b, dim=-1, keepdim=True) + self.eps
        return (dot_ab / b_norm_sq) * b


class VectorReflection(nn.Module):
    """Reflect vector a across the hyperplane orthogonal to n."""

    def __init__(self):
        super().__init__()
        self.projection = VectorProjection()

    def forward(self, a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # reflection = a - 2 * proj_n(a)
        proj = self.projection(a, n)
        return a - 2 * proj


class Rotation2D(nn.Module):
    """2D rotation matrix."""

    def __init__(self, angle: Optional[float] = None, learnable: bool = False):
        super().__init__()

        if angle is not None:
            if learnable:
                self.angle = nn.Parameter(torch.tensor(angle))
            else:
                self.register_buffer("angle", torch.tensor(angle))
        else:
            self.angle = nn.Parameter(torch.zeros(1)) if learnable else torch.zeros(1)

    def forward(
        self, x: torch.Tensor, angle: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if angle is None:
            angle = self.angle

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Rotation matrix [[cos, -sin], [sin, cos]]
        rotation_matrix = torch.stack(
            [torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])]
        )

        # Apply rotation
        if x.dim() == 1:
            return rotation_matrix @ x
        else:
            return torch.matmul(x, rotation_matrix.T)


class GramSchmidt(nn.Module):
    """Gram-Schmidt orthogonalization process."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors: tensor of shape (..., n_vectors, vector_dim)
        Returns:
            orthogonal vectors of same shape
        """
        *batch_dims, n_vectors, vector_dim = vectors.shape

        # Flatten batch dimensions for processing
        vectors = vectors.view(-1, n_vectors, vector_dim)
        batch_size = vectors.shape[0]

        orthogonal = torch.zeros_like(vectors)

        for b in range(batch_size):
            for i in range(n_vectors):
                # Start with the original vector
                orthogonal[b, i] = vectors[b, i].clone()

                # Subtract projections onto previous orthogonal vectors
                for j in range(i):
                    prev_vec = orthogonal[b, j]
                    # Projection of vectors[b, i] onto prev_vec
                    dot_product = torch.sum(vectors[b, i] * prev_vec)
                    norm_sq = torch.sum(prev_vec * prev_vec) + self.eps
                    projection = (dot_product / norm_sq) * prev_vec
                    orthogonal[b, i] -= projection

                # Normalize (optional, makes it orthonormal)
                norm = torch.norm(orthogonal[b, i]) + self.eps
                orthogonal[b, i] /= norm

        # Restore original shape
        return orthogonal.view(*batch_dims, n_vectors, vector_dim)


class AttentionWeights(nn.Module):
    """Compute attention weights between query and key vectors."""

    def __init__(self, temperature: float = 1.0, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # Compute attention scores
        if self.normalize:
            query = F.normalize(query, p=2, dim=-1)
            keys = F.normalize(keys, p=2, dim=-1)

        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)

        # Apply temperature scaling and softmax
        attention_weights = F.softmax(scores / self.temperature, dim=-1)

        return attention_weights


# Backward compatibility functions
def shift_left_one_hot(vec: torch.Tensor, shift: int = -1) -> torch.Tensor:
    """Functional interface for circular shift."""
    module = CircularShift(shift)
    result = module(vec)
    if vec.dim() == 1:
        return result.unsqueeze(0)  # Match original behavior
    return result


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Functional interface for dot product."""
    module = DotProduct()
    return module(x, y)
