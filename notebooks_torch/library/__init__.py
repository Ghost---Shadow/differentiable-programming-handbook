"""
PyTorch Differentiable Programming Library

A collection of differentiable operations, loss functions, and data structures
implemented as PyTorch modules for enhanced composability and torch-like behavior.
"""

# Array Operations
from .array_ops import (
    # Core modules
    AssignIndex,
    AssignIndexVectored,
    NaiveLookup,
    LinearLookup,
    SuperpositionLookup,
    AsymmetricalLookup,
    Bandwidthify,
    BulkBandwidthify,
    SuperpositionArrayLookup,
    ResidualLookup,
    ShapeMatcher,
    BroadcastMultiply,
    TensorLookup2D,
    TensorWrite2D,
    # Functional interfaces (backward compatibility)
    assign_index,
    assign_index_vectored,
    naive_lookup,
    linear_lookup,
    superposition_lookup_vectored,
    asymmetrical_vectored_lookup,
    residual_lookup,
    broadcast_multiply,
    tensor_lookup_2d,
    tensor_write_2d,
    superposition_lookup,
    bandwidthify,
    bulk_bandwidthify,
    match_shapes,
)

# Loss Functions
from .loss import (
    # Loss modules
    BistableLoss,
    PermuteMatrixLoss,
    DoublyStochasticLoss,
    OrthogonalLoss,
    SparsityLoss,
    GroupSparsityLoss,
    # Functional interfaces (backward compatibility)
    bistable_loss,
    permute_matrix_loss,
)

# Stack Data Structures
from .stacks import (
    # Stack modules
    DifferentiableStack,
    StackFactory,
    MultiStack,
    StackAttention,
    # Functional interfaces (backward compatibility)
    new_stack,
    new_stack_from_buffer,
    stack_push,
    stack_pop,
    stack_peek,
)

# Statistical Math Operations
from .statistical_math import (
    # Statistical modules
    ProbabilityDistribution,
    CrossEntropy,
    Entropy,
    KLDivergence,
    JSdivergence,
    MutualInformation,
    DistributionMatcher,
    TemperatureScaling,
    GumbelSoftmax,
    # Functional interfaces (backward compatibility)
    to_prob_dist_all,
    cross_entropy,
    entropy,
)

# Vector Math Operations
from .vector_math import (
    # Vector modules
    CircularShift,
    DotProduct,
    CosineSimilarity,
    VectorNorm,
    Normalize,
    OuterProduct,
    VectorProjection,
    VectorReflection,
    Rotation2D,
    GramSchmidt,
    AttentionWeights,
    # Functional interfaces (backward compatibility)
    shift_left_one_hot,
    dot,
)

__version__ = "1.0.0"

__all__ = [
    # Array Operations
    "AssignIndex",
    "AssignIndexVectored",
    "NaiveLookup",
    "LinearLookup",
    "SuperpositionLookup",
    "AsymmetricalLookup",
    "Bandwidthify",
    "BulkBandwidthify",
    "SuperpositionArrayLookup",
    "ResidualLookup",
    "ShapeMatcher",
    "BroadcastMultiply",
    "TensorLookup2D",
    "TensorWrite2D",
    "assign_index",
    "assign_index_vectored",
    "naive_lookup",
    "linear_lookup",
    "superposition_lookup_vectored",
    "asymmetrical_vectored_lookup",
    "residual_lookup",
    "broadcast_multiply",
    "tensor_lookup_2d",
    "tensor_write_2d",
    "superposition_lookup",
    "bandwidthify",
    "bulk_bandwidthify",
    "match_shapes",
    # Loss Functions
    "BistableLoss",
    "PermuteMatrixLoss",
    "DoublyStochasticLoss",
    "OrthogonalLoss",
    "SparsityLoss",
    "GroupSparsityLoss",
    "bistable_loss",
    "permute_matrix_loss",
    # Stack Data Structures
    "DifferentiableStack",
    "StackFactory",
    "MultiStack",
    "StackAttention",
    "new_stack",
    "new_stack_from_buffer",
    "stack_push",
    "stack_pop",
    "stack_peek",
    # Statistical Math
    "ProbabilityDistribution",
    "CrossEntropy",
    "Entropy",
    "KLDivergence",
    "JSdivergence",
    "MutualInformation",
    "DistributionMatcher",
    "TemperatureScaling",
    "GumbelSoftmax",
    "to_prob_dist_all",
    "cross_entropy",
    "entropy",
    # Vector Math
    "CircularShift",
    "DotProduct",
    "CosineSimilarity",
    "VectorNorm",
    "Normalize",
    "OuterProduct",
    "VectorProjection",
    "VectorReflection",
    "Rotation2D",
    "GramSchmidt",
    "AttentionWeights",
    "shift_left_one_hot",
    "dot",
]
