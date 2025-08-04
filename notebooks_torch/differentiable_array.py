import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiableArray(nn.Module):
    """
    A differentiable array module that supports soft indexing with meaningful gradients.

    Usage:
        # Wrap a tensor
        diff_array = DifferentiableArray(my_tensor)

        # Get values (differentiable indexing)
        value = diff_array.get(2.5)  # fractional indexing
        values = diff_array.get([1.0, 3.7, 5.2])  # batch indexing

        # Set values (differentiable assignment)
        diff_array.set(2.5, new_value)
        diff_array.set([1.0, 3.7], [val1, val2])

    The lookup is differentiable w.r.t.:
    1. The lookup key (gradients point toward closest matching positions)
    2. The array values (standard backprop through attention weights)
    3. The position embeddings (learned representations)
    """

    def __init__(
        self,
        values=None,
        array_size=None,
        embedding_dim=64,
        temperature=1.0,
        position_init="uniform",
        auto_train=True,
        learning_rate=0.001,
        training_steps=5,
        loss_weights=None,
    ):
        super().__init__()

        # Handle tensor wrapping or explicit array_size
        if values is not None:
            if values.dim() == 1:
                values = values.unsqueeze(0)  # Add batch dimension
            self.array_size = values.shape[-1]
            self._values = nn.Parameter(values.clone())
        elif array_size is not None:
            self.array_size = array_size
            self._values = nn.Parameter(torch.randn(1, array_size))
        else:
            raise ValueError("Must provide either 'values' tensor or 'array_size'")

        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.auto_train = auto_train
        self.learning_rate = learning_rate
        self.training_steps = training_steps

        # Initialize loss function with custom weights
        if loss_weights is None:
            loss_weights = {"structure_weight": 0.1, "smoothness_weight": 0.01}
        self.loss_fn = DifferentiableArrayLoss(**loss_weights)

        # Will be initialized after parameters are created
        self.optimizer = None

        # Learnable position embeddings for each array position
        self.position_embeddings = nn.Parameter(
            torch.randn(self.array_size, embedding_dim)
        )

        # Project query keys to embedding space
        self.key_projection = nn.Linear(1, embedding_dim, bias=False)

        # Initialize position embeddings with meaningful structure
        self._initialize_positions(position_init)

        # Initialize optimizer after all parameters are created
        if self.auto_train:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_positions(self, init_type="uniform"):
        """Initialize position embeddings to have ordered structure."""
        with torch.no_grad():
            if init_type == "uniform":
                # Initialize so positions are roughly linearly spaced in embedding space
                for i in range(self.array_size):
                    # Create a base pattern that reflects position order
                    pos_ratio = i / max(1, self.array_size - 1)
                    self.position_embeddings[i, 0] = (
                        pos_ratio * 2 - 1
                    )  # Primary ordering dimension

                    # Add some structure to other dimensions
                    for j in range(1, self.embedding_dim):
                        phase = 2 * math.pi * j * pos_ratio
                        self.position_embeddings[i, j] = 0.1 * math.sin(phase)

            elif init_type == "binary":
                # Initialize with binary-tree-like structure for binary search properties
                for i in range(self.array_size):
                    binary_repr = format(i, f"0{self.embedding_dim}b")
                    for j, bit in enumerate(binary_repr[: self.embedding_dim]):
                        self.position_embeddings[i, j] = float(bit) * 2 - 1

    def __getitem__(self, index):
        """
        Magic method for array-like indexing: diff_array[2.5]

        Args:
            index: int, float, slice, list, or torch.Tensor - the index/indices to lookup
                  Can be fractional for interpolation between array elements

        Returns:
            torch.Tensor: The interpolated values at the specified indices
        """
        if isinstance(index, slice):
            # Handle slice notation: diff_array[1:5] or diff_array[::2]
            start, stop, step = index.indices(self.array_size)
            indices = torch.arange(start, stop, step, dtype=torch.float)
            return self._get_internal(indices)

        elif isinstance(index, (list, tuple)):
            # Handle list indexing: diff_array[[1, 2.5, 4]]
            indices = torch.tensor([float(i) for i in index])
            return self._get_internal(indices)

        else:
            # Handle single index: diff_array[2] or diff_array[2.5]
            if isinstance(index, torch.Tensor):
                if index.dim() == 0:
                    index = index.item()
                else:
                    return self._get_internal(index)
            indices = torch.tensor([float(index)])
            result = self._get_internal(indices)
            return result.squeeze(0) if result.dim() > 0 else result

    def __setitem__(self, index, value):
        """
        Magic method for array-like assignment: diff_array[2.5] = 10.0

        Args:
            index: int, float, slice, list, or torch.Tensor - the index/indices to set
            value: float, list, or torch.Tensor - the value(s) to assign
        """
        if isinstance(index, slice):
            # Handle slice assignment: diff_array[1:5] = [10, 20, 30, 40]
            start, stop, step = index.indices(self.array_size)
            indices = torch.arange(start, stop, step, dtype=torch.float)
            self._set_internal(indices, value)

        elif isinstance(index, (list, tuple)):
            # Handle list assignment: diff_array[[1, 2.5, 4]] = [10, 20, 30]
            indices = torch.tensor([float(i) for i in index])
            self._set_internal(indices, value)

        else:
            # Handle single assignment: diff_array[2.5] = 10.0
            if isinstance(index, torch.Tensor):
                if index.dim() == 0:
                    index = index.item()
                else:
                    self._set_internal(index, value)
                    return
            indices = torch.tensor([float(index)])
            self._set_internal(indices, value)

    def __len__(self):
        """Return the length of the array."""
        return self.array_size

    def __iter__(self):
        """Make the array iterable."""
        for i in range(self.array_size):
            yield self[i]

    def __contains__(self, value):
        """Check if a value exists in the array (approximate)."""
        return torch.any(torch.abs(self._values.squeeze() - float(value)) < 1e-6)

    def __repr__(self):
        """String representation of the array."""
        values_str = str(self._values.squeeze().tolist())
        return f"DifferentiableArray({values_str})"

    def __str__(self):
        """Human-readable string representation."""
        return f"DifferentiableArray(size={self.array_size}, values={self._values.squeeze()})"

    def _get_internal(self, indices):
        """
        Internal method for getting values at specified indices.

        Args:
            indices: torch.Tensor - the indices to lookup

        Returns:
            torch.Tensor: The interpolated values at the specified indices
        """
        # Ensure indices is a proper tensor
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor([float(indices)])

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        # Clamp indices to valid range [0, array_size-1]
        indices = torch.clamp(indices, 0, self.array_size - 1)

        # Check if all indices are integers (for exact lookup)
        integer_mask = torch.abs(indices - torch.round(indices)) < 1e-6
        
        if torch.all(integer_mask):
            # All indices are integers - return exact values
            int_indices = torch.round(indices).long()
            return self._values.squeeze(0)[int_indices]
        
        # Check for mixed integer/fractional indices
        results = torch.zeros_like(indices)
        
        # Handle integer indices exactly
        if torch.any(integer_mask):
            int_indices = torch.round(indices[integer_mask]).long()
            results[integer_mask] = self._values.squeeze(0)[int_indices]
        
        # Handle fractional indices with soft attention
        fractional_mask = ~integer_mask
        if torch.any(fractional_mask):
            fractional_indices = indices[fractional_mask]
            
            # Normalize indices to [0, 1] range based on array size
            normalized_indices = fractional_indices / max(1, self.array_size - 1)
            normalized_indices = normalized_indices.unsqueeze(-1)  # Add feature dimension

            # Use forward pass for differentiable lookup
            batch_size = normalized_indices.shape[0]
            values_batch = self._values.expand(batch_size, -1)

            output, _, _ = self.forward(values_batch, normalized_indices)
            results[fractional_mask] = output.squeeze(-1)

        return results

    def _set_internal(self, indices, values):
        """
        Internal method for setting values at specified indices using differentiable training.
        This method now performs self-training to optimize the array structure.

        Args:
            indices: torch.Tensor - the indices to set
            values: float, list, or torch.Tensor - the values to assign
        """
        # Convert values to tensor if needed
        if isinstance(values, (int, float)):
            values = torch.tensor([float(values)])
        elif isinstance(values, (list, tuple)):
            values = torch.tensor([float(val) for val in values])
        elif not isinstance(values, torch.Tensor):
            values = torch.tensor(values)

        if values.dim() == 0:
            values = values.unsqueeze(0)

        # Ensure indices is a proper tensor
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor([float(indices)])

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        # Broadcast values to match indices if needed
        if values.shape[0] == 1 and indices.shape[0] > 1:
            values = values.expand(indices.shape[0])
        elif values.shape[0] != indices.shape[0]:
            raise ValueError(
                f"Shape mismatch: {values.shape[0]} values for {indices.shape[0]} indices"
            )

        # Clamp indices to valid range
        indices = torch.clamp(indices, 0, self.array_size - 1)

        # Normalize indices to [0, 1] range
        normalized_indices = indices / max(1, self.array_size - 1)
        normalized_indices = normalized_indices.unsqueeze(-1)  # Add feature dimension

        # Target values for training
        target_values = values.unsqueeze(-1)  # Shape: (batch_size, 1)

        if self.auto_train and self.optimizer is not None:
            # Self-training: optimize the array to better match the assignment
            self.train()  # Set to training mode

            for step in range(self.training_steps):
                self.optimizer.zero_grad()

                # Forward pass to get current predictions
                batch_size = normalized_indices.shape[0]
                values_batch = self._values.expand(batch_size, -1)

                output, attention_weights, similarities = self.forward(
                    values_batch, normalized_indices
                )

                # Compute loss using the specialized loss function
                total_loss, recon_loss, structure_loss, smoothness_loss = self.loss_fn(
                    output, target_values, self.position_embeddings, similarities
                )

                # Backpropagate and update
                total_loss.backward()
                self.optimizer.step()

                # Optional: early stopping if loss is very small
                if total_loss.item() < 1e-6:
                    break

            self.eval()  # Set back to eval mode
        else:
            # Fallback: direct soft assignment (non-trainable update)
            with torch.no_grad():
                batch_size = normalized_indices.shape[0]
                key_embeddings = self.key_projection(normalized_indices)
                similarities = torch.matmul(key_embeddings, self.position_embeddings.T)
                attention_weights = F.softmax(similarities / self.temperature, dim=-1)

                # Apply soft assignment using attention weights
                for i in range(batch_size):
                    for j in range(self.array_size):
                        weight = attention_weights[i, j].item()
                        if (
                            weight > 0.01
                        ):  # Only update positions with significant attention
                            self._values[0, j] += weight * (
                                values[i] - self._values[0, j]
                            )

    # Keep the old methods for backwards compatibility and explicit API
    def get(self, indices):
        """Explicit get method (same as __getitem__ but more explicit)."""
        return self[indices]

    def set(self, indices, values):
        """Explicit set method (same as __setitem__ but more explicit)."""
        self[indices] = values

    @property
    def values(self):
        """Get the current array values."""
        return self._values.squeeze(0)

    @values.setter
    def values(self, new_values):
        """Set new array values."""
        old_size = self.array_size

        if isinstance(new_values, torch.Tensor):
            if new_values.dim() == 1:
                new_values = new_values.unsqueeze(0)
            # Update array size if it changes
            self.array_size = new_values.shape[-1]
            self._values.data = new_values.clone()
            # Reinitialize position embeddings for new size
            if self.array_size != self.position_embeddings.shape[0]:
                self.position_embeddings = nn.Parameter(
                    torch.randn(self.array_size, self.embedding_dim)
                )
                self._initialize_positions("uniform")
        else:
            new_values_tensor = torch.tensor(new_values).unsqueeze(0)
            self.array_size = new_values_tensor.shape[-1]
            self._values.data = new_values_tensor
            # Reinitialize position embeddings for new size
            if self.array_size != self.position_embeddings.shape[0]:
                self.position_embeddings = nn.Parameter(
                    torch.randn(self.array_size, self.embedding_dim)
                )
                self._initialize_positions("uniform")

        # Reinitialize optimizer if auto_train is enabled and size changed
        if self.auto_train and old_size != self.array_size:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @property
    def shape(self):
        """Return shape as property (torch.Tensor compatibility)."""
        return torch.Size([self.array_size])

    def append(self, value):
        """Add a value to the end of the array (creates new array)."""
        new_values = torch.cat([self._values.squeeze(), torch.tensor([float(value)])])
        return DifferentiableArray(
            new_values, embedding_dim=self.embedding_dim, temperature=self.temperature
        )

    def extend(self, values):
        """Extend the array with multiple values (creates new array)."""
        if not isinstance(values, torch.Tensor):
            values = torch.tensor([float(v) for v in values])
        new_values = torch.cat([self._values.squeeze(), values])
        return DifferentiableArray(
            new_values, embedding_dim=self.embedding_dim, temperature=self.temperature
        )

    def copy(self):
        """Create a copy of the differentiable array."""
        new_array = DifferentiableArray(
            self._values.clone(),
            embedding_dim=self.embedding_dim,
            temperature=self.temperature,
            auto_train=self.auto_train,
            learning_rate=self.learning_rate,
            training_steps=self.training_steps,
        )
        new_array.position_embeddings.data = self.position_embeddings.data.clone()
        new_array.key_projection.weight.data = self.key_projection.weight.data.clone()
        return new_array

    def tolist(self):
        """Convert to regular Python list."""
        return self._values.squeeze().tolist()

    def numpy(self):
        """Convert to numpy array."""
        return self._values.squeeze().detach().numpy()

    def size(self, dim=None):
        """Return size (compatible with torch.Tensor.size())."""
        if dim is None:
            return torch.Size([self.array_size])
        elif dim == 0:
            return self.array_size
        else:
            raise IndexError(f"Dimension {dim} out of range for 1D array")

    def dim(self):
        """Return number of dimensions (always 1)."""
        return 1

    def forward(self, values, key):
        """
        Forward pass of differentiable array lookup.

        Args:
            values: torch.Tensor of shape (batch_size, array_size) - the array values
            key: torch.Tensor of shape (batch_size, 1) - the lookup key

        Returns:
            output: torch.Tensor of shape (batch_size, 1) - interpolated value
            attention_weights: torch.Tensor of shape (batch_size, array_size) - for analysis
            similarities: torch.Tensor of shape (batch_size, array_size) - raw similarities
        """
        batch_size = values.shape[0]

        # Project key to embedding space
        key_embedding = self.key_projection(key)  # (batch_size, embedding_dim)

        # Compute similarities between key and all positions
        # This creates the "binary search" gradient property
        similarities = torch.matmul(
            key_embedding, self.position_embeddings.T
        )  # (batch_size, array_size)

        # Apply temperature scaling for sharpness control
        scaled_similarities = similarities / self.temperature

        # Compute attention weights (soft indexing)
        attention_weights = F.softmax(
            scaled_similarities, dim=-1
        )  # (batch_size, array_size)

        # Compute weighted sum of values
        output = torch.sum(
            attention_weights * values, dim=-1, keepdim=True
        )  # (batch_size, 1)

        return output, attention_weights, similarities

    def get_gradient_direction(self, key, target_position):
        """
        Analyze the gradient direction of the key w.r.t. a target position.
        This helps verify the "binary search" property.
        """
        key = key.clone().requires_grad_(True)
        key_embedding = self.key_projection(key)

        # Compute similarity to target position
        target_embedding = self.position_embeddings[target_position]
        similarity = torch.dot(key_embedding.squeeze(), target_embedding)

        # Compute gradient
        similarity.backward()
        gradient_direction = key.grad

        return gradient_direction

    def visualize_position_space(self):
        """Return position embeddings for visualization."""
        return self.position_embeddings.detach()

    def set_training_mode(
        self, auto_train=True, learning_rate=None, training_steps=None
    ):
        """Enable or disable auto-training and update parameters."""
        self.auto_train = auto_train

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if training_steps is not None:
            self.training_steps = training_steps

        # Reinitialize optimizer with new learning rate if needed
        if self.auto_train:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = None

    def get_training_info(self):
        """Get current training configuration."""
        return {
            "auto_train": self.auto_train,
            "learning_rate": self.learning_rate,
            "training_steps": self.training_steps,
            "has_optimizer": self.optimizer is not None,
        }


class DifferentiableArrayLoss(nn.Module):
    """
    Custom loss functions for training the differentiable array.
    """

    def __init__(self, structure_weight=0.1, smoothness_weight=0.01):
        super().__init__()
        self.structure_weight = structure_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, output, target, position_embeddings, similarities):
        """
        Compute loss with regularization terms.

        Args:
            output: predicted values
            target: target values
            position_embeddings: the learned position embeddings
            similarities: raw similarity scores for regularization
        """
        # Main reconstruction loss
        recon_loss = F.mse_loss(output, target)

        # Structure loss: encourage position embeddings to maintain order
        structure_loss = 0
        for i in range(len(position_embeddings) - 1):
            # Adjacent positions should be similar but distinguishable
            diff = position_embeddings[i + 1] - position_embeddings[i]
            structure_loss += torch.norm(diff) ** 2
        structure_loss = structure_loss / len(position_embeddings)

        # Smoothness loss: encourage smooth attention distributions
        smoothness_loss = torch.var(similarities, dim=-1).mean()

        total_loss = (
            recon_loss
            + self.structure_weight * structure_loss
            + self.smoothness_weight * smoothness_loss
        )

        return total_loss, recon_loss, structure_loss, smoothness_loss
