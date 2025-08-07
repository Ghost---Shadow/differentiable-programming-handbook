import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiableArray(nn.Module):
    def __init__(
        self,
        values=None,
        array_size=None,
        embedding_dim=64,
        temperature=1.0,
        auto_train=True,
        learning_rate=0.001,
        training_steps=5,
        loss_weights=None,
    ):
        super().__init__()

        if values is not None:
            if values.dim() == 1:
                values = values.unsqueeze(0)
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

        if loss_weights is None:
            loss_weights = {"structure_weight": 0.1, "smoothness_weight": 0.01}
        self.loss_fn = DifferentiableArrayLoss(**loss_weights)

        self.position_embeddings = nn.Parameter(
            torch.randn(self.array_size, embedding_dim)
        )
        self.key_projection = nn.Linear(1, embedding_dim, bias=False)

        self._initialize_positions()

        self.optimizer = None
        if self.auto_train:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _initialize_positions(self):
        with torch.no_grad():
            for i in range(self.array_size):
                pos_ratio = i / max(1, self.array_size - 1)
                self.position_embeddings[i, 0] = pos_ratio * 2 - 1
                for j in range(1, self.embedding_dim):
                    phase = 2 * math.pi * j * pos_ratio
                    self.position_embeddings[i, j] = 0.1 * math.sin(phase)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            indices = torch.tensor([float(i) for i in index])
            return self._get_internal(indices)
        else:
            if isinstance(index, torch.Tensor):
                if index.dim() == 0:
                    index = index.item()
                else:
                    return self._get_internal(index)
            indices = torch.tensor([float(index)])
            result = self._get_internal(indices)
            return result.squeeze(0) if result.dim() > 0 else result

    def __setitem__(self, index, value):
        if isinstance(index, (list, tuple)):
            indices = torch.tensor([float(i) for i in index])
            self._set_internal(indices, value)
        else:
            if isinstance(index, torch.Tensor):
                if index.dim() == 0:
                    index = index.item()
                else:
                    self._set_internal(index, value)
                    return
            indices = torch.tensor([float(index)])
            self._set_internal(indices, value)

    def _get_internal(self, indices):
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor([float(indices)])
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        # Only clamp if gradients are not required for indices
        if not indices.requires_grad:
            indices = torch.max(
                torch.tensor(0.0),
                torch.min(indices, torch.tensor(float(self.array_size - 1))),
            )

        # If gradients are required, always use attention mechanism
        if indices.requires_grad:
            batch_size = indices.shape[0]
            normalized_indices = indices / max(1, self.array_size - 1)
            normalized_indices = normalized_indices.unsqueeze(-1)
            values_batch = self._values.expand(batch_size, -1)
            output, _, _ = self.forward(values_batch, normalized_indices)
            return output.squeeze(-1)

        # Integer indices - exact lookup (only when no gradients required)
        integer_mask = torch.abs(indices - torch.round(indices)) < 1e-6
        if torch.all(integer_mask):
            int_indices = torch.round(indices).long()
            return self._values.squeeze(0)[int_indices]

        # Mixed indices (no gradients required)
        results = torch.zeros_like(indices, dtype=self._values.dtype)

        if torch.any(integer_mask):
            int_indices = torch.round(indices[integer_mask]).long()
            results[integer_mask] = self._values.squeeze(0)[int_indices]

        fractional_mask = ~integer_mask
        if torch.any(fractional_mask):
            fractional_indices = indices[fractional_mask]
            if self.temperature > 0.5:
                # Attention mechanism
                batch_size = fractional_indices.shape[0]
                normalized_indices = fractional_indices / max(1, self.array_size - 1)
                normalized_indices = normalized_indices.unsqueeze(-1)
                values_batch = self._values.expand(batch_size, -1)
                output, _, _ = self.forward(values_batch, normalized_indices)
                results[fractional_mask] = output.squeeze(-1)
            else:
                # Linear interpolation
                fractional_results = torch.zeros_like(fractional_indices)
                for i, idx in enumerate(fractional_indices):
                    floor_idx = torch.floor(idx).long()
                    ceil_idx = torch.clamp(
                        floor_idx + 1, max=self.array_size - 1
                    ).long()
                    frac = idx - floor_idx
                    val_floor = self._values.squeeze(0)[floor_idx]
                    val_ceil = self._values.squeeze(0)[ceil_idx]
                    fractional_results[i] = val_floor + frac * (val_ceil - val_floor)
                results[fractional_mask] = fractional_results

        return results

    def _set_internal(self, indices, values):
        if isinstance(values, (int, float)):
            values = torch.tensor([float(values)])
        elif isinstance(values, (list, tuple)):
            values = torch.tensor([float(val) for val in values])
        elif not isinstance(values, torch.Tensor):
            values = torch.tensor(values)

        if values.dim() == 0:
            values = values.unsqueeze(0)

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor([float(indices)])
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        if values.shape[0] == 1 and indices.shape[0] > 1:
            values = values.expand(indices.shape[0])
        elif values.shape[0] != indices.shape[0]:
            raise ValueError(
                f"Shape mismatch: {values.shape[0]} values for {indices.shape[0]} indices"
            )

        # Only clamp if gradients are not required for indices
        if not indices.requires_grad:
            indices = torch.max(
                torch.tensor(0.0),
                torch.min(indices, torch.tensor(float(self.array_size - 1))),
            )

        # Integer indices with low temperature - exact assignment
        integer_mask = torch.abs(indices - torch.round(indices)) < 1e-6
        if torch.all(integer_mask) and self.temperature <= 0.5:
            with torch.no_grad():
                int_indices = torch.round(indices).long()
                for i, idx in enumerate(int_indices):
                    self._values[0, idx] = values[i]
            return

        # Soft assignment with training
        normalized_indices = indices / max(1, self.array_size - 1)
        normalized_indices = normalized_indices.unsqueeze(-1)
        target_values = values.unsqueeze(-1)

        if self.auto_train and self.optimizer is not None:
            self.train()
            for step in range(self.training_steps):
                self.optimizer.zero_grad()
                batch_size = normalized_indices.shape[0]
                values_batch = self._values.expand(batch_size, -1)
                output, _, similarities = self.forward(values_batch, normalized_indices)
                total_loss, _, _, _ = self.loss_fn(
                    output, target_values, self.position_embeddings, similarities
                )
                total_loss.backward()
                self.optimizer.step()
                if total_loss.item() < 1e-6:
                    break
            self.eval()

    def get(self, indices):
        return self[indices]

    @property
    def values(self):
        return self._values.squeeze(0)

    def forward(self, values, key):
        key_embedding = self.key_projection(key)
        similarities = torch.matmul(key_embedding, self.position_embeddings.T)
        scaled_similarities = similarities / self.temperature
        attention_weights = F.softmax(scaled_similarities, dim=-1)
        output = torch.sum(attention_weights * values, dim=-1, keepdim=True)
        return output, attention_weights, similarities


class DifferentiableArrayLoss(nn.Module):
    def __init__(self, structure_weight=0.1, smoothness_weight=0.01):
        super().__init__()
        self.structure_weight = structure_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, output, target, position_embeddings, similarities):
        recon_loss = F.mse_loss(output, target)

        structure_loss = 0
        for i in range(len(position_embeddings) - 1):
            diff = position_embeddings[i + 1] - position_embeddings[i]
            structure_loss += torch.norm(diff) ** 2
        structure_loss = structure_loss / len(position_embeddings)

        smoothness_loss = torch.var(similarities, dim=-1).mean()

        total_loss = (
            recon_loss
            + self.structure_weight * structure_loss
            + self.smoothness_weight * smoothness_loss
        )

        return total_loss, recon_loss, structure_loss, smoothness_loss
