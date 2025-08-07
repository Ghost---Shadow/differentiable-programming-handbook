import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-2
num_epochs = 1000
log_interval = 200

# Data setup - using complex numbers
data_real = torch.tensor([54, 87, 67, 29, 85], dtype=torch.float32, device=device)
target_real = torch.tensor([54, 67, 87, 85, 29], dtype=torch.float32, device=device)

# Convert to complex (starting with zero imaginary part)
data = torch.complex(data_real, torch.zeros_like(data_real))
target_data = torch.complex(target_real, torch.zeros_like(target_real))
data_len = len(data)


def complex_softmax(x, dim=-1):
    """Softmax for complex matrices - applies to magnitude while preserving phase"""
    # For stability, subtract max before exponentiating
    x_real = x.real
    x_real = x_real - x_real.max(dim=dim, keepdim=True)[0]

    # Apply softmax to real part (treating as logits)
    exp_x = torch.exp(x_real)
    soft_real = exp_x / exp_x.sum(dim=dim, keepdim=True)

    # Add phase information
    phases = x.angle()
    return soft_real * torch.exp(1j * phases)


def iterate_over_complex(data, nexts):
    """
    Iterate through complex data following the permutation defined by nexts matrix.
    """
    data_len = data.shape[0]
    current_pos = torch.zeros(data_len, device=data.device, dtype=torch.complex64)
    current_pos[0] = 1.0 + 0j

    result = torch.zeros(data_len, device=data.device, dtype=data.dtype)

    for step in range(data_len):
        # Complex dot product
        current_value = torch.sum(torch.conj(current_pos) * data)
        result[step] = current_value
        # Move to next position using complex matrix multiplication
        current_pos = current_pos @ nexts

    return result


class ComplexBistableLoss(nn.Module):
    """Bistable loss for complex numbers - applies to magnitude"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        mag = x.abs()
        a = mag**2
        b = (mag - 1) ** 2
        return a * b


class MobiusPermutationLoss(nn.Module):
    def __init__(self, cycle_length=5, mobius_twist=True, cycle_weight=1.0):
        super().__init__()
        self.cycle_length = cycle_length
        self.mobius_twist = mobius_twist
        self.cycle_weight = cycle_weight
        self.bistable_loss = ComplexBistableLoss()

    def forward(self, P):
        loss = 0

        # Work with magnitude for constraints
        P_mag = P.abs()
        P_mag_square = P_mag**2

        # Row sum constraint (each row should sum to 1)
        axis_1_sum = P_mag_square.sum(dim=1)
        loss += F.mse_loss(axis_1_sum, torch.ones_like(axis_1_sum))

        # Column sum constraint
        axis_0_sum = P_mag_square.sum(dim=0)
        loss += F.mse_loss(axis_0_sum, torch.ones_like(axis_0_sum))

        # Bistable loss - encourages elements to be 0 or 1 in magnitude
        loss += self.bistable_loss(P).sum() * 0.1

        # Entropy regularization to encourage exploration
        entropy = -(P_mag * torch.log(P_mag + 1e-8)).sum()
        loss -= entropy * 0.01  # Small negative weight to encourage diversity

        # Cycle/Möbius constraint
        Q = P
        for _ in range(self.cycle_length - 1):
            Q = Q @ P  # Note: changed order for consistency

        if self.mobius_twist:
            # Möbius constraint: P^n = -I (need to go around twice to return)
            # We'll use a phase shift that accumulates to π
            target_phase = torch.tensor(
                -1.0 + 0j, device=P.device, dtype=torch.complex64
            )  # -1
            target = target_phase * torch.eye(
                Q.shape[0], device=P.device, dtype=torch.complex64
            )
        else:
            # Regular cycle: P^n = I
            target = torch.eye(Q.shape[0], device=P.device, dtype=torch.complex64)

        # Complex MSE loss
        cycle_loss = F.mse_loss(Q.real, target.real)
        cycle_loss += F.mse_loss(Q.imag, target.imag)
        loss += cycle_loss * self.cycle_weight

        return loss


# Initialize with a better starting point
# Create a circular shift pattern with phases
nexts_indices = torch.tensor([2, 4, 1, 0, 3])  # Known good permutation
nexts_real = torch.zeros((data_len, data_len), device=device)

# Add some noise to break symmetry
for i in range(data_len):
    nexts_real[i, nexts_indices[i]] = 10.0  # Strong initial preference
    # Add small random values to other positions
    nexts_real[i] += torch.randn(data_len, device=device) * 0.1

# Add Möbius phase structure
# Each step accumulates π/5 phase, so after 5 steps we have π phase (sign flip)
phase_per_step = np.pi / data_len
phase_matrix = torch.zeros((data_len, data_len), device=device)

for i in range(data_len):
    # Add phase that accumulates as we traverse
    phase_matrix[i, nexts_indices[i]] = phase_per_step * i

# Create complex nexts matrix
nexts = torch.complex(nexts_real, torch.zeros_like(nexts_real))
nexts = nexts * torch.exp(1j * phase_matrix)
nexts.requires_grad_(True)

# Setup optimizer and loss
optimizer = optim.Adam([nexts], lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)

permute_loss_fn = MobiusPermutationLoss(
    cycle_length=data_len,
    mobius_twist=True,  # Set to True for Möbius, False for regular
    cycle_weight=0.5,  # Start with lower weight, can increase later
).to(device)


def train_step(epoch):
    optimizer.zero_grad()

    # Apply complex softmax to get valid probability matrix
    prob_nexts = complex_softmax(nexts, dim=1)

    # Forward pass
    predicted_data = iterate_over_complex(data, prob_nexts)

    # Compute losses (use real part for data loss since target is real)
    data_loss = F.mse_loss(predicted_data.real, target_data.real)

    # Gradually increase cycle weight
    current_cycle_weight = min(1.0, epoch / 5000)  # Ramp up over first 5000 epochs
    permute_loss_fn.cycle_weight = current_cycle_weight

    regularization_loss = permute_loss_fn(prob_nexts)

    total_loss = data_loss + regularization_loss * 0.1  # Scale down regularization

    # Backward pass
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_([nexts], max_norm=10.0)

    optimizer.step()

    return total_loss, predicted_data, prob_nexts, data_loss.item()


def check_mobius_property(P):
    """Check if P^n = -I (Möbius) or P^n = I (regular)"""
    Q = P
    for _ in range(data_len - 1):
        Q = Q @ P

    # Check if Q ≈ -I
    neg_eye = -torch.eye(data_len, dtype=torch.complex64, device=device)
    mobius_error = torch.norm(Q - neg_eye).item()

    # Check if Q ≈ I
    eye = torch.eye(data_len, dtype=torch.complex64, device=device)
    regular_error = torch.norm(Q - eye).item()

    # Check if diagonal has consistent phase
    diagonal_phases = Q.diagonal().angle() / np.pi
    phase_consistency = diagonal_phases.std().item()

    return mobius_error, regular_error, phase_consistency


# Training loop
print("Training Complex Permutation with Möbius Properties")
print("=" * 90)
print(
    "| Epoch | Total Loss | Data Loss | Möbius Err | Regular Err | Phase Std |   Predicted (Real)  |"
)
print("-" * 90)

best_loss = float("inf")
for epoch in range(num_epochs):
    loss, predicted_data, prob_nexts, data_loss = train_step(epoch)

    # Learning rate scheduling
    scheduler.step(loss)

    if epoch % log_interval == 0:
        with torch.no_grad():
            # Check Möbius property
            mobius_err, regular_err, phase_std = check_mobius_property(prob_nexts)

            print(
                f"| {epoch:5d} | {loss.item():10.4f} | {data_loss:9.4f} | {mobius_err:10.4f} | "
                f"{regular_err:11.4f} | {phase_std:9.4f} | {torch.round(predicted_data.real).cpu().numpy()} |"
            )

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_nexts = nexts.detach().clone()

print("\n" + "=" * 90)
print("Final Analysis:")
print("-" * 90)

# Use best nexts for final analysis
nexts = best_nexts

with torch.no_grad():
    prob_nexts_final = complex_softmax(nexts, dim=1)

    # Get the permutation structure
    print("\nLearned permutation (argmax of magnitude):")
    perm_indices = torch.argmax(prob_nexts_final.abs(), dim=1)
    print(f"Indices: {perm_indices.cpu().numpy()}")

    print("\nPermutation matrix magnitude:")
    print(prob_nexts_final.abs().cpu().numpy().round(3))

    print("\nPermutation matrix phase (in units of π):")
    print((prob_nexts_final.angle() / np.pi).cpu().numpy().round(3))

    # Compute P^5
    P_power = prob_nexts_final
    for _ in range(data_len - 1):
        P_power = P_power @ prob_nexts_final

    print("\nP^5 diagonal (should be close to -1 for Möbius):")
    diag = P_power.diagonal()
    for i in range(len(diag)):
        mag = diag[i].abs().item()
        phase = diag[i].angle().item() / np.pi
        print(f"  Position {i}: magnitude={mag:.4f}, phase={phase:.4f}π")

    # Compute P^10
    P_power_10 = P_power
    for _ in range(data_len):
        P_power_10 = P_power_10 @ prob_nexts_final

    print("\nP^10 diagonal (should be close to 1 for Möbius):")
    diag10 = P_power_10.diagonal()
    for i in range(len(diag10)):
        mag = diag10[i].abs().item()
        phase = diag10[i].angle().item() / np.pi
        print(f"  Position {i}: magnitude={mag:.4f}, phase={phase:.4f}π")

    # Trace the path with phase accumulation
    print("\n" + "=" * 90)
    print("Traversing the data with phase accumulation:")
    print("-" * 90)

    pos = torch.zeros(data_len, dtype=torch.complex64, device=device)
    pos[0] = 1.0 + 0j

    print("First cycle:")
    for step in range(data_len):
        idx = torch.argmax(pos.abs()).item()
        value = data[idx].real.item()
        phase = pos[idx].angle().item() / np.pi
        print(f"  Step {step}: Position {idx} (value={value:.0f}), Phase={phase:.3f}π")
        pos = pos @ prob_nexts_final

    print("\nAfter first cycle:")
    idx = torch.argmax(pos.abs()).item()
    phase = pos[idx].angle().item() / np.pi
    print(f"  Back at position {idx} with phase {phase:.3f}π")

    if abs(phase - 1.0) < 0.1 or abs(phase + 1.0) < 0.1:
        print("  ✓ Möbius twist detected! Phase flipped by ~π")
    else:
        print(f"  Phase shift: {phase:.3f}π (target: ±1.0π for Möbius)")

    print("\nSecond cycle:")
    for step in range(data_len):
        idx = torch.argmax(pos.abs()).item()
        value = data[idx].real.item()
        phase = pos[idx].angle().item() / np.pi
        print(
            f"  Step {step+data_len}: Position {idx} (value={value:.0f}), Phase={phase:.3f}π"
        )
        pos = pos @ prob_nexts_final

    print("\nAfter second cycle:")
    idx = torch.argmax(pos.abs()).item()
    phase = pos[idx].angle().item() / np.pi
    print(f"  Back at position {idx} with phase {phase:.3f}π")

    if abs(phase) < 0.1:
        print("  ✓ Completed double loop - returned to original phase!")
