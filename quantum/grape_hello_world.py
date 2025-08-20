import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Physical parameters (working in units where hbar = 1)
qubit_frequency = 0.0  # Set qubit frequency to 0 for simplicity (rotating frame)
gate_time = 1.0  # Dimensionless gate time
num_time_steps = 20  # Time discretization
time_step = gate_time / num_time_steps

# Quantum operators (2x2 matrices)
bit_flip_matrix = torch.tensor(
    [[0, 1], [1, 0]], dtype=torch.complex64, device=device
)  # X rotation
phase_flip_matrix = torch.tensor(
    [[1, 0], [0, -1]], dtype=torch.complex64, device=device
)  # Z rotation
identity_matrix = torch.eye(2, dtype=torch.complex64, device=device)

# Initial and target states
initial_state = torch.tensor([1, 0], dtype=torch.complex64, device=device)  # |0⟩
target_state = torch.tensor([0, 1], dtype=torch.complex64, device=device)  # |1⟩


class GRAPEOptimizer(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        # Initialize pulse amplitudes (dimensionless units)
        # Start with a worse initial guess to see optimization in action
        initial_amplitude = 2.0  # Deliberately bad guess (instead of π ≈ 3.14)
        self.amplitudes = nn.Parameter(
            torch.full(
                (num_steps,), initial_amplitude, dtype=torch.float32, device=device
            )
        )

    def forward(self, initial_quantum_state):
        """
        Simulate quantum evolution under the pulse sequence
        """
        quantum_state = initial_quantum_state.clone()

        for step in range(num_time_steps):
            # Time-dependent energy operator: combines qubit energy + control energy
            # In rotating frame, qubit_freq = 0, so only control energy remains
            energy_operator = (self.amplitudes[step] / 2) * bit_flip_matrix

            # Time evolution operator: U = exp(-i * energy_operator * time_step)
            evolution_operator = torch.matrix_exp(-1j * energy_operator * time_step)

            # Apply evolution
            quantum_state = torch.matmul(evolution_operator, quantum_state)

        return quantum_state


def fidelity_loss(final_state, target_state):
    """
    Compute 1 - fidelity as loss function
    Fidelity = |⟨target|final⟩|²
    """
    overlap = torch.dot(target_state.conj(), final_state)
    fidelity = torch.abs(overlap) ** 2
    return 1.0 - fidelity


def run_grape_optimization():
    # Initialize model and optimizer
    model = GRAPEOptimizer(num_time_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # Use Adam with reasonable LR

    # Training loop
    loss_history = []
    fidelity_history = []

    print("Starting GRAPE optimization...")
    print("Epoch | Loss      | Fidelity  | Max Amp")
    print("-" * 40)

    for epoch in range(500):
        optimizer.zero_grad()

        # Forward pass
        final_state = model(initial_state)
        loss = fidelity_loss(final_state, target_state)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate fidelity for monitoring
        with torch.no_grad():
            current_fidelity = 1.0 - loss.item()
            loss_history.append(loss.item())
            fidelity_history.append(current_fidelity)

            if epoch % 50 == 0 or epoch < 10:
                max_amplitude = model.amplitudes.max().item()
                print(
                    f"{epoch:5d} | {loss.item():.6f} | {current_fidelity:.6f} | {max_amplitude:8.3f}"
                )

        # Early stopping
        if current_fidelity > 0.999:
            print(f"Converged at epoch {epoch} with fidelity {current_fidelity:.6f}")
            break

    return model, loss_history, fidelity_history


def plot_results(model, loss_history, fidelity_history):
    """
    Plot optimization results and final pulse shape
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot pulse shape
    time_points = np.linspace(0, gate_time, num_time_steps)
    optimized_amplitudes = model.amplitudes.detach().cpu().numpy()

    ax1.plot(time_points, optimized_amplitudes, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Optimized Pulse Shape")
    ax1.grid(True, alpha=0.3)

    # Plot loss
    # Clip loss values to avoid log(negative) warnings
    clipped_loss = np.maximum(loss_history, 1e-10)
    ax2.semilogy(clipped_loss)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (1 - Fidelity)")
    ax2.set_title("Optimization Loss")
    ax2.grid(True, alpha=0.3)

    # Plot fidelity
    ax3.plot(fidelity_history)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Fidelity")
    ax3.set_title("Gate Fidelity")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Compare with rectangular pulse
    rectangular_amplitude = np.pi / gate_time  # Initial guess
    rectangular_amplitudes = np.full(num_time_steps, rectangular_amplitude)

    ax4.plot(
        time_points,
        rectangular_amplitudes,
        "r--",
        label="Rectangular (initial)",
        linewidth=2,
    )
    ax4.plot(
        time_points, optimized_amplitudes, "b-", label="GRAPE optimized", linewidth=2
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Pulse Shape Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grape_optimization_results.png", dpi=300, bbox_inches="tight")
    print("Results saved to 'grape_optimization_results.png'")


def analyze_final_state(model):
    """
    Analyze the final quantum state achieved
    """
    with torch.no_grad():
        final_state = model(initial_state)

        # Calculate final state probabilities
        state_probabilities = torch.abs(final_state) ** 2

        print("\nFinal State Analysis:")
        print(f"Final state: [{final_state[0]:.4f}, {final_state[1]:.4f}]")
        print(
            f"Probabilities: |0⟩: {state_probabilities[0]:.6f}, |1⟩: {state_probabilities[1]:.6f}"
        )

        # Calculate overlap with target
        overlap = torch.dot(target_state.conj(), final_state)
        achieved_fidelity = torch.abs(overlap) ** 2
        print(f"State fidelity: {achieved_fidelity:.8f}")

        # Calculate total pulse area (should be pi for X-gate)
        total_pulse_area = torch.sum(model.amplitudes) * time_step
        print(f"Total pulse area: {total_pulse_area:.4f} (pi = {np.pi:.4f})")


if __name__ == "__main__":
    # Run optimization
    model, loss_history, fidelity_history = run_grape_optimization()

    # Analyze results
    analyze_final_state(model)

    # Plot results
    plot_results(model, loss_history, fidelity_history)

    print(f"\nOptimized amplitudes:")
    optimized_amplitudes = model.amplitudes.detach().cpu().numpy()
    for step_idx, amplitude in enumerate(optimized_amplitudes):
        print(f"  t_{step_idx+1:2d}: {amplitude:6.3f}")
