"""
Pi Estimation using Differentiable Programming

This module estimates the value of π using gradient descent and differentiable
programming techniques without relying on built-in trigonometric functions.

Core Concept:
-------------
π is treated as a learnable parameter. A point at (r, 0) is rotated N times using
a rotation matrix parameterized by the learned π. After N rotations (completing a
full circle), the point should return to (r, 0).

Loss Function:
--------------
The L2 distance between the final position and the starting point (r, 0) serves as
the loss. If π is overestimated, the final position overshoots; if underestimated,
it undershoots. Backpropagation optimizes π to minimize this distance.

Circle Geometry:
----------------
Instead of using built-in sin/cos functions, we compute rotation matrices using:
- The unit circle constraint: x² + y² = 1
- Differential equations: dx/dθ = -y, dy/dθ = x
- Numerical integration (Heun's method) with normalization to stay on the circle

This demonstrates how complex mathematical operations can be learned from first
principles using differentiable programming.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def compute_sin_cos_from_circle(angle):
    """
    Compute sin and cos without using built-in functions.

    Uses the unit circle constraint x^2 + y^2 = 1 and differential approach.
    For a point rotating on the unit circle, we have:
        dx/dθ = -y
        dy/dθ = x

    We approximate this with small steps using an improved Euler method.
    """
    # Number of small steps to approximate the angle
    # Using fewer steps for efficiency but normalizing to maintain accuracy
    n_steps = 10
    dt = angle / n_steps

    # Start at (1, 0) which corresponds to angle 0
    x = torch.ones_like(angle)
    y = torch.zeros_like(angle)

    # Use improved Euler method (Heun's method) to solve the differential equation
    # This traces a path on the unit circle
    for _ in range(n_steps):
        # Predictor step
        dx_pred = -y * dt
        dy_pred = x * dt

        # Corrector step (average of slopes at beginning and predicted end)
        x_pred = x + dx_pred
        y_pred = y + dy_pred

        dx = 0.5 * (-y - y_pred) * dt
        dy = 0.5 * (x + x_pred) * dt

        x = x + dx
        y = y + dy

        # Normalize to keep on unit circle (enforce x^2 + y^2 = 1)
        norm = torch.sqrt(x * x + y * y)
        x = x / norm
        y = y / norm

    return x, y  # x = cos(angle), y = sin(angle)


def rotation_matrix(angle):
    """Create a 2D rotation matrix for the given angle using circle geometry."""
    cos_theta, sin_theta = compute_sin_cos_from_circle(angle)
    return torch.stack(
        [torch.stack([cos_theta, -sin_theta]), torch.stack([sin_theta, cos_theta])]
    )


def estimate_pi(
    initial_guess=3.0,
    n_rotations=1000,
    learning_rate=0.01,
    n_iterations=10000,
    radius=1.0,
):
    """
    Estimate pi by learning it as a parameter through gradient descent.

    Args:
        n_rotations: Number of rotations to complete a full circle
        learning_rate: Learning rate for optimization
        n_iterations: Number of optimization steps
        radius: Radius of the circular path

    Returns:
        Tuple of (estimated pi value, loss history)
    """
    # Initialize pi as a learnable parameter with a reasonable starting guess
    pi_learned = nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32))

    # Starting point at (r, 0)
    start_point = torch.tensor([radius, 0.0], dtype=torch.float32)

    # Optimizer
    optimizer = optim.Adam([pi_learned], lr=learning_rate)

    # Track loss history
    loss_history = []

    # Training loop
    for iteration in tqdm(range(n_iterations)):
        optimizer.zero_grad()

        # Calculate the rotation angle: 2*pi/N
        angle_per_rotation = 2 * pi_learned / n_rotations

        # Compute the rotation matrix once
        rot_matrix = rotation_matrix(angle_per_rotation)

        # Multiply the matrix with itself N-1 times to get the full rotation
        full_rotation = rot_matrix.clone()
        for _ in range(n_rotations - 1):
            full_rotation = torch.matmul(full_rotation, rot_matrix)

        # Apply the full rotation to the starting point
        current_point = torch.matmul(full_rotation, start_point)

        # Calculate L2 loss between final point and starting point
        loss = torch.norm(current_point - start_point)

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Track loss
        loss_history.append(loss.item())

        # Print progress
        if (iteration + 1) % 200 == 0:
            print(
                f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss.item():.6f}, Pi estimate: {pi_learned.item():.6f}"
            )

    return pi_learned.item(), loss_history


def plot_loss_curve(loss_history, filename="pi_estimation_loss.png"):
    """Plot and save the loss curve during training."""
    # Save to the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss (L2 Distance)", fontsize=12)
    plt.title("Pi Estimation Training Loss Curve", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Loss curve saved to {filepath}")
    plt.close()


def plot_circle_with_learned_pi(estimated_pi, filename="pi_estimation_circle.png"):
    """
    Draw a circle using the learned pi value and the compute_sin_cos_from_circle function.
    This visualizes how well our learned pi approximates a true circle.
    """
    # Save to the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    # Number of points to draw the circle
    n_points = 100

    # Use the learned pi to compute angles
    angles = [2 * estimated_pi * i / n_points for i in range(n_points + 1)]

    # Compute points on the circle using our custom sin/cos function
    x_coords = []
    y_coords = []

    for angle in angles:
        angle_tensor = torch.tensor(angle, dtype=torch.float32)
        cos_val, sin_val = compute_sin_cos_from_circle(angle_tensor)
        x_coords.append(cos_val.item())
        y_coords.append(sin_val.item())

    # Also compute with true pi for comparison
    true_angles = np.linspace(0, 2 * np.pi, n_points + 1)
    x_true = np.cos(true_angles)
    y_true = np.sin(true_angles)

    # Plot both circles
    plt.figure(figsize=(10, 10))
    plt.plot(
        x_coords, y_coords, "b-", linewidth=2, label=f"Learned π = {estimated_pi:.6f}"
    )
    plt.plot(
        x_true, y_true, "r--", linewidth=1.5, alpha=0.7, label=f"True π = {np.pi:.6f}"
    )
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title(
        "Circle Drawn Using Learned Pi\n(No Built-in Sin/Cos Functions)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Circle plot saved to {filepath}")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Estimating Pi using Differentiable Programming")
    print("=" * 70)
    print(
        "Using circle equation x^2 + y^2 = 1 to compute rotations (no built-in sin/cos)"
    )
    print(f"True value of Pi: {torch.pi:.6f}\n")

    # Estimate pi using 96-sided polygon for historical accuracy
    estimated_pi, loss_history = estimate_pi(
        initial_guess=3.0,
        n_rotations=96,
        learning_rate=0.001,
        n_iterations=1000,
    )

    # Sanity check
    # estimated_pi, loss_history = estimate_pi(
    #     initial_guess=torch.pi,
    #     n_rotations=1000,
    #     learning_rate=0.0000001,
    #     n_iterations=1000,
    # )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Final estimated Pi: {estimated_pi:.6f}")
    print(f"True Pi (PyTorch):  {torch.pi:.6f}")
    print(f"Absolute Error:     {abs(estimated_pi - torch.pi):.6f}")
    print(f"Relative Error:     {abs(estimated_pi - torch.pi) / torch.pi * 100:.4f}%")

    # Create visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)
    plot_loss_curve(loss_history)
    plot_circle_with_learned_pi(estimated_pi)
    print("\nAll plots saved successfully!")
