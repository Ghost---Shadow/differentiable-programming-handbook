#!/usr/bin/env python3
import torch
import torch.nn as nn

# Physical constants
SUN_MASS = 1.9884e30  # kg (nominal solar mass)
GRAVITATIONAL_CONSTANT = 6.6743e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
ASTRONOMICAL_UNIT = 1.495978707e11  # m (exact as of IAU 2012)
EARTH_MASS = 5.9722e24  # kg (NASA Earth Fact Sheet)
URANUS_RADIUS = 2.5559e7  # m (equatorial radius at 1 bar)
SPHERE_MOMENT_INERTIA_COEFFICIENT = 2 / 5

# Current Uranus observations
URANUS_CURRENT_MASS = 8.68e25  # kg (NASA Uranus Fact Sheet)
URANUS_CURRENT_SEMI_MAJOR_AXIS = 19.19126393 * ASTRONOMICAL_UNIT  # m (NASA)
URANUS_CURRENT_ORBITAL_VELOCITY = torch.sqrt(
    torch.tensor(
        GRAVITATIONAL_CONSTANT * SUN_MASS / URANUS_CURRENT_SEMI_MAJOR_AXIS,
        dtype=torch.float64
    )
)  # m/s
URANUS_CURRENT_TILT = torch.tensor(98.0, dtype=torch.float64)  # degrees


class UranusImpactModel(nn.Module):
    """Model for Uranus impact scenario with learnable parameters"""

    def __init__(self):
        super().__init__()

        # Learnable parameters - initialize with reasonable guesses
        self.impactor_mass_earth_units = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.relative_velocity_km_per_s = nn.Parameter(torch.tensor(10.0, dtype=torch.float64))
        self.impact_angle_degrees = nn.Parameter(torch.tensor(45.0, dtype=torch.float64))
        self.original_semi_major_axis_au = nn.Parameter(torch.tensor(19.0, dtype=torch.float64))
        self.original_rotation_period_hours = nn.Parameter(torch.tensor(12.0, dtype=torch.float64))

    def forward(self):
        """Forward pass computing final orbital velocity and tilt"""
        # Convert learnable parameters to SI units
        impactor_mass = self.impactor_mass_earth_units * EARTH_MASS  # kg
        original_uranus_mass = URANUS_CURRENT_MASS - impactor_mass  # kg
        relative_velocity = self.relative_velocity_km_per_s * 1000  # m/s
        impact_angle_radians = torch.deg2rad(self.impact_angle_degrees)
        original_semi_major_axis = self.original_semi_major_axis_au * ASTRONOMICAL_UNIT  # m
        original_rotation_period = self.original_rotation_period_hours * 3600  # seconds

        # Calculate original orbital velocity before impact
        original_orbital_velocity = torch.sqrt(
            GRAVITATIONAL_CONSTANT * SUN_MASS / original_semi_major_axis
        )

        # Decompose impact velocity into tangential and radial components
        tangential_velocity = relative_velocity * torch.cos(impact_angle_radians)
        radial_velocity = relative_velocity * torch.sin(impact_angle_radians)

        # Calculate orbital velocity change from impact
        velocity_change = (impactor_mass * tangential_velocity) / (
            original_uranus_mass + impactor_mass
        )
        final_orbital_velocity = original_orbital_velocity + velocity_change

        # Calculate angular momentum delivered by impact
        delivered_angular_momentum = impactor_mass * radial_velocity * URANUS_RADIUS

        # Calculate original angular momentum from rotation
        uranus_moment_of_inertia = (
            SPHERE_MOMENT_INERTIA_COEFFICIENT * original_uranus_mass * URANUS_RADIUS**2
        )
        original_angular_velocity = 2 * torch.pi / original_rotation_period
        original_angular_momentum = uranus_moment_of_inertia * original_angular_velocity
        required_angular_momentum = original_angular_momentum * torch.sin(
            torch.deg2rad(URANUS_CURRENT_TILT)
        )

        # Calculate achieved tilt from angular momentum ratio
        angular_momentum_ratio = delivered_angular_momentum / required_angular_momentum
        achieved_tilt = torch.rad2deg(
            torch.arcsin(torch.clamp(angular_momentum_ratio, -1, 1))
        )

        return final_orbital_velocity, achieved_tilt

    def compute_loss(self):
        """Compute loss between model predictions and observations"""
        predicted_orbital_velocity, predicted_tilt = self.forward()

        # Calculate normalized squared errors
        orbital_velocity_error = (
            (predicted_orbital_velocity - URANUS_CURRENT_ORBITAL_VELOCITY)
            / URANUS_CURRENT_ORBITAL_VELOCITY
        ) ** 2
        tilt_error = (
            (predicted_tilt - URANUS_CURRENT_TILT)
            / URANUS_CURRENT_TILT
        ) ** 2

        return orbital_velocity_error + tilt_error


def optimize():
    # Create model instance
    model = UranusImpactModel()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = float("inf")
    best_state = None

    print("Optimizing all parameters including pre-impact conditions...")
    print("Format: [impactor_mass, velocity, angle, original_orbit_AU, rotation_hours]")

    for epoch in range(10000):
        optimizer.zero_grad()

        # Compute loss
        current_loss = model.compute_loss()

        # Check for NaN/Inf in loss
        if torch.isnan(current_loss) or torch.isinf(current_loss):
            print(f"\n!!! NaN/Inf detected at epoch {epoch} !!!")
            print(f"  Loss = {current_loss.item()}")
            print(f"  Parameters:")
            for name, param in model.named_parameters():
                print(f"    {name} = {param.item():.6f}")

            # Compute forward model to see where NaN originates
            with torch.no_grad():
                v_final, tilt = model()
                print(f"  v_final = {v_final.item()}, tilt = {tilt.item()}")
            break

        if current_loss.item() < best_loss:
            best_loss = current_loss.item()
            best_state = {name: param.detach().clone() for name, param in model.named_parameters()}

        # Backward pass
        current_loss.backward()

        # Check for NaN in gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan = True
                break

        if has_nan:
            print(f"\n!!! NaN/Inf in gradients at epoch {epoch} !!!")
            print(f"  Loss = {current_loss.item()}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"    {name}: value={param.item():.6f}, grad={param.grad.item():.6e}")
            break

        # Update parameters
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"\nEpoch {epoch}: Loss = {current_loss.item():.8f}")
            for name, param in model.named_parameters():
                grad_val = param.grad.item() if param.grad is not None else 0.0
                print(f"  {name}: {param.item():.6f} (grad: {grad_val:.6e})")

    print(f"\nBest loss found: {best_loss:.8f}")

    # Load best parameters
    if best_state is not None:
        for name, param in model.named_parameters():
            param.data = best_state[name]

    return model


# Run optimization
model = optimize()

# Extract and display results
with torch.no_grad():
    predicted_velocity, predicted_tilt = model()

    impactor_mass_earth_masses = model.impactor_mass_earth_units.item()
    relative_velocity_km_s = model.relative_velocity_km_per_s.item()
    impact_angle_deg = model.impact_angle_degrees.item()
    original_orbit_au = model.original_semi_major_axis_au.item()
    rotation_period_hours = model.original_rotation_period_hours.item()

    original_uranus_mass_earth_masses = (
        URANUS_CURRENT_MASS - impactor_mass_earth_masses * EARTH_MASS
    ) / EARTH_MASS

    # Convert to Python scalars for printing
    predicted_velocity_val = predicted_velocity.item()
    predicted_tilt_val = predicted_tilt.item()
    current_velocity_val = URANUS_CURRENT_ORBITAL_VELOCITY.item()
    current_tilt_val = URANUS_CURRENT_TILT.item()

print(f"\n=== OPTIMIZED SCENARIO ===")
print(f"\nPRE-IMPACT URANUS:")
print(f"  Orbital distance: {original_orbit_au:.2f} AU (current: 19.19 AU)")
print(f"  Rotation period: {rotation_period_hours:.1f} hours (current: 17.2 hours)")
print(f"  Mass: {original_uranus_mass_earth_masses:.1f} Earth masses")
print(f"  Orientation: Assumed upright (~0° tilt)")

print(f"\nIMPACTOR:")
print(f"  Mass: {impactor_mass_earth_masses:.2f} Earth masses")
print(f"  Relative velocity: {relative_velocity_km_s:.1f} km/s")
print(f"  Impact angle: {impact_angle_deg:.1f}° from orbital direction")

print(f"\nRESULT ACCURACY:")
print(f"  Final orbital velocity: {predicted_velocity_val:.1f} m/s (target: {current_velocity_val:.1f})")
print(f"  Orbit error: {abs(predicted_velocity_val - current_velocity_val) / current_velocity_val * 100:.2f}%")
print(f"  Tilt achieved: {predicted_tilt_val:.1f}° (target: {current_tilt_val:.1f}°)")
print(f"  Tilt error: {abs(predicted_tilt_val - current_tilt_val) / current_tilt_val * 100:.2f}%")

# Analysis
if (
    abs(predicted_velocity_val - current_velocity_val) / current_velocity_val < 0.01
    and abs(predicted_tilt_val - current_tilt_val) / current_tilt_val < 0.01
):
    print(f"\n✓ PERFECT MATCH! This scenario fully explains Uranus's current state.")
else:
    print(f"\n✗ Still can't perfectly match both constraints.")
    print(f"  Even with learnable pre-impact conditions, single impact theory fails.")
