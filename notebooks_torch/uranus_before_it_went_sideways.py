#!/usr/bin/env python3
import math
import torch
import torch.nn as nn

# Physical constants
GRAVITATIONAL_CONSTANT = 6.6743e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
ASTRONOMICAL_UNIT = 1.495978707e11  # m (exact as of IAU 2012)
EARTH_MASS = 5.9722e24  # kg (NASA Earth Fact Sheet)
URANUS_RADIUS = 2.5559e7  # m (equatorial radius at 1 bar)
SPHERE_MOMENT_INERTIA_COEFFICIENT = 2 / 5

# Current observations
SUN_MASS_CURRENT = 1.9884e30  # kg (current nominal solar mass)
URANUS_CURRENT_MASS = 8.68e25  # kg (NASA Uranus Fact Sheet)
URANUS_CURRENT_SEMI_MAJOR_AXIS = 19.19126393 * ASTRONOMICAL_UNIT  # m (NASA)
URANUS_CURRENT_ORBITAL_VELOCITY = torch.sqrt(
    torch.tensor(
        GRAVITATIONAL_CONSTANT * SUN_MASS_CURRENT / URANUS_CURRENT_SEMI_MAJOR_AXIS,
        dtype=torch.float64,
    )
)  # m/s

# Uranus's axial tilt: 98° obliquity (measured from north orbital pole)
# This means Uranus rotates RETROGRADE (backwards) with axis 8° past horizontal
# The tilt > 90° indicates the planet's rotation was reversed by the impact
# Store in radians for internal calculations
URANUS_CURRENT_TILT = torch.tensor(
    98.0 * torch.pi / 180.0, dtype=torch.float64
)  # radians (98°)


class UranusImpactModel(nn.Module):
    """Model for Uranus impact scenario with learnable parameters"""

    def __init__(self):
        super().__init__()

        # Learnable parameters - initialize with reasonable guesses
        self.impactor_mass_earth_units = nn.Parameter(
            torch.tensor(2.0, dtype=torch.float64)
        )
        self.relative_velocity_km_per_s = nn.Parameter(
            torch.tensor(15.0, dtype=torch.float64)
        )  # Typical impact velocity
        self.impact_angle_degrees = nn.Parameter(
            torch.tensor(45.0, dtype=torch.float64)
        )
        self.original_semi_major_axis_au = nn.Parameter(
            torch.tensor(19.0, dtype=torch.float64)
        )
        self.original_rotation_period_hours = nn.Parameter(
            torch.tensor(15.0, dtype=torch.float64)
        )
        # Sun's mass at time of impact (in units of current solar mass)
        # Initialize to 1.0 (same as now), let optimizer find if it was different
        self.original_sun_mass_solar_units = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64)
        )

    def forward(self):
        """Forward pass computing final orbital velocity and tilt (in radians)"""
        # Convert learnable parameters to SI units
        impactor_mass = self.impactor_mass_earth_units * EARTH_MASS  # kg
        original_uranus_mass = URANUS_CURRENT_MASS - impactor_mass  # kg
        relative_velocity = self.relative_velocity_km_per_s * 1000  # m/s
        impact_angle_radians = (
            self.impact_angle_degrees * torch.pi / 180.0
        )  # convert to radians
        original_semi_major_axis = (
            self.original_semi_major_axis_au * ASTRONOMICAL_UNIT
        )  # m
        original_rotation_period = self.original_rotation_period_hours * 3600  # seconds
        original_sun_mass = self.original_sun_mass_solar_units * SUN_MASS_CURRENT  # kg

        # Calculate original orbital velocity before impact (using original sun mass)
        original_orbital_velocity = torch.sqrt(
            GRAVITATIONAL_CONSTANT * original_sun_mass / original_semi_major_axis
        )

        # Decompose impact velocity into tangential and radial components
        tangential_velocity = relative_velocity * torch.cos(impact_angle_radians)
        radial_velocity = relative_velocity * torch.sin(impact_angle_radians)

        # Calculate orbital velocity change from impact
        velocity_change = (impactor_mass * tangential_velocity) / (
            original_uranus_mass + impactor_mass
        )
        final_orbital_velocity = original_orbital_velocity + velocity_change

        # Calculate angular momentum from impact
        # Radial component: adds perpendicular to rotation axis (tilts the planet)
        perpendicular_angular_momentum = impactor_mass * radial_velocity * URANUS_RADIUS

        # Tangential component: affects rotation speed (can slow/reverse rotation)
        # The tangential impact at radius R changes the spin angular momentum
        parallel_angular_momentum_change = (
            impactor_mass * tangential_velocity * URANUS_RADIUS
        )

        # Calculate original angular momentum from rotation
        uranus_moment_of_inertia = (
            SPHERE_MOMENT_INERTIA_COEFFICIENT * original_uranus_mass * URANUS_RADIUS**2
        )
        original_angular_velocity = 2 * torch.pi / original_rotation_period
        original_angular_momentum = uranus_moment_of_inertia * original_angular_velocity

        # The tangential impact can slow or reverse the rotation
        # Depending on impact direction, it can add or subtract from spin
        # For retrograde tilt (>90°), we need it to oppose the original rotation
        final_parallel_angular_momentum = (
            original_angular_momentum - parallel_angular_momentum_change
        )

        # Vector addition of angular momenta
        # The tilt angle is: θ = atan2(perpendicular component, parallel component)
        # This naturally handles:
        # - Small tilts (0 to π/2 rad) when parallel component stays positive (prograde)
        # - Large tilts (π/2 to π rad) when parallel component becomes negative (retrograde)
        #
        # For Uranus at 98° (1.71 rad):
        # - perpendicular_L is large (tips it sideways)
        # - final_parallel_L is slightly negative (rotation reversed)
        # - atan2(large, small_negative) ≈ 1.71 rad
        achieved_tilt_radians = torch.atan2(
            perpendicular_angular_momentum, final_parallel_angular_momentum
        )

        return final_orbital_velocity, achieved_tilt_radians

    def compute_loss(self):
        """Compute loss between model predictions and observations"""
        predicted_orbital_velocity, predicted_tilt_radians = self.forward()

        # Calculate normalized squared errors (tilt in radians)
        orbital_velocity_error = (
            (predicted_orbital_velocity - URANUS_CURRENT_ORBITAL_VELOCITY)
            / URANUS_CURRENT_ORBITAL_VELOCITY
        ) ** 2
        tilt_error = (
            (predicted_tilt_radians - URANUS_CURRENT_TILT) / URANUS_CURRENT_TILT
        ) ** 2

        # Add STRONG penalties for non-physical parameters
        # Sun mass should be 0.99-1.01 of current (very small change over billions of years)
        sun_mass_penalty = 1000.0 * ((self.original_sun_mass_solar_units - 1.0) ** 2)

        # Impactor mass should be < 5 Earth masses (realistic for giant impacts)
        impactor_penalty = (
            100.0 * torch.clamp(self.impactor_mass_earth_units - 5.0, min=0) ** 2
        )

        # Rotation period: gas giants typically 10-20 hours
        rotation_penalty = 50.0 * (
            torch.clamp(10.0 - self.original_rotation_period_hours, min=0) ** 2
            + torch.clamp(self.original_rotation_period_hours - 20.0, min=0) ** 2
        )

        # Orbital migration: should be within ±2 AU of current position
        orbit_penalty = (
            50.0
            * torch.clamp(
                torch.abs(self.original_semi_major_axis_au - 19.19) - 2.0, min=0
            )
            ** 2
        )

        # Velocity must be realistic (10-30 km/s)
        # Below 10 km/s is non-physical (below escape velocity from Sun at Uranus)
        velocity_penalty = 100.0 * (
            torch.clamp(10.0 - self.relative_velocity_km_per_s, min=0) ** 2
            + torch.clamp(self.relative_velocity_km_per_s - 30.0, min=0) ** 2
        )

        return (
            orbital_velocity_error
            + tilt_error
            + sun_mass_penalty
            + impactor_penalty
            + rotation_penalty
            + orbit_penalty
            + velocity_penalty
        )


def optimize():
    # Create model instance
    model = UranusImpactModel()

    # Use Adam with lower learning rate for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    best_loss = float("inf")
    best_state = None

    print("Optimizing all parameters including pre-impact conditions...")
    print("Format: [impactor_mass, velocity, angle, original_orbit_AU, rotation_hours]")

    for epoch in range(20000):
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
            break

        if current_loss.item() < best_loss:
            best_loss = current_loss.item()
            best_state = {
                name: param.detach().clone() for name, param in model.named_parameters()
            }

        # Backward pass
        current_loss.backward()

        # Update parameters
        optimizer.step()

        if epoch % 2000 == 0:
            print(f"\nEpoch {epoch}: Loss = {current_loss.item():.8f}")
            for name, param in model.named_parameters():
                print(f"  {name}: {param.item():.6f}")

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
    predicted_velocity, predicted_tilt_radians = model()

    impactor_mass_earth_masses = model.impactor_mass_earth_units.item()
    relative_velocity_km_s = model.relative_velocity_km_per_s.item()
    impact_angle_deg = model.impact_angle_degrees.item()
    original_orbit_au = model.original_semi_major_axis_au.item()
    rotation_period_hours = model.original_rotation_period_hours.item()
    original_sun_mass_ratio = model.original_sun_mass_solar_units.item()

    original_uranus_mass_earth_masses = (
        URANUS_CURRENT_MASS - impactor_mass_earth_masses * EARTH_MASS
    ) / EARTH_MASS

    # Convert to Python scalars for printing
    predicted_velocity_val = predicted_velocity.item()
    predicted_tilt_radians_val = predicted_tilt_radians.item()
    current_velocity_val = URANUS_CURRENT_ORBITAL_VELOCITY.item()
    current_tilt_radians_val = URANUS_CURRENT_TILT.item()

    # Convert tilts to degrees for display
    predicted_tilt_degrees = predicted_tilt_radians_val * 180.0 / math.pi
    current_tilt_degrees = current_tilt_radians_val * 180.0 / math.pi

print(f"\n=== OPTIMIZED SCENARIO ===")
print(f"\nPRE-IMPACT CONDITIONS:")
print(f"  Sun's mass: {original_sun_mass_ratio:.6f} × current solar mass")
print(f"    (Sun was {(original_sun_mass_ratio - 1) * 100:+.4f}% different)")
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
print(
    f"  Final orbital velocity: {predicted_velocity_val:.1f} m/s (target: {current_velocity_val:.1f})"
)
print(
    f"  Orbit error: {abs(predicted_velocity_val - current_velocity_val) / current_velocity_val * 100:.2f}%"
)
print(
    f"  Tilt achieved: {predicted_tilt_degrees:.1f}° (target: {current_tilt_degrees:.1f}°)"
)
print(
    f"  Tilt error: {abs(predicted_tilt_radians_val - current_tilt_radians_val) / current_tilt_radians_val * 100:.2f}%"
)

# Analysis (use radians for comparison)
if (
    abs(predicted_velocity_val - current_velocity_val) / current_velocity_val < 0.01
    and abs(predicted_tilt_radians_val - current_tilt_radians_val)
    / current_tilt_radians_val
    < 0.01
):
    print(f"\n✓ PERFECT MATCH! This scenario fully explains Uranus's current state.")
else:
    print(f"\n✗ Still can't perfectly match both constraints.")
    print(f"  Even with learnable pre-impact conditions, single impact theory fails.")
