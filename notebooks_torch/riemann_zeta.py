import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma as scipy_gamma
import math


class RiemannZeta(nn.Module):
    def __init__(self):
        super().__init__()

    def reflect_zeta(self, s):
        """
        Use the functional equation to reflect from Re(s) < 0 to Re(s) > 1
        ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
        """
        # Convert s to complex numpy for scipy.gamma
        s_np = complex(s.real.item(), s.imag.item())
        one_minus_s_np = 1.0 - s_np

        # Compute ζ(1-s) using direct summation or eta
        one_minus_s_torch = torch.tensor(1.0) - s
        if one_minus_s_torch.real > 1.0:
            zeta_1ms = self.direct_zeta(one_minus_s_torch)
        else:
            zeta_1ms = self.eta_to_zeta(one_minus_s_torch)

        # Compute the functional equation terms
        # ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
        two_s = torch.pow(torch.tensor(2.0 + 0.0j), s)
        pi_s_minus_1 = torch.pow(torch.tensor(math.pi + 0.0j), s - 1.0)
        sin_term = torch.sin(torch.tensor(math.pi) * s / 2.0)

        # Use scipy gamma for complex values
        gamma_val = scipy_gamma(one_minus_s_np)
        gamma_torch = torch.tensor(complex(gamma_val.real, gamma_val.imag), dtype=torch.complex64)

        result = two_s * pi_s_minus_1 * sin_term * gamma_torch * zeta_1ms
        return result

    def eta_to_zeta(self, s):
        """
        Use η(s) = (1 - 2^(1-s)) ζ(s) relationship
        So ζ(s) = η(s) / (1 - 2^(1-s))
        """
        eta_val = self.dirichlet_eta(s)
        denominator = 1.0 - torch.pow(2.0, 1.0 - s)

        # Avoid division by zero at s=1
        epsilon = 1e-10
        denominator = torch.where(
            torch.abs(denominator) < epsilon, torch.tensor(epsilon), denominator
        )

        return eta_val / denominator

    def dirichlet_eta(self, s, n_terms=1000):
        """
        Compute Dirichlet eta function: η(s) = Σ (-1)^(n+1) / n^s
        This converges for Re(s) > 0
        """
        n = torch.arange(1, n_terms + 1, dtype=torch.float32)
        signs = torch.pow(torch.tensor(-1.0), n + 1)
        n_complex = n.to(torch.complex64)
        terms = signs / torch.pow(n_complex, s)
        return torch.sum(terms)

    def direct_zeta(self, s, n_terms=1000):
        """
        Direct computation for Re(s) > 1
        ζ(s) = Σ 1/n^s
        """
        n = torch.arange(1, n_terms + 1, dtype=torch.float32)
        n_complex = n.to(torch.complex64)
        terms = 1.0 / torch.pow(n_complex, s)
        return torch.sum(terms)

    def forward(self, s):
        """
        Compute ζ(s) using appropriate method based on domain
        """
        real_part = s.real

        if real_part > 1.5:
            # Direct summation converges well here
            return self.direct_zeta(s)
        elif real_part > 0.0:
            # Use analytic continuation via eta function for 0 < Re(s) ≤ 1.5
            return self.eta_to_zeta(s)
        else:
            # Use functional equation for Re(s) ≤ 0
            return self.reflect_zeta(s)


class ZeroFinder(nn.Module):
    def __init__(self, init_imag=14.0):
        super().__init__()
        # We know zeros are on critical line Re(s) = 0.5
        # So we only optimize the imaginary part
        self.imag_part = nn.Parameter(torch.tensor([init_imag]))
        self.zeta = RiemannZeta()

    def forward(self):
        # Construct s = 0.5 + i*t where t is our parameter
        s = torch.complex(torch.tensor(0.5), self.imag_part[0])
        return self.zeta(s)


# Find a non-trivial zero
def find_zero(initial_guess=14.0, lr=0.1, max_iters=1000):
    model = ZeroFinder(init_imag=initial_guess)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(max_iters):
        optimizer.zero_grad()

        zeta_val = model()
        # We want |ζ(s)|^2 = 0
        loss = torch.abs(zeta_val) ** 2

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print(
                f"Iter {i}: s = 0.5 + {model.imag_part[0].item():.6f}i, "
                f"|ζ(s)| = {torch.abs(zeta_val).item():.6e}"
            )

    return model.imag_part[0].item(), losses


# Test: Find the first non-trivial zero around 14.135i
print("Finding first non-trivial zero of ζ(s)...")
print("Known value: 0.5 + 14.134725...i\n")

t_found, losses = find_zero(initial_guess=14.0, lr=0.05, max_iters=1000)
print(f"\nFound zero at: s = 0.5 + {t_found:.6f}i")

# Let's also verify ζ(-1) = -1/12
print("\n" + "=" * 50)
print("Verifying ζ(-1) = -1/12:")
zeta_model = RiemannZeta()
s_minus_one = torch.tensor(-1.0 + 0.0j, dtype=torch.complex64)
result = zeta_model(s_minus_one)
print(f"ζ(-1) = {result.real.item():.6f}")
print(f"Expected: {-1/12:.6f}")


# Try to find a zero with Re(s) ≠ 0.5 (should fail if Riemann Hypothesis is true!)
class GeneralZeroFinder(nn.Module):
    def __init__(self, init_real=0.7, init_imag=14.0):
        super().__init__()
        self.real_part = nn.Parameter(torch.tensor([init_real]))
        self.imag_part = nn.Parameter(torch.tensor([init_imag]))
        self.zeta = RiemannZeta()

    def forward(self):
        s = torch.complex(self.real_part[0], self.imag_part[0])
        return self.zeta(s)


print("\n" + "=" * 50)
print("Attempting to find a zero OFF the critical line...")
print("(This should fail to converge to zero if Riemann Hypothesis is true!)\n")

general_model = GeneralZeroFinder(init_real=0.7, init_imag=14.0)
optimizer = torch.optim.Adam(general_model.parameters(), lr=0.05)

for i in range(1000):
    optimizer.zero_grad()
    zeta_val = general_model()
    loss = torch.abs(zeta_val) ** 2
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print(
            f"Iter {i}: s = {general_model.real_part[0].item():.3f} + "
            f"{general_model.imag_part[0].item():.3f}i, "
            f"|ζ(s)| = {torch.abs(zeta_val).item():.6e}"
        )

print(
    f"\nFinal: s = {general_model.real_part[0].item():.6f} + "
    f"{general_model.imag_part[0].item():.6f}i"
)
print("Notice it either diverged or converged to Re(s) ≈ 0.5!")
