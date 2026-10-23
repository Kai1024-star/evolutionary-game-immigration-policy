"""Three-population immigration-policy games reconstructed from thesis scripts."""

from .models import parameters, payoffs, rhs, jacobian, simulate, reference_solution

__all__ = ["parameters", "payoffs", "rhs", "jacobian", "simulate", "reference_solution"]
