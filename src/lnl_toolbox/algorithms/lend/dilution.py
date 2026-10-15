from __future__ import annotations

"""LEND Eq. (3), with explicit finite-step or convergence stopping."""

import math
from numbers import Integral
import torch
from torch import Tensor


def dilute_labels(noisy_onehot: Tensor, graph: Tensor, *, alpha: float,
                  steps: int = 10, policy: str = "fixed_steps",
                  tolerance: float = 1e-5, max_steps: int = 1000,
                  diagnostics: dict[str, float] | None = None) -> Tensor:
    if not isinstance(noisy_onehot, Tensor) or noisy_onehot.ndim != 2:
        raise ValueError("LEND noisy one-hot labels must have shape [B,C]")
    if not torch.is_floating_point(noisy_onehot) or noisy_onehot.requires_grad:
        raise ValueError("LEND noisy one-hot labels must be detached floating values")
    if not isinstance(graph, Tensor) or graph.shape != (noisy_onehot.shape[0], noisy_onehot.shape[0]):
        raise ValueError("LEND graph and noisy labels must align")
    if graph.device != noisy_onehot.device or graph.dtype != noisy_onehot.dtype:
        raise ValueError("LEND graph and noisy labels must share dtype and device")
    if not bool(torch.isfinite(noisy_onehot).all()) or bool((noisy_onehot < 0).any()):
        raise ValueError("LEND noisy one-hot labels must be finite and non-negative")
    if graph.requires_grad or not bool(torch.isfinite(graph).all()) or bool((graph < 0).any()):
        raise ValueError("LEND graph must be detached, finite and non-negative")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("LEND dilution requires alpha in (0,1)")
    if policy not in {"fixed_steps", "until_converged"}:
        raise ValueError("LEND dilution policy must be fixed_steps or until_converged")
    limit = steps if policy == "fixed_steps" else max_steps
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("LEND dilution iteration limit must be a positive integer")
    if policy == "until_converged" and (not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("LEND dilution tolerance must be finite and positive")
    current = noisy_onehot.detach()
    for iteration in range(1, int(limit) + 1):
        updated = alpha * (graph @ current) + (1.0 - alpha) * current
        if not bool(torch.isfinite(updated).all()):
            raise ValueError("LEND dilution produced non-finite values")
        residual = float((updated - current).abs().max().item())
        current = updated
        converged = policy == "until_converged" and residual <= tolerance
        if diagnostics is not None:
            diagnostics.update(iterations=float(iteration), residual=residual, converged=float(converged))
        if converged:
            break
    else:
        if policy == "until_converged":
            raise RuntimeError(f"LEND dilution did not converge after {limit} iterations "
                               f"(residual={residual:.6g}, tolerance={tolerance:.6g})")
    return current.detach()


__all__ = ["dilute_labels"]
