"""The irreducible DivideMix estimation primitive."""

from __future__ import annotations

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DivideMix requires PyTorch; install the `train` extra.") from exc
    return torch


@block(
    id="fit_gmm",
    name="Fit Two-component GMM",
    category="Paper Specific",
    description="Fit the formal two-cluster loss mixture and publish per-sample clean probabilities.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "clean_probability"}},
    requires=("losses",), provides=("save_as",), placement=("batch", "epoch", "top"), stage="train",
    formula="w_i=P(clean|loss_i)", formula_ref="DivideMix co-divide GMM", formula_kind="special", paper="DivideMix",
)
def fit_gmm(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "clean_probability") -> None:
    torch = _torch()
    values = ctx[losses].detach().reshape(-1).float()
    if values.numel() < 2:
        raise ValueError("DivideMix GMM needs at least two loss values")
    normalized = (values - values.min()) / (values.max() - values.min()).clamp_min(torch.finfo(values.dtype).eps)
    # Deterministic two-component diagonal GMM EM.  Component zero is kept as
    # the low-loss (clean) component; no legacy estimator is imported.
    means = torch.stack((normalized.min(), normalized.max()))
    variance = normalized.var(unbiased=False).clamp_min(5e-4)
    variances = torch.full((2,), variance, device=values.device, dtype=values.dtype)
    weights = torch.full((2,), 0.5, device=values.device, dtype=values.dtype)
    for _ in range(10):
        log_prob = -0.5 * ((normalized[:, None] - means[None, :]).square() / variances[None, :] + variances.log()[None, :]) + weights.log()[None, :]
        posterior = (log_prob - log_prob.logsumexp(dim=1, keepdim=True)).exp()
        mass = posterior.sum(dim=0).clamp_min(torch.finfo(values.dtype).tiny)
        weights = mass / mass.sum()
        means = (posterior * normalized[:, None]).sum(dim=0) / mass
        variances = (posterior * (normalized[:, None] - means[None, :]).square()).sum(dim=0) / mass
        variances = variances.clamp_min(5e-4)
    clean_component = torch.argmin(means)
    ctx[save_as] = posterior[:, clean_component].to(values.device, dtype=values.dtype)
