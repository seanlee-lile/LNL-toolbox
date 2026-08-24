"""Explicit DSS lifecycle blocks: state, MDA, CCS, selection, and masked loss."""

from __future__ import annotations

from ...context import ScratchContext
from ...registry import block


def _torch():
    import torch
    return torch


@block(
    id="create_dss_state",
    name="DSS: Create Indexed State",
    category="State",
    description="Create the stable-index DSS state used by the epoch lifecycle.",
    params={"num_samples": {"type": "int", "default": 50000, "min": 1}, "num_classes": {"type": "int", "default": 10, "min": 2}, "total_epochs": {"type": "int", "default": 150, "min": 1}, "warmup_epochs": {"type": "int", "default": 30, "min": 0}, "alpha": {"type": "float", "default": 0.1, "min": 0.000001, "max": 0.999999}, "prior_decay": {"type": "float", "default": 0.99, "min": 0.0, "max": 0.999999}, "save_as": {"type": "slot", "default": "dss_state"}},
    provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
    formula="S={history, marginal, trend, selected, excluded}", formula_ref="DSS indexed selector lifecycle", paper="Debiased Sample Selection",
)
def create_dss_state(ctx: ScratchContext, num_samples: int = 50000, num_classes: int = 10, total_epochs: int = 150, warmup_epochs: int = 30, alpha: float = 0.1, prior_decay: float = 0.99, save_as: str = "dss_state") -> None:
    from lnl_toolbox.selectors.dss import DSSSelectorState
    ctx[save_as] = DSSSelectorState(int(num_samples), int(num_classes), int(total_epochs), warmup_epochs=int(warmup_epochs), alpha=float(alpha), prior_decay=float(prior_decay), mda=True, ccs=True)


@block(
    id="dss_warmup_lifecycle",
    name="DSS: Warmup / Cycle Start",
    category="Paper Specific",
    description="Advance DSS to the current epoch before observing indexed predictions.",
    params={"state": {"type": "slot", "default": "dss_state"}, "epoch": {"type": "slot", "default": "epoch"}},
    requires=("state", "epoch"), provides=(), placement=("epoch",), stage="train", ui_group="⑩ 论文专用",
    formula="S.on_cycle_start(t)", formula_ref="DSS warmup lifecycle", paper="Debiased Sample Selection",
)
def dss_warmup_lifecycle(ctx: ScratchContext, state: str = "dss_state", epoch: str = "epoch") -> None:
    ctx[state].on_cycle_start(int(ctx[epoch]))


@block(
    id="dss_mda_marginal_adjustment",
    name="DSS: MDA Marginal Adjustment",
    category="Paper Specific",
    description="Apply DSS moving marginal-distribution adjustment to detached class probabilities.",
    params={"state": {"type": "slot", "default": "dss_state"}, "probabilities": {"type": "slot", "default": "probabilities"}, "save_as": {"type": "slot", "default": "dss_adjusted_probabilities"}},
    requires=("state", "probabilities"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="p'_c=p_c/(C m_c); normalize rows", formula_ref="DSS MDA", paper="Debiased Sample Selection",
)
def dss_mda_marginal_adjustment(ctx: ScratchContext, state: str = "dss_state", probabilities: str = "probabilities", save_as: str = "dss_adjusted_probabilities") -> None:
    torch = _torch()
    values = ctx[probabilities].detach()
    selector = ctx[state]
    selector.marginal.mul_(float(selector.prior_decay)).add_(values.mean(0).cpu(), alpha=1.0 - float(selector.prior_decay))
    adjusted = values / (int(selector.num_classes) * selector.marginal.to(values)).clamp_min(torch.finfo(values.dtype).tiny)
    ctx[save_as] = adjusted / adjusted.sum(1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)


@block(
    id="dss_ccs_trend_exclusion",
    name="DSS: CCS Trend Exclusion",
    category="Paper Specific",
    description="Finalize DSS trend-based class exclusions and sample selection at epoch end.",
    params={"state": {"type": "slot", "default": "dss_state"}, "epoch": {"type": "slot", "default": "epoch"}},
    requires=("state", "epoch"), provides=(), placement=("epoch",), stage="train", ui_group="⑩ 论文专用",
    formula="CCS: z_c>Phi^{-1}(1-alpha) => excluded_c", formula_ref="DSS CCS trend exclusion", paper="Debiased Sample Selection",
)
def dss_ccs_trend_exclusion(ctx: ScratchContext, state: str = "dss_state", epoch: str = "epoch") -> None:
    ctx[state].on_cycle_end(int(ctx[epoch]))


@block(
    id="dss_masked_training_loss",
    name="DSS: Masked Training Loss",
    category="Loss",
    description="Compute DSS candidate-masked cross entropy using the indexed selected/excluded state.",
    params={"state": {"type": "slot", "default": "dss_state"}, "logits": {"type": "slot", "default": "logits"}, "labels": {"type": "slot", "default": "labels"}, "indices": {"type": "slot", "default": "indices"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("state", "logits", "labels", "indices"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="L=mean_i selected_i * CE(z_i,y_i) with excluded classes masked", formula_ref="DSS masked risk", paper="Debiased Sample Selection",
)
def dss_masked_training_loss(ctx: ScratchContext, state: str = "dss_state", logits: str = "logits", labels: str = "labels", indices: str = "indices", save_as: str = "loss") -> None:
    import torch.nn.functional as F
    selected, excluded = ctx[state].masks(ctx[indices], ctx[labels])
    per_sample = -F.log_softmax(ctx[logits], 1).gather(1, ctx[labels].long()[:, None]).squeeze(1)
    ctx[save_as] = (per_sample * selected.to(per_sample.device, per_sample.dtype)).mean()
