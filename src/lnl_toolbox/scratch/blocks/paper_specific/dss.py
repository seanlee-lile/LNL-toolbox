"""Explicit DSS lifecycle blocks: state, MDA, CCS, selection, and masked loss."""

from __future__ import annotations

from statistics import NormalDist
from ...context import ScratchContext
from ...registry import block


def _torch():
    import torch
    return torch


class _ScratchDSSState:
    """Small Scratch-native indexed DSS state (no legacy selector object)."""
    def __init__(self, num_samples, num_classes, total_epochs, warmup_epochs=30, alpha=0.1, prior_decay=0.99, *, strict=True, mda=True, ccs=True):
        torch = _torch()
        self.num_samples = int(num_samples); self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs); self.warmup_epochs = int(warmup_epochs)
        self.alpha = float(alpha); self.prior_decay = float(prior_decay); self.current_epoch = -1
        self.strict = bool(strict); self.mda = bool(mda); self.ccs = bool(ccs)
        self.labels = torch.full((self.num_samples,), -1, dtype=torch.long)
        self.history = torch.zeros((self.num_samples, self.total_epochs, self.num_classes), dtype=torch.float32)
        self.observed = torch.zeros((self.num_samples, self.total_epochs), dtype=torch.bool)
        self.trend_score = torch.zeros((self.num_samples, self.num_classes), dtype=torch.float32)
        self.current_prediction = torch.full((self.num_samples, self.num_classes), 1.0 / self.num_classes)
        self.marginal = torch.full((self.num_classes,), 1.0 / self.num_classes)
        self.selected = torch.ones(self.num_samples, dtype=torch.bool)
        self.excluded = torch.zeros((self.num_samples, self.num_classes), dtype=torch.bool)

    def on_cycle_start(self, epoch):
        if int(epoch) < 0 or int(epoch) >= self.total_epochs:
            raise ValueError("DSS epoch is outside total_epochs")
        if self.current_epoch not in {-1, int(epoch) - 1, int(epoch)}:
            raise ValueError("DSS epochs must be processed sequentially")
        self.current_epoch = int(epoch)

    def on_cycle_end(self, epoch):
        torch = _torch()
        if int(epoch) != self.current_epoch: raise ValueError("DSS cycle end does not match current epoch")
        known = self.labels >= 0
        if self.strict and bool(known.any()) and not bool(self.observed[known, int(epoch)].all()):
            raise ValueError("DSS did not observe every known sample this epoch")
        if int(epoch) + 1 < self.warmup_epochs:
            return
        self.selected[known] = self.current_prediction[known].argmax(1).eq(self.labels[known])
        if not self.ccs:
            self.excluded.zero_(); return
        n = int(epoch) + 1
        variance = n * (n - 1) * (2 * n + 5) / 18
        score = self.trend_score[known]
        z_score = torch.zeros_like(score) if variance == 0 else (score - score.sign()) / variance ** 0.5
        z_score[score == 0] = 0
        labels = self.labels[known]
        z_score[torch.arange(labels.numel()), labels] = float("-inf")
        self.excluded[known] = z_score > NormalDist().inv_cdf(1.0 - self.alpha)

    def observe(self, indices, labels, probabilities, epoch):
        torch = _torch(); rows = torch.as_tensor(indices).long().cpu(); targets = torch.as_tensor(labels).long().cpu(); epoch = int(epoch)
        if epoch != self.current_epoch: raise ValueError("DSS observation epoch does not match lifecycle state")
        if rows.ndim != 1 or targets.ndim != 1 or targets.shape != rows.shape or rows.numel() == 0 or torch.unique(rows).numel() != rows.numel(): raise ValueError("DSS indices and labels must be aligned, unique and non-empty")
        if int(rows.min()) < 0 or int(rows.max()) >= self.num_samples: raise IndexError("DSS sample index exceeds num_samples")
        if int(targets.min()) < 0 or int(targets.max()) >= self.num_classes: raise ValueError("DSS noisy target is outside the class range")
        known = self.labels[rows]
        if bool(((known >= 0) & (known != targets)).any()): raise ValueError("DSS noisy target changed for a stable sample index")
        values = torch.as_tensor(probabilities).detach().float().cpu()
        if values.shape != (rows.numel(), self.num_classes): raise ValueError("DSS probabilities shape mismatch")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or not bool(torch.allclose(values.sum(1), torch.ones(rows.numel()), atol=1e-5)): raise ValueError("DSS probabilities must be finite rows summing to one")
        if epoch > 0 and self.strict and not bool(self.observed[rows, :epoch].all()): raise ValueError("DSS requires one observation per prior epoch")
        if self.mda:
            self.marginal.mul_(self.prior_decay).add_(values.mean(0), alpha=1.0 - self.prior_decay)
            values = values / (self.num_classes * self.marginal).clamp_min(torch.finfo(values.dtype).tiny)
            values = values / values.sum(1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)
        if epoch:
            # CCS compares the current posterior with every prior epoch for
            # the same stable sample, matching the formal indexed history
            # contract (not merely the immediately preceding epoch).
            previous = self.history[rows, :epoch]
            self.trend_score[rows] += ((values[:, None, :] > previous).sum(1) - (values[:, None, :] < previous).sum(1)).float()
        self.current_prediction[rows] = values
        self.labels[rows] = targets
        self.history[rows, epoch] = self.current_prediction[rows]
        self.observed[rows, epoch] = True

    def masks(self, indices, labels):
        rows = _torch().as_tensor(indices).long().cpu(); excluded = self.excluded[rows].clone()
        excluded[_torch().arange(rows.numel()), _torch().as_tensor(labels).long().cpu()] = False
        return self.selected[rows].clone(), excluded

    def state_dict(self):
        """Return a detached, serialisable lifecycle snapshot for audits."""
        return {
            "config": {"num_samples": self.num_samples, "num_classes": self.num_classes,
                       "total_epochs": self.total_epochs, "warmup_epochs": self.warmup_epochs,
                       "alpha": self.alpha, "prior_decay": self.prior_decay,
                       "mda": self.mda, "ccs": self.ccs},
            "labels": self.labels.clone(), "history": self.history.clone(),
            "observed": self.observed.clone(), "trend_score": self.trend_score.clone(),
            "current_prediction": self.current_prediction.clone(), "marginal": self.marginal.clone(),
            "selected": self.selected.clone(), "excluded": self.excluded.clone(),
            "current_epoch": int(self.current_epoch),
        }

    def load_state_dict(self, state):
        config = state.get("config", {})
        expected = {"num_samples": self.num_samples, "num_classes": self.num_classes,
                    "total_epochs": self.total_epochs, "warmup_epochs": self.warmup_epochs,
                    "alpha": self.alpha, "prior_decay": self.prior_decay,
                    "mda": self.mda, "ccs": self.ccs}
        if dict(config) != expected:
            raise ValueError("DSS state configuration mismatch")
        torch = _torch()
        for name in ("labels", "history", "observed", "trend_score", "current_prediction", "marginal", "selected", "excluded"):
            value = torch.as_tensor(state[name])
            target = getattr(self, name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"DSS state shape/dtype mismatch for {name}")
            target.copy_(value)
        self.current_epoch = int(state.get("current_epoch", -1))


@block(
    id="create_dss_state",
    name="DSS: Create Indexed State",
    category="State",
    description="Create the stable-index DSS state used by the epoch lifecycle.",
    params={"prepared_data": {"type": "slot", "default": "prepared_data"}, "total_epochs": {"type": "int", "required": True, "min": 1}, "warmup_epochs": {"type": "int", "required": True, "min": 0}, "alpha": {"type": "float", "required": True, "min": 0.000001, "max": 0.999999}, "prior_decay": {"type": "float", "required": True, "min": 0.0, "max": 0.999999}, "mda": {"type": "bool", "default": True}, "ccs": {"type": "bool", "default": True}, "save_as": {"type": "slot", "default": "dss_state"}},
    requires=("prepared_data",),
    provides=("save_as",), placement=("top",), stage="setup", ui_group="② 初始化",
    formula="S={history, marginal, trend, selected, excluded}", formula_ref="DSS indexed selector lifecycle", paper="Debiased Sample Selection",
)
def create_dss_state(ctx: ScratchContext, prepared_data: str = "prepared_data", total_epochs: int | None = None, warmup_epochs: int | None = None, alpha: float | None = None, prior_decay: float | None = None, mda: bool = True, ccs: bool = True, save_as: str = "dss_state") -> None:
    prepared = ctx[prepared_data]
    train_indices = getattr(prepared, "train_indices", None)
    if train_indices is not None:
        values = _torch().as_tensor(train_indices, dtype=_torch().long).reshape(-1)
        num_samples = int(values.max().item()) + 1 if values.numel() else 0
    else:
        train_dataset = getattr(prepared, "datasets", {}).get("train")
        num_samples = len(train_dataset) if train_dataset is not None else 0
    num_classes = getattr(prepared, "num_classes", None)
    if num_classes is None:
        num_classes = ctx.get("num_classes")
    if not num_samples or num_classes is None or total_epochs is None or warmup_epochs is None or alpha is None or prior_decay is None:
        raise ValueError("DSS state requires prepared_data-derived sizes and explicit lifecycle parameters")
    fixture = bool((ctx.get("_runtime_limits") or {}).get("fixture"))
    ctx[save_as] = _ScratchDSSState(int(num_samples), int(num_classes), int(total_epochs), warmup_epochs=int(warmup_epochs), alpha=float(alpha), prior_decay=float(prior_decay), strict=not fixture, mda=bool(mda), ccs=bool(ccs))


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
    params={"state": {"type": "slot", "default": "dss_state"}, "probabilities": {"type": "slot", "default": "probabilities"}, "labels": {"type": "slot", "default": "labels"}, "indices": {"type": "slot", "default": "indices"}, "epoch": {"type": "slot", "default": "epoch"}, "save_as": {"type": "slot", "default": "dss_adjusted_probabilities"}},
    requires=("state", "probabilities"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="p'_c=p_c/(C m_c); normalize rows", formula_ref="DSS MDA", paper="Debiased Sample Selection",
)
def dss_mda_marginal_adjustment(ctx: ScratchContext, state: str = "dss_state", probabilities: str = "probabilities", labels: str = "labels", indices: str = "indices", epoch: str = "epoch", save_as: str = "dss_adjusted_probabilities") -> None:
    torch = _torch()
    values = ctx[probabilities].detach()
    selector = ctx[state]
    if labels in ctx and indices in ctx and epoch in ctx:
        selector.observe(ctx[indices], ctx[labels], values, int(ctx[epoch]))
        adjusted = selector.current_prediction[torch.as_tensor(ctx[indices]).long().cpu()].to(values.device)
        ctx[save_as] = adjusted
        return
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
    import torch
    selected, excluded = ctx[state].masks(ctx[indices], ctx[labels])
    logits_value = ctx[logits]
    excluded = excluded.to(logits_value.device)
    targets = ctx[labels].long()
    if bool(excluded.gather(1, targets[:, None]).any()) or bool(excluded.all(1).any()):
        raise ValueError("DSS candidate mask must retain the observed target and one class")
    masked_logits = logits_value.masked_fill(excluded, float("-inf"))
    per_sample = torch.logsumexp(masked_logits, dim=1) - masked_logits.gather(1, targets[:, None]).squeeze(1)
    ctx[save_as] = (per_sample * selected.to(per_sample.device, per_sample.dtype)).mean()
