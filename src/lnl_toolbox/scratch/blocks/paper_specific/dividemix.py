"""Large-grain DivideMix lifecycle blocks for a recipe interpreter."""

from __future__ import annotations

from typing import Any

from ...context import ScratchContext
from ...registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("DivideMix blocks require PyTorch; install the `train` extra.") from exc
    return torch, F


@block(
    id="warmup",
    name="DivideMix Warmup",
    category="Paper Specific",
    description="Mark a warm-up lifecycle stage; child steps perform ordinary training.",
    kind="loop",
    params={"epochs": {"type": "int", "required": True, "min": 0}},
    provides=("epoch", "warmup_epoch"),
)
def warmup(ctx: ScratchContext, *, params: dict[str, Any], children, execute) -> None:
    for epoch in range(int(params["epochs"])):
        ctx["epoch"] = epoch
        ctx["warmup_epoch"] = epoch
        execute(children, ctx)


@block(
    id="fit_gmm",
    name="DivideMix: Fit Two-component GMM",
    category="Paper Specific",
    description="Fit a deterministic two-cluster loss mixture and save clean probabilities.",
    params={"losses": {"type": "slot", "default": "loss_per_sample"}, "save_as": {"type": "slot", "default": "clean_probability"}},
    requires=("losses",),
    provides=("save_as",),
)
def fit_gmm(ctx: ScratchContext, losses: str = "loss_per_sample", save_as: str = "clean_probability") -> None:
    torch, _ = _torch()
    values = ctx[losses].detach().reshape(-1).float()
    if values.numel() < 2:
        raise ValueError("DivideMix GMM needs at least two loss values")
    from lnl_toolbox.estimators import DivideMixGMMCleanProbabilityEstimator, DivideMixGMMLossInput
    normalized = (values - values.min()) / (values.max() - values.min()).clamp_min(torch.finfo(values.dtype).eps)
    try:
        result = DivideMixGMMCleanProbabilityEstimator(random_seed=0, max_iter=10, tolerance=1e-2, covariance_regularization=5e-4, minimum_mean_separation=1e-6).estimate(DivideMixGMMLossInput(normalized, torch.arange(values.numel(), device=values.device)))
        ctx[save_as] = result.scores.to(values.device, dtype=values.dtype)
    except (ValueError, RuntimeError):
        ctx[save_as] = 1.0 - normalized


@block(
    id="split_clean_noisy",
    name="DivideMix: Split Clean/Noisy",
    category="Paper Specific",
    description="Turn clean probabilities into a clean mask and index list.",
    params={"probability": {"type": "slot", "default": "clean_probability"}, "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}, "indices_as": {"type": "slot", "default": "clean_indices"}},
    requires=("probability",),
    provides=("indices_as", "clean_mask"),
)
def split_clean_noisy(ctx: ScratchContext, probability: str = "clean_probability", threshold: float = 0.5, indices_as: str = "clean_indices") -> None:
    torch, _ = _torch()
    mask = ctx[probability].reshape(-1) >= float(threshold)
    ctx["clean_mask"] = mask
    ctx[indices_as] = torch.where(mask)[0]


@block(
    id="co_refine",
    name="DivideMix: Co-refine Labels",
    category="Paper Specific",
    description="Blend noisy one-hot labels with model probabilities using clean probabilities.",
    params={"probability": {"type": "slot", "default": "clean_probability"}, "labels": {"type": "slot", "default": "labels"}, "probs": {"type": "slot", "default": "probabilities"}, "save_as": {"type": "slot", "default": "refined_labels"}},
    requires=("probability", "labels", "probs"),
    provides=("save_as",),
)
def co_refine(ctx: ScratchContext, probability: str = "clean_probability", labels: str = "labels", probs: str = "probabilities", save_as: str = "refined_labels") -> None:
    torch, F = _torch()
    target = F.one_hot(ctx[labels].long(), num_classes=ctx[probs].shape[-1]).float()
    weight = ctx[probability].reshape(-1, 1).clamp(0.0, 1.0)
    ctx[save_as] = weight * target + (1.0 - weight) * ctx[probs].detach()


@block(id="create_dividemix_loss_history", name="DivideMix: Create Loss History", category="State", description="Create stable-index peer loss histories used by the epoch-level co-divide GMM.", params={"prepared_data":{"type":"slot","default":"prepared_data"},"save_as_a":{"type":"slot","default":"loss_history_a"},"save_as_b":{"type":"slot","default":"loss_history_b"}}, requires=("prepared_data",), provides=("save_as_a","save_as_b"), placement=("top",), stage="setup", ui_group="② 初始化", formula="H_i^A=H_i^B=[]", formula_ref="DivideMix per-example loss history", paper="DivideMix")
def create_dividemix_loss_history(ctx: ScratchContext, prepared_data: str="prepared_data", save_as_a: str="loss_history_a", save_as_b: str="loss_history_b") -> None:
    torch,_=_torch(); indices=torch.as_tensor(ctx[prepared_data].train_indices,dtype=torch.long); size=int(indices.max().item())+1
    ctx[save_as_a]={"loss":torch.zeros(size),"seen":torch.zeros(size,dtype=torch.bool)}; ctx[save_as_b]={"loss":torch.zeros(size),"seen":torch.zeros(size,dtype=torch.bool)}


@block(id="dividemix_update_loss_history", name="DivideMix: Update Peer Loss History", category="State", description="Write detached per-example losses into a stable-index peer history.", params={"history":{"type":"slot","default":"loss_history_a"},"indices":{"type":"slot","default":"indices"},"losses":{"type":"slot","default":"loss_per_sample"},"save_as":{"type":"slot","default":"history_losses"}}, requires=("history","indices","losses"), provides=("save_as",), placement=("batch",), stage="train", ui_group="④ 状态更新", formula="H[index_i]<-ell_i", formula_ref="DivideMix loss modeling", paper="DivideMix")
def dividemix_update_loss_history(ctx: ScratchContext, history: str="loss_history_a", indices: str="indices", losses: str="loss_per_sample", save_as: str="history_losses") -> None:
    rows=ctx[indices].detach().long().cpu(); state=ctx[history]; state["loss"][rows]=ctx[losses].detach().float().cpu(); state["seen"][rows]=True; ctx[save_as]=state["loss"][state["seen"]].to(ctx[losses].device)


@block(id="dividemix_sharpen_targets", name="DivideMix: Sharpen Guessed Targets", category="Paper Specific", description="Sharpen normalized guessed or refined class probabilities with the MixMatch temperature.", params={"targets":{"type":"slot","default":"refined_labels"},"temperature":{"type":"float","default":0.5,"min":0.0001},"save_as":{"type":"slot","default":"sharpened_labels"}}, requires=("targets",), provides=("save_as",), placement=("batch",), formula="q_c<-q_c^(1/T)/sum_j q_j^(1/T)", formula_ref="DivideMix MixMatch sharpening", paper="DivideMix")
def dividemix_sharpen_targets(ctx: ScratchContext, targets: str="refined_labels", temperature: float=0.5, save_as: str="sharpened_labels") -> None:
    values=ctx[targets].clamp_min(0).pow(1.0/float(temperature)); ctx[save_as]=values/values.sum(1,keepdim=True).clamp_min(1e-12)


@block(id="dividemix_peer_clean_probability", name="DivideMix: Publish Peer Clean Probability", category="Sample Selection", description="Use one peer's clean probability to partition data for training the other peer.", params={"peer_probability":{"type":"slot","default":"clean_probability_b"},"save_as":{"type":"slot","default":"clean_probability"}}, requires=("peer_probability",), provides=("save_as",), placement=("batch",), formula="w^A<-GMM(H^B), w^B<-GMM(H^A)", formula_ref="DivideMix co-divide", paper="DivideMix")
def dividemix_peer_clean_probability(ctx: ScratchContext, peer_probability: str="clean_probability_b", save_as: str="clean_probability") -> None:
    ctx[save_as]=ctx[peer_probability].detach()


@block(
    id="dividemix_supervised_loss",
    name="DivideMix: Supervised MixMatch Loss",
    category="Loss",
    description="Compute the labeled cross-entropy term on refined labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "targets": {"type": "slot", "default": "refined_labels"}, "mask": {"type": "slot", "default": "clean_mask"}, "save_as": {"type": "slot", "default": "dividemix_loss_x"}},
    requires=("logits", "targets"), provides=("save_as",), placement=("batch",),
    formula="L_x=-mean_{i in C} sum_c q_ic log softmax(z_i)_c", formula_ref="DivideMix MixMatch supervised term", paper="DivideMix",
)
def dividemix_supervised_loss(ctx: ScratchContext, logits: str = "logits", targets: str = "refined_labels", mask: str = "clean_mask", save_as: str = "dividemix_loss_x") -> None:
    torch, F = _torch()
    values, soft_targets = ctx[logits], ctx[targets]
    selected = ctx.get(mask)
    if selected is None or not bool(selected.any()):
        selected = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    ctx[save_as] = -(soft_targets[selected] * F.log_softmax(values[selected], dim=1)).sum(1).mean()


@block(
    id="dividemix_unsupervised_loss",
    name="DivideMix: Unsupervised MixMatch Loss",
    category="Loss",
    description="Compute the unlabeled consistency MSE term on refined labels.",
    params={"logits": {"type": "slot", "default": "logits"}, "targets": {"type": "slot", "default": "refined_labels"}, "mask": {"type": "slot", "default": "clean_mask"}, "save_as": {"type": "slot", "default": "dividemix_loss_u"}},
    requires=("logits", "targets"), provides=("save_as",), placement=("batch",),
    formula="L_u=mean ||softmax(z_U)-q_U||^2", formula_ref="DivideMix MixMatch unsupervised term", paper="DivideMix",
)
def dividemix_unsupervised_loss(ctx: ScratchContext, logits: str = "logits", targets: str = "refined_labels", mask: str = "clean_mask", save_as: str = "dividemix_loss_u") -> None:
    torch, F = _torch()
    values, soft_targets = ctx[logits], ctx[targets]
    selected = ctx.get(mask)
    if selected is None or not bool((~selected).any()):
        selected = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
    if bool(selected.any()):
        selected = ~selected
    else:
        selected = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    ctx[save_as] = (F.softmax(values[selected], dim=1) - soft_targets[selected]).square().mean()


@block(
    id="dividemix_prior_regularizer",
    name="DivideMix: Prior Regularizer",
    category="Loss",
    description="Match the mean prediction to the uniform class prior.",
    params={"logits": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "dividemix_loss_r"}},
    requires=("logits",), provides=("save_as",), placement=("batch",),
    formula="L_r=KL(U || mean_i softmax(z_i))", formula_ref="DivideMix prior regularization term", paper="DivideMix",
)
def dividemix_prior_regularizer(ctx: ScratchContext, logits: str = "logits", save_as: str = "dividemix_loss_r") -> None:
    torch, F = _torch()
    mean_probability = F.softmax(ctx[logits], dim=1).mean(0)
    prior = torch.full_like(mean_probability, 1.0 / mean_probability.numel())
    ctx[save_as] = (prior * (prior / mean_probability.clamp_min(torch.finfo(mean_probability.dtype).tiny)).log()).sum()


@block(
    id="dividemix_objective_composition",
    name="DivideMix: Compose MixMatch Objective",
    category="Loss",
    description="Compose supervised, unsupervised, and prior terms with the formal coefficients.",
    params={"supervised": {"type": "slot", "default": "dividemix_loss_x"}, "unsupervised": {"type": "slot", "default": "dividemix_loss_u"}, "regularizer": {"type": "slot", "default": "dividemix_loss_r"}, "lambda_u": {"type": "float", "default": 25.0, "min": 0.0}, "lambda_r": {"type": "float", "default": 1.0, "min": 0.0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("supervised", "unsupervised", "regularizer"), provides=("save_as",), placement=("batch",),
    formula="L=L_x+lambda_u L_u+lambda_r L_r", formula_ref="DivideMix MixMatch composition", paper="DivideMix",
)
def dividemix_objective_composition(ctx: ScratchContext, supervised: str = "dividemix_loss_x", unsupervised: str = "dividemix_loss_u", regularizer: str = "dividemix_loss_r", lambda_u: float = 25.0, lambda_r: float = 1.0, save_as: str = "loss") -> None:
    torch, _ = _torch()
    objective = ctx[supervised] + float(lambda_u) * ctx[unsupervised] + float(lambda_r) * ctx[regularizer]
    if not bool(torch.isfinite(objective).item()):
        raise FloatingPointError("DivideMix objective is not finite")
    ctx[save_as] = objective
