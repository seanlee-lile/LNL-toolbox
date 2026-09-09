"""Explicit directional-label-diffusion lifecycle blocks."""
from __future__ import annotations
from ...context import ScratchContext
from ...registry import block

def _torch():
    import torch
    return torch


def _schedule(timesteps, *, device, dtype=None):
    torch = _torch(); dtype = dtype or torch.float32
    alpha = torch.full((int(timesteps),), 1.0 / int(timesteps), device=device, dtype=dtype)
    beta = torch.full_like(alpha, 1.0 / (int(timesteps) ** 0.5))
    return alpha.cumsum(0), beta.square().cumsum(0).sqrt()


def _timestep_embedding(timestep, dimension):
    torch = _torch(); half = max(int(dimension) // 2, 1)
    freq = torch.exp(-torch.log(torch.tensor(10000.0, device=timestep.device)) * torch.arange(half, device=timestep.device).float() / max(half - 1, 1))
    value = timestep.float()[:, None] * freq[None, :]
    result = torch.cat((value.sin(), value.cos()), dim=1)
    return result[:, :int(dimension)]


@block(
    id="dld_pre_correct_labels",
    name="DLD: Two-view Label Pre-correction",
    category="Paper Specific",
    description=(
        "Build DLD's y0 and yn from detached weak/strong view posteriors "
        "before the directional diffusion step."
    ),
    params={
        "weak_probabilities": {"type": "slot", "default": "weak_probabilities"},
        "strong_probabilities": {"type": "slot", "default": "strong_probabilities"},
        "noisy_labels": {"type": "slot", "default": "labels"},
        "gmm_iterations": {"type": "int", "default": 10, "min": 1},
        "y0_as": {"type": "slot", "default": "y0"},
        "yn_as": {"type": "slot", "default": "yn"},
        "direction_as": {"type": "slot", "default": "direction"},
        "partition_as": {"type": "slot", "default": "dld_partition"},
        "divergence_as": {"type": "slot", "default": "dld_divergence"},
    },
    requires=("weak_probabilities", "strong_probabilities", "noisy_labels"),
    provides=("y0_as", "yn_as", "direction_as", "partition_as", "divergence_as"),
    placement=("batch",), stage="train", ui_group="⑩ 论文专用",
    formula="y_d=y_n-y_0; (y_0,y_n)=PreCorrect(p_w,p_s,\\tilde y)", formula_kind="special",
    formula_ref="DLD two-view label pre-correction before diffusion",
    paper="Directional Label Diffusion",
)
def dld_pre_correct_labels(
    ctx: ScratchContext,
    weak_probabilities="weak_probabilities",
    strong_probabilities="strong_probabilities",
    noisy_labels="labels",
    gmm_iterations=10,
    y0_as="y0",
    yn_as="yn",
    direction_as="direction",
    partition_as="dld_partition",
    divergence_as="dld_divergence",
):
    """Materialize the two-view pre-correction states used by DLD.

    The operation is deliberately detached from the classifier graph: the
    pre-correction is an epoch/batch target-construction stage, not another
    gradient path through the current classifier.  A small two-component EM
    fit is used on the per-example view divergence, with the lower-loss
    component treated as the reliable component.
    """
    torch = _torch()
    import torch.nn.functional as F

    p_w = torch.as_tensor(ctx[weak_probabilities]).detach()
    p_s = torch.as_tensor(ctx[strong_probabilities]).detach()
    labels = torch.as_tensor(ctx[noisy_labels], device=p_w.device).long().reshape(-1)
    if p_w.ndim != 2 or p_s.shape != p_w.shape or labels.shape != (p_w.shape[0],):
        raise ValueError("DLD pre-correction expects aligned [B,C] view posteriors and labels")
    if p_w.shape[1] < 2 or not torch.is_floating_point(p_w) or not torch.is_floating_point(p_s):
        raise ValueError("DLD pre-correction posteriors must be floating [B,C] tensors")
    if not bool(torch.isfinite(p_w).all()) or not bool(torch.isfinite(p_s).all()):
        raise ValueError("DLD pre-correction posteriors must be finite")
    if bool((p_w < 0).any()) or bool((p_s < 0).any()):
        raise ValueError("DLD pre-correction posteriors must be non-negative")
    row_ones = torch.ones(p_w.shape[0], device=p_w.device, dtype=p_w.dtype)
    if not torch.allclose(p_w.sum(1), row_ones, atol=1e-5, rtol=0) or not torch.allclose(p_s.sum(1), row_ones, atol=1e-5, rtol=0):
        raise ValueError("DLD pre-correction posteriors must be row-stochastic")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= int(p_w.shape[1])):
        raise ValueError("DLD pre-correction labels are outside the class range")

    floor = torch.finfo(p_w.dtype).tiny
    divergence = (p_s * (p_s.clamp_min(floor).log() - p_w.clamp_min(floor).log())).sum(dim=1)
    normalized = (divergence - divergence.min()) / (divergence.max() - divergence.min()).clamp_min(torch.finfo(p_w.dtype).eps)
    means = torch.stack((normalized.min(), normalized.max()))
    variance = normalized.var(unbiased=False).clamp_min(5e-4)
    variances = torch.full((2,), variance, device=p_w.device, dtype=p_w.dtype)
    weights = torch.full((2,), 0.5, device=p_w.device, dtype=p_w.dtype)
    posterior = torch.full((normalized.numel(), 2), 0.5, device=p_w.device, dtype=p_w.dtype)
    for _ in range(int(gmm_iterations)):
        log_prob = -0.5 * ((normalized[:, None] - means[None, :]).square() / variances[None, :] + variances.log()[None, :]) + weights.log()[None, :]
        posterior = (log_prob - log_prob.logsumexp(dim=1, keepdim=True)).exp()
        mass = posterior.sum(dim=0).clamp_min(torch.finfo(p_w.dtype).tiny)
        weights = mass / mass.sum()
        means = (posterior * normalized[:, None]).sum(dim=0) / mass
        variances = (posterior * (normalized[:, None] - means[None, :]).square()).sum(dim=0) / mass
        variances = variances.clamp_min(5e-4)
    low_component = torch.argmin(means)
    assignments = posterior.argmax(dim=1)
    p_ws = (p_w + p_s) * 0.5
    predicted = p_ws.argmax(dim=1)
    partition = torch.full_like(labels, 2, dtype=torch.long)
    low = assignments == low_component
    partition[low & predicted.eq(labels)] = 0
    partition[low & predicted.ne(labels)] = 1
    noisy_one_hot = F.one_hot(labels, p_w.shape[1]).to(p_w.dtype)
    corrected = F.one_hot(predicted, p_w.shape[1]).to(p_w.dtype)
    y0 = torch.empty_like(p_ws)
    y0[partition == 0] = noisy_one_hot[partition == 0]
    y0[partition == 1] = corrected[partition == 1]
    y0[partition == 2] = p_ws[partition == 2]
    yn = torch.zeros_like(p_ws)
    yn[partition == 1] = noisy_one_hot[partition == 1]
    hard = partition == 2
    if bool(hard.any()):
        delta = (p_w[hard] - p_s[hard]).abs()
        denominator = delta.sum(dim=1, keepdim=True)
        if bool((denominator <= 0).any()):
            raise ValueError("DLD hard pre-correction state has zero normalization")
        yn[hard] = delta / denominator
    ctx[y0_as] = y0
    ctx[yn_as] = yn
    ctx[direction_as] = yn - y0
    ctx[partition_as] = partition
    ctx[divergence_as] = divergence


@block(id="dld_sample_forward_state", name="DLD: Sample Forward State", category="Paper Specific", description="Sample the scheduled directional forward diffusion state.", params={"y0":{"type":"slot","default":"y0"},"direction":{"type":"slot","default":"direction"},"timestep":{"type":"slot","default":"timestep"},"epsilon":{"type":"slot","default":"epsilon"},"timesteps":{"type":"int","required":True,"min":1},"save_as":{"type":"slot","default":"yd"}}, requires=("y0","direction","timestep","epsilon"), provides=("save_as",), placement=("batch",), formula="y_t=y_0+alpha_bar(t)d+beta_bar(t)epsilon", formula_kind="special", formula_ref="DLD forward diffusion schedule", paper="Directional Label Diffusion")
def dld_sample_forward_state(ctx: ScratchContext, y0="y0", direction="direction", timestep="timestep", epsilon="epsilon", timesteps=None, save_as="yd"):
    if timesteps is None:
        raise ValueError("DLD forward state requires an explicit timesteps value")
    alpha_bar, beta_bar = _schedule(int(timesteps), device=ctx[y0].device, dtype=ctx[y0].dtype)
    t = ctx[timestep].long()
    ctx[save_as] = ctx[y0] + alpha_bar[t, None] * ctx[direction] + beta_bar[t, None] * ctx[epsilon]

@block(id="dld_accelerated_inference", name="DLD: Accelerated Label Inference", category="Evaluation", description="Run the paper's deterministic accelerated reverse-label trajectory with an explicit fixed number of inference steps.", params={"feature_model":{"type":"slot","default":"model"},"direction":{"type":"slot","default":"direction_predictor"},"noise":{"type":"slot","default":"noise_predictor"},"loader":{"type":"slot","default":"test_loader"},"timesteps":{"type":"int","required":True,"min":1},"steps":{"type":"int","required":True,"min":1},"save_as":{"type":"slot","default":"test_accuracy"}}, requires=("feature_model","direction","noise","loader"), provides=("save_as",), placement=("top",), stage="evaluate", ui_group="⑨ 评估", formula="y_0_hat=Reverse_DLD(y_T; K=steps)", formula_kind="special", formula_ref="DLD accelerated inference", paper="Directional Label Diffusion")
def dld_accelerated_inference(ctx: ScratchContext, feature_model="model", direction="direction_predictor", noise="noise_predictor", loader="test_loader", timesteps=None, steps=None, save_as="test_accuracy"):
    if timesteps is None or steps is None:
        raise ValueError("DLD accelerated inference requires explicit timesteps and steps")
    torch=_torch(); model=ctx[feature_model]; direction_model=getattr(ctx[direction], "model", ctx[direction]); noise_model=getattr(ctx[noise], "model", ctx[noise]); device=ctx.get("device",next(model.parameters()).device); correct=total=0
    if int(steps) <= 0 or int(steps) > int(timesteps):
        raise ValueError("DLD inference steps must satisfy 0 < steps <= timesteps")
    model.eval(); direction_model.eval(); noise_model.eval()
    with torch.no_grad():
        for batch in ctx[loader]:
            if isinstance(batch,dict): images=batch.get("input",batch.get("images")); labels=batch.get("target",batch.get("labels"))
            else: images,labels=batch[:2]
            output=model.forward_with_features(images.to(device)); features=output.features if hasattr(output,"features") else output[1]
            output_logits = output[0] if isinstance(output, (tuple, list)) else output.logits if hasattr(output, "logits") else output
            state = torch.zeros_like(output_logits)
            endpoint = torch.softmax(output_logits, dim=1)
            alpha_bar, beta_bar = _schedule(int(timesteps), device=device, dtype=features.dtype)
            points = torch.linspace(-1, int(timesteps) - 1, int(steps) + 1, device=device).long().flip(0)
            if torch.unique(points).numel() != points.numel():
                raise ValueError("DLD accelerated timestep sequence contains duplicates")
            for current, following in zip(points[:-1], points[1:]):
                current_index, following_index = int(current), int(following)
                timestep = torch.full((features.shape[0],), current_index, device=device, dtype=torch.long)
                next_alpha = 0.0 if following_index < 0 else alpha_bar[following_index]
                next_beta = 0.0 if following_index < 0 else beta_bar[following_index]
                predicted_direction = direction_model(state, endpoint, features.detach(), timestep)
                predicted_noise = noise_model(state, endpoint, features.detach(), timestep)
                state = state - (alpha_bar[current_index] - next_alpha) * predicted_direction - (beta_bar[current_index] - next_beta) * predicted_noise
                if not bool(torch.isfinite(state).all()):
                    raise ValueError("DLD reverse sample became non-finite")
            prediction = state
            correct+=int(prediction.argmax(1).eq(labels.to(device).long()).sum().item()); total+=int(labels.numel())
    ctx[save_as]=correct/max(total,1)
