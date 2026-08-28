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


@block(id="dld_sample_forward_state", name="DLD: Sample Forward State", category="Paper Specific", description="Sample the scheduled directional forward diffusion state.", params={"y0":{"type":"slot","default":"y0"},"direction":{"type":"slot","default":"direction"},"timestep":{"type":"slot","default":"timestep"},"epsilon":{"type":"slot","default":"epsilon"},"timesteps":{"type":"int","default":1000,"min":1},"save_as":{"type":"slot","default":"yd"}}, requires=("y0","direction","timestep","epsilon"), provides=("save_as",), placement=("batch",), formula="y_t=y_0+alpha_bar(t)d+beta_bar(t)epsilon", formula_ref="DLD forward diffusion schedule", paper="Directional Label Diffusion")
def dld_sample_forward_state(ctx: ScratchContext, y0="y0", direction="direction", timestep="timestep", epsilon="epsilon", timesteps=1000, save_as="yd"):
    alpha_bar, beta_bar = _schedule(int(timesteps), device=ctx[y0].device, dtype=ctx[y0].dtype)
    t = ctx[timestep].long()
    ctx[save_as] = ctx[y0] + alpha_bar[t, None] * ctx[direction] + beta_bar[t, None] * ctx[epsilon]

@block(id="dld_predict_quantity", name="DLD: Predict Diffusion Quantity", category="Forward", description="Apply one reusable DLD predictor to the noisy state, endpoint, frozen features, and timestep.", params={"predictor":{"type":"slot","default":"direction_predictor"},"state":{"type":"slot","default":"yd"},"endpoint":{"type":"slot","default":"yn"},"features":{"type":"slot","default":"features"},"timestep":{"type":"slot","default":"timestep"},"save_as":{"type":"slot","default":"predicted_direction"}}, requires=("predictor","state","endpoint","features","timestep"), provides=("save_as",), placement=("batch",), formula="q_theta(y_t,y_n,h,t)", formula_ref="DLD direction/noise predictor", paper="Directional Label Diffusion")
def dld_predict_quantity(ctx: ScratchContext, predictor="direction_predictor", state="yd", endpoint="yn", features="features", timestep="timestep", save_as="predicted_direction"):
    ctx[save_as]=ctx[predictor](ctx[state],ctx[endpoint],ctx[features].detach(),ctx[timestep].long())


@block(id="dld_accelerated_inference", name="DLD: Accelerated Label Inference", category="Evaluation", description="Run the paper's deterministic accelerated reverse-label trajectory with an explicit fixed number of inference steps.", params={"feature_model":{"type":"slot","default":"model"},"direction":{"type":"slot","default":"direction_predictor"},"noise":{"type":"slot","default":"noise_predictor"},"loader":{"type":"slot","default":"test_loader"},"timesteps":{"type":"int","default":1000,"min":1},"steps":{"type":"int","default":5,"min":1},"save_as":{"type":"slot","default":"test_accuracy"}}, requires=("feature_model","direction","noise","loader"), provides=("save_as",), placement=("top",), stage="evaluate", ui_group="⑨ 评估", formula="y_0_hat=Reverse_DLD(y_T; K=5)", formula_ref="DLD accelerated inference", paper="Directional Label Diffusion")
def dld_accelerated_inference(ctx: ScratchContext, feature_model="model", direction="direction_predictor", noise="noise_predictor", loader="test_loader", timesteps=1000, steps=5, save_as="test_accuracy"):
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
