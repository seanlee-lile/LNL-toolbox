"""Explicit directional-label-diffusion lifecycle blocks."""
from __future__ import annotations
from ...context import ScratchContext
from ...registry import block

def _torch():
    import torch
    return torch

@block(id="dld_prepare_targets", name="DLD: Prepare Diffusion Targets", category="Paper Specific", description="Prepare averaged-view y0, estimated yn, timestep, and Gaussian noise tensors.", params={"logits":{"type":"slot","default":"logits"},"labels":{"type":"slot","default":"labels"},"num_classes":{"type":"int","default":10,"min":2}}, requires=("logits","labels"), provides=("y0","yn","timestep","epsilon"), placement=("batch",), formula="y0=one_hot(y); yn=softmax(f(x))", formula_ref="DLD target preparation", paper="Directional Label Diffusion")
def dld_prepare_targets(ctx: ScratchContext, logits="logits", labels="labels", num_classes=10):
    torch=_torch(); z=ctx[logits]; y=torch.nn.functional.one_hot(ctx[labels].long(),int(num_classes)).to(z.dtype); p=torch.softmax(z.detach(),1)
    ctx["y0"]=y; ctx["yn"]=p; ctx["timestep"]=torch.zeros(z.shape[0],dtype=torch.long,device=z.device); ctx["epsilon"]=torch.zeros_like(z)

@block(id="dld_construct_direction", name="DLD: Construct Direction", category="Paper Specific", description="Construct the directional target y_n-y_0.", params={"y0":{"type":"slot","default":"y0"},"yn":{"type":"slot","default":"yn"},"save_as":{"type":"slot","default":"direction"}}, requires=("y0","yn"), provides=("save_as",), placement=("batch",), formula="d=y_n-y_0", formula_ref="DLD directional target", paper="Directional Label Diffusion")
def dld_construct_direction(ctx: ScratchContext, y0="y0", yn="yn", save_as="direction"):
    from lnl_toolbox.algorithms.dld.objective import construct_direction
    ctx[save_as] = construct_direction(ctx[y0], ctx[yn])

@block(id="dld_sample_forward_state", name="DLD: Sample Forward State", category="Paper Specific", description="Sample the scheduled directional forward diffusion state.", params={"y0":{"type":"slot","default":"y0"},"direction":{"type":"slot","default":"direction"},"timestep":{"type":"slot","default":"timestep"},"epsilon":{"type":"slot","default":"epsilon"},"timesteps":{"type":"int","default":1000,"min":1},"save_as":{"type":"slot","default":"yd"}}, requires=("y0","direction","timestep","epsilon"), provides=("save_as",), placement=("batch",), formula="y_t=y_0+alpha_bar(t)d+beta_bar(t)epsilon", formula_ref="DLD forward diffusion schedule", paper="Directional Label Diffusion")
def dld_sample_forward_state(ctx: ScratchContext, y0="y0", direction="direction", timestep="timestep", epsilon="epsilon", timesteps=1000, save_as="yd"):
    from lnl_toolbox.algorithms.dld.objective import sample_forward_state
    from lnl_toolbox.algorithms.dld.schedules import DirectionalDiffusionSchedule
    ctx[save_as] = sample_forward_state(ctx[y0], ctx[direction], ctx[timestep].long(), ctx[epsilon], DirectionalDiffusionSchedule.average(int(timesteps), device=ctx[y0].device, dtype=ctx[y0].dtype))

@block(id="dld_direction_loss", name="DLD: Direction MSE", category="Loss", description="Compute the directional predictor MSE term.", params={"predicted":{"type":"slot","default":"predicted_direction"},"target":{"type":"slot","default":"direction"},"save_as":{"type":"slot","default":"dld_direction_loss"}}, requires=("predicted","target"), provides=("save_as",), placement=("batch",), formula="L_d=MSE(d_hat,d)", formula_ref="DLD direction objective", paper="Directional Label Diffusion")
def dld_direction_loss(ctx: ScratchContext, predicted="predicted_direction", target="direction", save_as="dld_direction_loss"):
    ctx[save_as] = (ctx[predicted] - ctx[target]).square().mean(dim=1).mean()

@block(id="dld_noise_loss", name="DLD: Noise MSE", category="Loss", description="Compute the noise predictor MSE term.", params={"predicted":{"type":"slot","default":"predicted_noise"},"target":{"type":"slot","default":"epsilon"},"save_as":{"type":"slot","default":"dld_noise_loss"}}, requires=("predicted","target"), provides=("save_as",), placement=("batch",), formula="L_e=MSE(e_hat,e)", formula_ref="DLD noise objective", paper="Directional Label Diffusion")
def dld_noise_loss(ctx: ScratchContext, predicted="predicted_noise", target="epsilon", save_as="dld_noise_loss"):
    ctx[save_as] = (ctx[predicted] - ctx[target]).square().mean(dim=1).mean()

@block(id="dld_objective_composition", name="DLD: Compose Diffusion Objective", category="Loss", description="Compose direction and noise MSE terms.", params={"direction_loss":{"type":"slot","default":"dld_direction_loss"},"noise_loss":{"type":"slot","default":"dld_noise_loss"},"save_as":{"type":"slot","default":"loss"}}, requires=("direction_loss","noise_loss"), provides=("save_as",), placement=("batch",), formula="L=L_d+L_e", formula_ref="DLD objective composition", paper="Directional Label Diffusion")
def dld_objective_composition(ctx: ScratchContext, direction_loss="dld_direction_loss", noise_loss="dld_noise_loss", save_as="loss"):
    ctx[save_as] = ctx[direction_loss] + ctx[noise_loss]
