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


class _DLDLabelPredictor(_torch().nn.Module):
    def __init__(self, classes, feature_dim, hidden_dim, time_dim):
        super().__init__()
        width = int(classes) * 2 + int(feature_dim) + int(time_dim)
        self.classes = int(classes); self.feature_dim = int(feature_dim); self.time_dim = int(time_dim)
        self.network = self.__class__._torch_nn(width, int(hidden_dim), int(classes))

    @staticmethod
    def _torch_nn(width, hidden, classes):
        torch = _torch(); nn = torch.nn
        return nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, classes))

    def forward(self, y_t, y_n, features, timestep):
        torch = _torch()
        embedded = _timestep_embedding(timestep.long(), self.time_dim).to(y_t)
        return self.network(torch.cat((y_t, y_n, features, embedded), dim=1))

@block(id="dld_sample_forward_state", name="DLD: Sample Forward State", category="Paper Specific", description="Sample the scheduled directional forward diffusion state.", params={"y0":{"type":"slot","default":"y0"},"direction":{"type":"slot","default":"direction"},"timestep":{"type":"slot","default":"timestep"},"epsilon":{"type":"slot","default":"epsilon"},"timesteps":{"type":"int","default":1000,"min":1},"save_as":{"type":"slot","default":"yd"}}, requires=("y0","direction","timestep","epsilon"), provides=("save_as",), placement=("batch",), formula="y_t=y_0+alpha_bar(t)d+beta_bar(t)epsilon", formula_ref="DLD forward diffusion schedule", paper="Directional Label Diffusion")
def dld_sample_forward_state(ctx: ScratchContext, y0="y0", direction="direction", timestep="timestep", epsilon="epsilon", timesteps=1000, save_as="yd"):
    alpha_bar, beta_bar = _schedule(int(timesteps), device=ctx[y0].device, dtype=ctx[y0].dtype)
    t = ctx[timestep].long()
    ctx[save_as] = ctx[y0] + alpha_bar[t, None] * ctx[direction] + beta_bar[t, None] * ctx[epsilon]

@block(id="dld_ensure_independent_predictors", name="DLD: Initialize Independent Predictors", category="Model", description="Lazily initialize separate direction and noise predictors plus their independent optimizers from the exposed feature width.", params={"features":{"type":"slot","default":"features"},"num_classes":{"type":"int","default":10,"min":2},"hidden_dim":{"type":"int","default":64,"min":1},"time_dim":{"type":"int","default":16,"min":1},"learning_rate":{"type":"float","default":0.0001,"min":0.0}}, requires=("features",), provides=("direction_predictor","noise_predictor","direction_optimizer","noise_optimizer"), placement=("batch",), stage="setup", ui_group="② 初始化", formula="theta_d independent theta_epsilon", formula_ref="DLD independent predictors", paper="Directional Label Diffusion")
def dld_ensure_independent_predictors(ctx: ScratchContext, features="features", num_classes=10, hidden_dim=64, time_dim=16, learning_rate=0.0001):
    if (ctx.get("direction_predictor") is not None
            and ctx.get("noise_predictor") is not None
            and ctx.get("direction_optimizer") is not None
            and ctx.get("noise_optimizer") is not None):
        return
    torch=_torch()
    width=int(ctx[features].shape[1]); device=ctx[features].device
    ctx["direction_predictor"]=_DLDLabelPredictor(int(num_classes),width,hidden_dim=int(hidden_dim),time_dim=int(time_dim)).to(device)
    ctx["noise_predictor"]=_DLDLabelPredictor(int(num_classes),width,hidden_dim=int(hidden_dim),time_dim=int(time_dim)).to(device)
    ctx["direction_optimizer"]=torch.optim.Adam(ctx["direction_predictor"].parameters(),lr=float(learning_rate)); ctx["noise_optimizer"]=torch.optim.Adam(ctx["noise_predictor"].parameters(),lr=float(learning_rate))


@block(id="dld_predict_quantity", name="DLD: Predict Diffusion Quantity", category="Forward", description="Apply one reusable DLD predictor to the noisy state, endpoint, frozen features, and timestep.", params={"predictor":{"type":"slot","default":"direction_predictor"},"state":{"type":"slot","default":"yd"},"endpoint":{"type":"slot","default":"yn"},"features":{"type":"slot","default":"features"},"timestep":{"type":"slot","default":"timestep"},"save_as":{"type":"slot","default":"predicted_direction"}}, requires=("predictor","state","endpoint","features","timestep"), provides=("save_as",), placement=("batch",), formula="q_theta(y_t,y_n,h,t)", formula_ref="DLD direction/noise predictor", paper="Directional Label Diffusion")
def dld_predict_quantity(ctx: ScratchContext, predictor="direction_predictor", state="yd", endpoint="yn", features="features", timestep="timestep", save_as="predicted_direction"):
    ctx[save_as]=ctx[predictor](ctx[state],ctx[endpoint],ctx[features].detach(),ctx[timestep].long())


@block(id="dld_update_ema", name="DLD: Update Predictor EMA", category="State", description="Maintain independent exponential moving averages of direction and noise predictor parameters.", params={"direction":{"type":"slot","default":"direction_predictor"},"noise":{"type":"slot","default":"noise_predictor"},"decay":{"type":"float","default":0.999,"min":0.0,"max":1.0}}, requires=("direction","noise"), provides=("dld_ema_state",), placement=("batch",), stage="train", ui_group="④ 状态更新", formula="theta_ema<-rho theta_ema+(1-rho)theta", formula_ref="DLD EMA", paper="Directional Label Diffusion")
def dld_update_ema(ctx: ScratchContext, direction="direction_predictor", noise="noise_predictor", decay=0.999):
    torch=_torch(); current={"direction":ctx[direction].state_dict(),"noise":ctx[noise].state_dict()}
    if "dld_ema_state" not in ctx: ctx["dld_ema_state"]={role:{k:v.detach().clone() for k,v in state.items()} for role,state in current.items()}; return
    with torch.no_grad():
        for role,state in current.items():
            for key,value in state.items(): ctx["dld_ema_state"][role][key].mul_(float(decay)).add_(value.detach(),alpha=1.0-float(decay))


@block(id="dld_accelerated_inference", name="DLD: Accelerated Label Inference", category="Evaluation", description="Run the paper's deterministic accelerated reverse-label trajectory with an explicit fixed number of inference steps.", params={"feature_model":{"type":"slot","default":"model"},"direction":{"type":"slot","default":"direction_predictor"},"noise":{"type":"slot","default":"noise_predictor"},"loader":{"type":"slot","default":"test_loader"},"timesteps":{"type":"int","default":1000,"min":1},"steps":{"type":"int","default":5,"min":1},"save_as":{"type":"slot","default":"test_accuracy"}}, requires=("feature_model","direction","noise","loader"), provides=("save_as",), placement=("top",), stage="evaluate", ui_group="⑨ 评估", formula="y_0_hat=Reverse_DLD(y_T; K=5)", formula_ref="DLD accelerated inference", paper="Directional Label Diffusion")
def dld_accelerated_inference(ctx: ScratchContext, feature_model="model", direction="direction_predictor", noise="noise_predictor", loader="test_loader", timesteps=1000, steps=5, save_as="test_accuracy"):
    torch=_torch(); model=ctx[feature_model]; device=ctx.get("device",next(model.parameters()).device); correct=total=0
    model.eval(); ctx[direction].eval(); ctx[noise].eval()
    with torch.no_grad():
        for batch in ctx[loader]:
            if isinstance(batch,dict): images=batch.get("input",batch.get("images")); labels=batch.get("target",batch.get("labels"))
            else: images,labels=batch[:2]
            output=model.forward_with_features(images.to(device)); features=output.features if hasattr(output,"features") else output[1]
            alpha_bar, beta_bar = _schedule(int(timesteps), device=device, dtype=features.dtype)
            state = torch.zeros_like(output[0] if isinstance(output, (tuple, list)) else features[:, :ctx[direction].classes])
            for index in torch.linspace(int(timesteps) - 1, 0, int(steps), device=device).long():
                step = torch.full((features.shape[0],), int(index), device=device, dtype=torch.long)
                state = ctx[direction](state, state, features.detach(), step)
            prediction = state
            correct+=int(prediction.argmax(1).eq(labels.to(device).long()).sum().item()); total+=int(labels.numel())
    ctx[save_as]=correct/max(total,1)
