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


@block(id="dld_onehot_start", name="DLD: Construct Start Label", category="Paper Specific", description="Construct the diffusion start label vector from the observed label.", params={"labels":{"type":"slot","default":"labels"},"num_classes":{"type":"int","default":10,"min":2},"save_as":{"type":"slot","default":"y0"}}, requires=("labels",), provides=("save_as",), placement=("batch",), formula="y_0=one_hot(y~)", formula_ref="DLD start state", paper="Directional Label Diffusion")
def dld_onehot_start(ctx: ScratchContext, labels="labels", num_classes=10, save_as="y0"):
    torch=_torch(); ctx[save_as]=torch.nn.functional.one_hot(ctx[labels].long(),int(num_classes)).float()


@block(id="dld_estimate_endpoint", name="DLD: Estimate Corrected Endpoint", category="Paper Specific", description="Publish the detached corrected endpoint distribution supplied by frozen feature pre-correction.", params={"logits":{"type":"slot","default":"logits"},"save_as":{"type":"slot","default":"yn"}}, requires=("logits",), provides=("save_as",), placement=("batch",), formula="y_n=correct(f_frozen(x))", formula_ref="DLD pre-correction endpoint", paper="Directional Label Diffusion")
def dld_estimate_endpoint(ctx: ScratchContext, logits="logits", save_as="yn"):
    torch=_torch(); ctx[save_as]=torch.softmax(ctx[logits].detach(),1)


@block(id="dld_sample_timestep_noise", name="DLD: Sample Timestep and Noise", category="Paper Specific", description="Sample independent diffusion timesteps and Gaussian noise for the current minibatch.", params={"reference":{"type":"slot","default":"y0"},"timesteps":{"type":"int","default":1000,"min":1},"timestep_as":{"type":"slot","default":"timestep"},"noise_as":{"type":"slot","default":"epsilon"}}, requires=("reference",), provides=("timestep_as","noise_as"), placement=("batch",), formula="t~Uniform{0,...,T-1}; epsilon~N(0,I)", formula_ref="DLD stochastic training", paper="Directional Label Diffusion")
def dld_sample_timestep_noise(ctx: ScratchContext, reference="y0", timesteps=1000, timestep_as="timestep", noise_as="epsilon"):
    torch=_torch(); value=ctx[reference]; ctx[timestep_as]=torch.randint(0,int(timesteps),(value.shape[0],),device=value.device); ctx[noise_as]=torch.randn_like(value)


@block(id="dld_ensure_independent_predictors", name="DLD: Initialize Independent Predictors", category="Model", description="Lazily initialize separate direction and noise predictors plus their independent optimizers from the exposed feature width.", params={"features":{"type":"slot","default":"features"},"num_classes":{"type":"int","default":10,"min":2},"hidden_dim":{"type":"int","default":64,"min":1},"time_dim":{"type":"int","default":16,"min":1},"learning_rate":{"type":"float","default":0.0001,"min":0.0}}, requires=("features",), provides=("direction_predictor","noise_predictor","direction_optimizer","noise_optimizer"), placement=("batch",), stage="setup", ui_group="② 初始化", formula="theta_d independent theta_epsilon", formula_ref="DLD independent predictors", paper="Directional Label Diffusion")
def dld_ensure_independent_predictors(ctx: ScratchContext, features="features", num_classes=10, hidden_dim=64, time_dim=16, learning_rate=0.0001):
    if "direction_predictor" in ctx: return
    torch=_torch(); from lnl_toolbox.algorithms.dld.networks import DLDLabelPredictor
    width=int(ctx[features].shape[1]); device=ctx[features].device
    ctx["direction_predictor"]=DLDLabelPredictor(int(num_classes),width,hidden_dim=int(hidden_dim),time_dim=int(time_dim)).to(device)
    ctx["noise_predictor"]=DLDLabelPredictor(int(num_classes),width,hidden_dim=int(hidden_dim),time_dim=int(time_dim)).to(device)
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
    torch=_torch(); from lnl_toolbox.algorithms.dld.sampling import sample_labels; from lnl_toolbox.algorithms.dld.schedules import DirectionalDiffusionSchedule
    model=ctx[feature_model]; device=ctx.get("device",next(model.parameters()).device); correct=total=0; schedule=DirectionalDiffusionSchedule.average(int(timesteps),device=device)
    model.eval(); ctx[direction].eval(); ctx[noise].eval()
    with torch.no_grad():
        for batch in ctx[loader]:
            if isinstance(batch,dict): images=batch.get("input",batch.get("images")); labels=batch.get("target",batch.get("labels"))
            else: images,labels=batch[:2]
            output=model.forward_with_features(images.to(device)); features=output.features if hasattr(output,"features") else output[1]
            prediction=sample_labels(ctx[direction],ctx[noise],features.detach(),schedule,inference_steps=int(steps))
            correct+=int(prediction.argmax(1).eq(labels.to(device).long()).sum().item()); total+=int(labels.numel())
    ctx[save_as]=correct/max(total,1)
