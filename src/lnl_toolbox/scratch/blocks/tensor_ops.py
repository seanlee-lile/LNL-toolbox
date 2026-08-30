"""Scratch-native research tensor operations shared across paper recipes."""

from __future__ import annotations

from typing import Any

from ..context import ScratchContext
from ..registry import block


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("Tensor operations require PyTorch; install the `train` extra.") from exc
    return torch, F


@block(
    id="one_hot",
    name="One-hot Labels",
    category="Tensor Operation",
    description="Convert integer labels to one-hot vectors.",
    params={"labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "one_hot_labels"}},
    requires=("labels",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def one_hot(ctx: ScratchContext, labels: str = "labels", num_classes: int = 10, save_as: str = "one_hot_labels") -> None:
    torch, F = _torch()
    ctx[save_as] = F.one_hot(ctx[labels].long(), int(num_classes)).to(dtype=torch.float32)


@block(
    id="zeros_like",
    name="Zeros Like",
    category="Tensor Operation",
    description="Create a zero tensor matching an input tensor's shape and dtype.",
    params={"input": {"type": "slot", "default": "values"}, "requires_grad": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "zeros"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def zeros_like(ctx: ScratchContext, input: str = "values", requires_grad: bool = False, save_as: str = "zeros") -> None:
    ctx[save_as] = __import__("torch").zeros_like(ctx[input], requires_grad=bool(requires_grad))


@block(
    id="ones_like",
    name="Ones Like",
    category="Tensor Operation",
    description="Create a floating-point tensor of ones matching an input tensor's shape.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "ones"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
)
def ones_like(ctx: ScratchContext, input: str = "values", save_as: str = "ones") -> None:
    import torch
    ctx[save_as] = torch.ones_like(ctx[input], dtype=torch.float32)


@block(
    id="elementwise_multiply",
    name="Elementwise Multiply",
    category="Tensor Operation",
    description="Multiply two aligned tensors element by element without reduction.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "product"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="z=x⊙y", formula_ref="elementwise multiplication",
)
def elementwise_multiply(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "product") -> None:
    if ctx[left].shape != ctx[right].shape:
        raise ValueError("elementwise_multiply inputs must have the same shape")
    ctx[save_as] = ctx[left] * ctx[right]


@block(
    id="sum_values",
    name="Sum Values",
    category="Tensor Operation",
    description="Reduce all elements of a tensor to one scalar sum.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "sum"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="s=Σ_i x_i", formula_ref="sum reduction",
)
def sum_values(ctx: ScratchContext, input: str = "values", save_as: str = "sum") -> None:
    ctx[save_as] = ctx[input].sum()


@block(
    id="detach",
    name="Detach Tensor",
    category="Tensor Operation",
    description="Publish a detached tensor without changing its values.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "detached"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
)
def detach(ctx: ScratchContext, input: str = "input", save_as: str = "detached") -> None:
    ctx[save_as] = ctx[input].detach()


@block(
    id="uniform_prior",
    name="Uniform Class Prior",
    category="Tensor Operation",
    description="Create a uniform class prior matching a probability vector.",
    params={"reference": {"type": "slot", "default": "probabilities"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "prior"}},
    requires=("reference",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def uniform_prior(ctx: ScratchContext, reference: str = "probabilities", num_classes: int = 10, save_as: str = "prior") -> None:
    import torch
    classes = int(ctx[reference].shape[-1]) if ctx[reference].ndim else int(num_classes)
    ctx[save_as] = torch.full((classes,), 1.0 / classes, dtype=ctx[reference].dtype, device=ctx[reference].device)


@block(
    id="subtract",
    name="Subtract",
    category="Tensor Operation",
    description="Subtract one tensor or scalar slot from another.",
    params={"minuend": {"type": "slot", "default": "first"}, "subtrahend": {"type": "slot", "default": "second"}, "save_as": {"type": "slot", "default": "difference"}},
    requires=("minuend", "subtrahend"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
)
def subtract(ctx: ScratchContext, minuend: str = "first", subtrahend: str = "second", save_as: str = "difference") -> None:
    ctx[save_as] = ctx[minuend] - ctx[subtrahend]


@block(
    id="sample_timestep_noise",
    name="Sample Timestep and Noise",
    category="Tensor Operation",
    description="Sample independent integer timesteps and Gaussian noise shaped like a reference tensor.",
    params={"reference": {"type": "slot", "default": "reference"}, "timesteps": {"type": "int", "default": 1000, "min": 1}, "timestep_as": {"type": "slot", "default": "timestep"}, "noise_as": {"type": "slot", "default": "epsilon"}},
    requires=("reference",), provides=("timestep_as", "noise_as"), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def sample_timestep_noise(ctx: ScratchContext, reference: str = "reference", timesteps: int = 1000, timestep_as: str = "timestep", noise_as: str = "epsilon") -> None:
    torch = _torch()[0]
    value = ctx[reference]
    ctx[timestep_as] = torch.randint(0, int(timesteps), (value.shape[0],), device=value.device)
    ctx[noise_as] = torch.randn_like(value)


@block(
    id="mean_squared_error",
    name="Mean Squared Error",
    category="Loss",
    description="Compute squared error with an explicit per-sample or scalar reduction.",
    params={"predicted": {"type": "slot", "default": "predicted"}, "target": {"type": "slot", "default": "target"}, "mask": {"type": "value", "default": None}, "reduction": {"type": "enum", "options": ["per_sample", "scalar"], "default": "scalar"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("predicted", "target"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="MSE(x,y)=mean((x-y)^2)", formula_ref="mean squared error",
)
def mean_squared_error(ctx: ScratchContext, predicted: str = "predicted", target: str = "target", mask: Any = None, reduction: str = "scalar", save_as: str = "loss") -> None:
    predicted_values, target_values = ctx[predicted], ctx[target]
    if isinstance(mask, str) and mask in ctx:
        predicted_values, target_values = predicted_values[ctx[mask].bool()], target_values[ctx[mask].bool()]
    values = (predicted_values - target_values).square()
    if str(reduction) == "per_sample":
        values = values.reshape(values.shape[0], -1).mean(dim=1)
    else:
        values = values.mean()
    ctx[save_as] = values


@block(
    id="weighted_sum",
    name="Weighted Sum",
    category="Loss",
    description="Sum explicitly named scalar or tensor terms with explicit weights.",
    params={"terms": {"type": "value", "default": []}, "weights": {"type": "value", "default": []}, "save_as": {"type": "slot", "default": "loss"}},
    provides=("save_as",), placement=("batch", "epoch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="L=sum_i w_i L_i", formula_ref="weighted objective composition",
)
def weighted_sum(ctx: ScratchContext, terms: Any = (), weights: Any = (), save_as: str = "loss") -> None:
    if not isinstance(terms, (list, tuple)) or not terms:
        raise ValueError("weighted_sum requires at least one term slot")
    if not isinstance(weights, (list, tuple)) or len(weights) != len(terms):
        raise ValueError("weighted_sum terms and weights must have equal length")
    result = None
    for slot, weight in zip(terms, weights):
        if not isinstance(slot, str) or slot not in ctx:
            raise KeyError(f"weighted_sum missing term slot: {slot}")
        value = float(weight) * ctx[slot]
        result = value if result is None else result + value
    ctx[save_as] = result


@block(
    id="nonnegative_projection",
    name="Nonnegative Projection",
    category="Weighting",
    description="Project a vector or its negation onto the nonnegative orthant.",
    params={"input": {"type": "slot", "default": "gradient"}, "negate": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "nonnegative_values"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w=max(0,+/-g)", formula_ref="nonnegative projection",
)
def nonnegative_projection(ctx: ScratchContext, input: str = "gradient", negate: bool = False, save_as: str = "nonnegative_values") -> None:
    torch, _ = _torch()
    values = -ctx[input] if bool(negate) else ctx[input]
    # Projection is a value operation.  Detachment, where required by a
    # method, is an explicit downstream ``detach`` block.
    ctx[save_as] = torch.relu(values)


@block(
    id="normalize_nonnegative_weights",
    name="Normalize Nonnegative Weights",
    category="Weighting",
    description="Normalize nonnegative weights to sum to one while preserving the all-zero case.",
    params={"weights": {"type": "slot", "default": "nonnegative_values"}, "save_as": {"type": "slot", "default": "weights"}},
    requires=("weights",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w=wbar/sum(wbar)", formula_ref="normalized nonnegative weighting",
)
def normalize_nonnegative_weights(ctx: ScratchContext, weights: str = "nonnegative_values", save_as: str = "weights") -> None:
    values = ctx[weights]
    total = values.sum()
    ctx[save_as] = values / total if bool(total > 0) else __import__("torch").zeros_like(values)


@block(
    id="negative_log",
    name="Negative Log",
    category="Tensor Operation",
    description="Apply an explicit finite floor and negative logarithm.",
    params={"input": {"type": "slot", "default": "probabilities"}, "minimum": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "negative_log_values"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def negative_log(ctx: ScratchContext, input: str = "probabilities", minimum: float = 1e-12, save_as: str = "negative_log_values") -> None:
    ctx[save_as] = -ctx[input].clamp_min(float(minimum)).log()


@block(
    id="safe_divide",
    name="Safe Divide",
    category="Tensor Operation",
    description="Divide two tensors after applying an explicit denominator floor.",
    params={"numerator": {"type": "slot", "default": "numerator"}, "denominator": {"type": "slot", "default": "denominator"}, "minimum": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "quotient"}},
    requires=("numerator", "denominator"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def safe_divide(ctx: ScratchContext, numerator: str = "numerator", denominator: str = "denominator", minimum: float = 1e-12, save_as: str = "quotient") -> None:
    ctx[save_as] = ctx[numerator] / ctx[denominator].clamp_min(float(minimum))


@block(
    id="weighted_blend",
    name="Weighted Blend",
    category="Tensor Operation",
    description="Blend two tensors using a sample-wise weight.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "weight": {"type": "slot", "default": "weight"}, "clamp_weight": {"type": "bool", "default": True}, "save_as": {"type": "slot", "default": "blended"}},
    requires=("left", "right", "weight"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="z=w left+(1-w) right", formula_ref="weighted blend",
)
def weighted_blend(ctx: ScratchContext, left: str = "left", right: str = "right", weight: str = "weight", clamp_weight: bool = True, save_as: str = "blended") -> None:
    values = ctx[weight]
    if bool(clamp_weight):
        values = values.clamp(0.0, 1.0)
    while values.ndim < ctx[left].ndim:
        values = values.unsqueeze(-1)
    ctx[save_as] = values * ctx[left] + (1.0 - values) * ctx[right]


@block(
    id="sharpen_distribution",
    name="Sharpen Distribution",
    category="Tensor Operation",
    description="Apply temperature sharpening and renormalize a class distribution.",
    params={"input": {"type": "slot", "default": "targets"}, "temperature": {"type": "float", "default": 0.5, "min": 0.0001}, "save_as": {"type": "slot", "default": "sharpened"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="q_c'=q_c^(1/T)/sum_j q_j^(1/T)", formula_ref="distribution sharpening",
)
def sharpen_distribution(ctx: ScratchContext, input: str = "targets", temperature: float = 0.5, save_as: str = "sharpened") -> None:
    values = ctx[input].clamp_min(0).pow(1.0 / float(temperature))
    ctx[save_as] = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@block(
    id="soft_target_cross_entropy",
    name="Soft-target Cross Entropy",
    category="Loss",
    description="Compute cross entropy against probability targets, optionally restricted by a mask.",
    params={"logits": {"type": "slot", "default": "logits"}, "targets": {"type": "slot", "default": "targets"}, "mask": {"type": "value", "default": None}, "reduction": {"type": "enum", "options": ["per_sample", "mean"], "default": "mean"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "targets"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="CE(q,z)=-sum_c q_c log softmax(z)_c", formula_ref="soft-target cross entropy",
)
def soft_target_cross_entropy(ctx: ScratchContext, logits: str = "logits", targets: str = "targets", mask: Any = None, reduction: str = "mean", save_as: str = "loss") -> None:
    _, F = _torch()
    values = -(ctx[targets] * F.log_softmax(ctx[logits], dim=-1)) .sum(dim=-1)
    if isinstance(mask, str) and mask in ctx:
        values = values[ctx[mask].bool()]
    ctx[save_as] = values if str(reduction) == "per_sample" else (values.mean() if values.numel() else ctx[logits].sum() * 0.0)


@block(
    id="apply_transition",
    name="Apply Transition Matrix",
    category="Correction",
    description="Map clean-class probabilities through a shared or per-sample transition matrix.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "transition": {"type": "slot", "default": "transition"}, "save_as": {"type": "slot", "default": "noisy_probabilities"}},
    requires=("probabilities", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def apply_transition(ctx: ScratchContext, probabilities: str = "probabilities", transition: str = "transition", save_as: str = "noisy_probabilities") -> None:
    values = ctx[probabilities]
    matrix = ctx[transition]
    if hasattr(matrix, "transition_for"):
        if "indices" not in ctx:
            raise ValueError("transition artifacts require aligned indices; materialize_transition first")
        matrix = matrix.transition_for(None, ctx["indices"], device=values.device, dtype=values.dtype)
    elif hasattr(matrix, "to"):
        matrix = matrix.to(device=values.device, dtype=values.dtype)
    if matrix.ndim == 2:
        ctx[save_as] = values @ matrix
    elif matrix.ndim == 3:
        ctx[save_as] = __import__("torch").einsum("nc,ncd->nd", values, matrix)
    else:
        raise ValueError("transition must have shape [C,C] or [N,C,C]")


@block(
    id="row_normalize",
    name="Row Normalize",
    category="Tensor Operation",
    description="Normalize the last dimension of a tensor with an explicit positive floor.",
    params={"input": {"type": "slot", "default": "values"},
            "minimum": {"type": "float", "default": 1e-12, "min": 0.0},
            "save_as": {"type": "slot", "default": "normalized"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="setup", ui_group="⑤ 损失公式",
)
def row_normalize(ctx: ScratchContext, input: str = "values", minimum: float = 1e-12,
                  save_as: str = "normalized") -> None:
    values = ctx[input]
    denominator = values.sum(dim=-1, keepdim=True)
    floor = float(minimum)
    if floor < 0:
        raise ValueError("row_normalize minimum must be non-negative")
    ctx[save_as] = values / denominator.clamp_min(floor)


@block(
    id="positive_logdet",
    name="Positive Log Determinant",
    category="Tensor Operation",
    description="Return log-determinants only for matrices with a strictly positive determinant.",
    params={"matrix": {"type": "slot", "default": "matrix"},
            "save_as": {"type": "slot", "default": "logdet"}},
    requires=("matrix",), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑤ 损失公式",
)
def positive_logdet(ctx: ScratchContext, matrix: str = "matrix", save_as: str = "logdet") -> None:
    torch = __import__("torch")
    values = ctx[matrix]
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("positive_logdet expects square matrices")
    sign, logabs = torch.linalg.slogdet(values)
    if bool((sign <= 0).any()) or not bool(torch.isfinite(logabs).all()):
        raise ValueError("positive_logdet requires finite matrices with positive determinant")
    ctx[save_as] = logabs


@block(
    id="compose_transition",
    name="Compose Transition Matrices",
    category="Transition",
    description="Compose two row-stochastic transition matrices and normalize each output row.",
    params={"first": {"type": "slot", "default": "transition_a"}, "second": {"type": "slot", "default": "transition_b"}, "save_as": {"type": "slot", "default": "composed_transition"}},
    requires=("first", "second"), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="T=T_1T_2; row-normalize(T)", formula_ref="transition composition",
)
def compose_transition(ctx: ScratchContext, first: str = "transition_a", second: str = "transition_b", save_as: str = "composed_transition") -> None:
    matrix = ctx[first] @ ctx[second]
    torch = __import__("torch")
    ctx[save_as] = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)


@block(
    id="materialize_transition",
    name="Materialize Transition",
    category="Transition",
    description="Materialize an explicit transition artifact for aligned indices without fitting or estimating it.",
    params={"artifact": {"type": "slot", "default": "transition"}, "indices": {"type": "slot", "default": "indices"}, "dtype": {"type": "value", "default": None}, "device": {"type": "slot", "default": "device"}, "save_as": {"type": "slot", "default": "transition_matrix"}},
    requires=("artifact",), provides=("save_as",), placement=("batch", "top", "epoch"), stage="setup", ui_group="⑥ 后验与权重",
)
def materialize_transition(ctx: ScratchContext, artifact: str = "transition", indices: str = "indices", dtype: Any = None, device: str = "device", save_as: str = "transition_matrix") -> None:
    torch, _ = _torch()
    source = ctx[artifact]
    target_device = ctx.get(device)
    kwargs = {}
    if target_device is not None:
        kwargs["device"] = target_device
    if dtype is not None and isinstance(dtype, torch.dtype):
        kwargs["dtype"] = dtype
    if hasattr(source, "transition_for"):
        if indices not in ctx:
            raise ValueError("materialize_transition requires aligned indices for transition artifacts")
        result = source.transition_for(None, ctx[indices], **kwargs)
    elif hasattr(source, "matrix") and callable(source.matrix):
        # Scratch-native trainable/artifact objects expose an explicit matrix
        # materializer.  Keep this operation limited to materialization; any
        # fitting or estimation belongs to the upstream estimator block.
        matrix_kwargs = {}
        if "dtype" in kwargs:
            matrix_kwargs["dtype"] = kwargs["dtype"]
        result = source.matrix(**matrix_kwargs)
        if target_device is not None:
            result = result.to(device=target_device)
    elif callable(source):
        # A transition revision may be exposed as a zero-argument callable
        # rather than a module with ``matrix``.  Calling it is the only
        # materialization step; fitting remains the upstream operation.
        result = source()
        result = torch.as_tensor(result, **kwargs)
    else:
        result = torch.as_tensor(source, **kwargs)
        if result.ndim not in (2, 3):
            raise ValueError("transition must have shape [C,C] or [N,C,C]")
    ctx[save_as] = result


@block(
    id="compose_revision_transition",
    name="Compose Revision Transition",
    category="Correction",
    description="Add a trainable transition revision and row-normalize the resulting per-sample matrix.",
    params={"transition": {"type": "slot", "default": "transition"}, "model": {"type": "slot", "default": "model"}, "indices": {"type": "slot", "default": "indices"}, "save_as": {"type": "slot", "default": "revised_transition"}},
    requires=("transition", "model", "indices"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 纠错风险",
    formula="T'=row_normalize(max(0,T(x)+T_revision))", formula_ref="trainable transition revision",
)
def compose_revision_transition(ctx: ScratchContext, transition: str = "transition", model: str = "model", indices: str = "indices", save_as: str = "revised_transition") -> None:
    import torch

    source = ctx[transition]
    if not hasattr(source, "transition_for"):
        raise TypeError("compose_revision_transition requires a per-sample transition artifact")
    revision = getattr(ctx[model], "T_revision", None)
    if revision is None or not hasattr(revision, "weight"):
        raise AttributeError("model must expose trainable T_revision.weight")
    base = source.transition_for(None, ctx[indices], device=revision.weight.device, dtype=revision.weight.dtype)
    matrices = (base + revision.weight.unsqueeze(0)).clamp_min(0.0)
    matrices = matrices / matrices.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(matrices.dtype).tiny)
    ctx[save_as] = matrices


@block(
    id="prior_kl",
    name="Prior KL Regularizer",
    category="Loss",
    description="Match the batch mean prediction to an explicit class prior.",
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "prior": {"type": "slot", "default": "prior"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("probabilities", "prior"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
)
def prior_kl(ctx: ScratchContext, probabilities: str = "probabilities", prior: str = "prior", save_as: str = "loss") -> None:
    values = ctx[probabilities].mean(dim=0).clamp_min(1e-12)
    target = ctx[prior].to(values).clamp_min(1e-12)
    ctx[save_as] = (target * (target / values).log()).sum()
