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
    id="constant",
    name="Constant",
    category="Tensor Operation",
    description="Publish an explicit scalar or structured constant as a formula operand.",
    params={"value": {"type": "value", "default": 0.0}, "save_as": {"type": "slot", "default": "constant"}},
    provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式",
    formula="z=c", formula_ref="explicit constant operand", formula_kind="primitive", formula_group="基础",
)
def constant(ctx: ScratchContext, value: Any = 0.0, save_as: str = "constant") -> None:
    """Publish a literal without inferring device, dtype, or a hidden input."""
    ctx[save_as] = value


@block(
    id="one_hot",
    name="One-hot Labels",
    category="Tensor Operation",
    description="Convert integer labels to one-hot vectors.",
    params={"labels": {"type": "slot", "default": "labels"}, "num_classes": {"type": "int", "default": 10, "min": 2}, "save_as": {"type": "slot", "default": "one_hot_labels"}},
    requires=("labels",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="one_hot(y, C)", formula_ref="one-hot label encoding", formula_kind="primitive", formula_group="概率 / Loss",
)
def one_hot(ctx: ScratchContext, labels: str = "labels", num_classes: int = 10, save_as: str = "one_hot_labels") -> None:
    torch, F = _torch()
    ctx[save_as] = F.one_hot(ctx[labels].long(), int(num_classes)).to(dtype=torch.float32)


@block(
    id="one_hot_like",
    name="One-hot Like",
    category="Tensor Operation",
    description="Convert labels to one-hot vectors using the class dimension of a reference tensor.",
    params={"labels": {"type": "slot", "default": "labels"}, "reference": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "one_hot_labels"}},
    requires=("labels", "reference"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="one_hot(y, shape(reference)[-1])", formula_ref="builtin/one_hot_like", formula_kind="composite", formula_group="概率 / Loss",
)
def one_hot_like(ctx: ScratchContext, labels: str = "labels", reference: str = "logits", save_as: str = "one_hot_labels") -> None:
    torch, F = _torch()
    classes = int(ctx[reference].shape[-1])
    ctx[save_as] = F.one_hot(ctx[labels].long(), classes).to(dtype=torch.float32)


@block(
    id="zeros_like",
    name="Zeros Like",
    category="Tensor Operation",
    description="Create a zero tensor matching an input tensor's shape and dtype.",
    params={"input": {"type": "slot", "default": "values"}, "requires_grad": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "zeros"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula_kind="primitive", formula_group="基础",
)
def zeros_like(ctx: ScratchContext, input: str = "values", requires_grad: bool = False, save_as: str = "zeros") -> None:
    ctx[save_as] = __import__("torch").zeros_like(ctx[input], requires_grad=bool(requires_grad))


@block(
    id="ones_like",
    name="Ones Like",
    category="Tensor Operation",
    description="Create a floating-point tensor of ones matching an input tensor's shape.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "ones"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_kind="primitive", formula_group="基础",
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
    formula="z=x⊙y", formula_ref="elementwise multiplication", formula_kind="primitive", formula_group="基础",
)
def elementwise_multiply(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "product") -> None:
    ctx[save_as] = ctx[left] * ctx[right]


@block(
    id="add",
    name="Add",
    category="Tensor Operation",
    description="Add two aligned tensor or scalar slots without reduction.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "sum"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式",
    formula="z=x+y", formula_ref="elementwise addition", formula_kind="primitive", formula_group="基础",
)
def add(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "sum") -> None:
    ctx[save_as] = ctx[left] + ctx[right]


@block(
    id="divide",
    name="Divide",
    category="Tensor Operation",
    description="Divide two tensor or scalar slots elementwise. No numerical floor is added; use Safe Divide when a denominator floor is part of the formula.",
    params={"numerator": {"type": "slot", "default": "numerator"}, "denominator": {"type": "slot", "default": "denominator"}, "save_as": {"type": "slot", "default": "quotient"}},
    requires=("numerator", "denominator"), provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式", formula_group="基础",
    formula="z=x/y", formula_ref="pure elementwise division", formula_kind="primitive",
)
def divide(ctx: ScratchContext, numerator: str = "numerator", denominator: str = "denominator", save_as: str = "quotient") -> None:
    """Perform pure elementwise division without silently clamping the denominator."""
    ctx[save_as] = ctx[numerator] / ctx[denominator]


@block(
    id="maximum",
    name="Maximum",
    category="Tensor Operation",
    description="Take the elementwise maximum of two aligned tensor or scalar slots.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "maximum"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式", formula_group="基础",
    formula="z=max(x,y)", formula_ref="elementwise maximum", formula_kind="primitive",
)
def maximum(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "maximum") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.maximum(torch.as_tensor(ctx[left]), torch.as_tensor(ctx[right]))


@block(
    id="minimum",
    name="Minimum",
    category="Tensor Operation",
    description="Take the elementwise minimum of two aligned tensor or scalar slots.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "minimum"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式", formula_group="基础",
    formula="z=min(x,y)", formula_ref="elementwise minimum", formula_kind="primitive",
)
def minimum(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "minimum") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.minimum(torch.as_tensor(ctx[left]), torch.as_tensor(ctx[right]))


@block(
    id="matrix_multiply",
    name="Matrix Multiply",
    category="Tensor Operation",
    description="Multiply matrices or batched matrices with the ordinary matrix product.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "product"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top", "epoch"), stage="train", ui_group="⑤ 损失公式", formula_group="基础",
    formula="Z=XY", formula_ref="matrix multiplication", formula_kind="primitive",
)
def matrix_multiply(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "product") -> None:
    ctx[save_as] = ctx[left] @ ctx[right]


@block(
    id="exp",
    name="Exponential",
    category="Tensor Operation",
    description="Apply the natural exponential elementwise.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "exponential"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="函数",
    formula="z=exp(x)", formula_ref="elementwise exponential", formula_kind="primitive",
)
def exp(ctx: ScratchContext, input: str = "input", save_as: str = "exponential") -> None:
    ctx[save_as] = ctx[input].exp()


@block(
    id="reciprocal",
    name="Reciprocal",
    category="Tensor Operation",
    description="Compute the elementwise reciprocal 1/x without an implicit floor.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "reciprocal"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=1/x", formula_ref="elementwise reciprocal", formula_kind="primitive", formula_group="基础",
)
def reciprocal(ctx: ScratchContext, input: str = "input", save_as: str = "reciprocal") -> None:
    ctx[save_as] = 1.0 / ctx[input]


@block(
    id="log",
    name="Natural Log",
    category="Tensor Operation",
    description="Apply the natural logarithm elementwise without an implicit clamp.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "log_values"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="函数",
    formula="z=log(x)", formula_ref="pure elementwise natural logarithm", formula_kind="primitive",
)
def log(ctx: ScratchContext, input: str = "input", save_as: str = "log_values") -> None:
    """Perform the pure logarithm; callers add clamp_min explicitly when needed."""
    ctx[save_as] = ctx[input].log()


@block(
    id="sqrt",
    name="Square Root",
    category="Tensor Operation",
    description="Apply the square root elementwise.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "square_root"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="函数",
    formula="z=sqrt(x)", formula_ref="elementwise square root", formula_kind="primitive",
)
def sqrt(ctx: ScratchContext, input: str = "input", save_as: str = "square_root") -> None:
    ctx[save_as] = ctx[input].sqrt()


@block(
    id="abs",
    name="Absolute Value",
    category="Tensor Operation",
    description="Take the elementwise absolute value.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "absolute"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="函数",
    formula="z=|x|", formula_ref="elementwise absolute value", formula_kind="primitive",
)
def abs(ctx: ScratchContext, input: str = "input", save_as: str = "absolute") -> None:
    ctx[save_as] = ctx[input].abs()


@block(
    id="sign",
    name="Sign",
    category="Tensor Operation",
    description="Return the elementwise sign of a tensor.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "sign"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="函数",
    formula="z=sign(x)", formula_ref="elementwise sign", formula_kind="primitive",
)
def sign(ctx: ScratchContext, input: str = "input", save_as: str = "sign") -> None:
    ctx[save_as] = ctx[input].sign()


@block(
    id="sum_values",
    name="Sum Values",
    category="Tensor Operation",
    description="Reduce all elements of a tensor to one scalar sum.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "sum"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="s=Σ_i x_i", formula_ref="builtin/sum_values", formula_kind="primitive", formula_group="归约",
)
def sum_values(ctx: ScratchContext, input: str = "values", save_as: str = "sum") -> None:
    ctx[save_as] = ctx[input].sum()


@block(
    id="sum_last_dimension",
    name="Sum Last Dimension",
    category="Tensor Operation",
    description="Reduce only the last tensor dimension, preserving leading sample dimensions.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "summed_values"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z_i=Σ_c x_{i,c}", formula_ref="explicit last-dimension reduction", formula_kind="primitive",
)
def sum_last_dimension(ctx: ScratchContext, input: str = "values", save_as: str = "summed_values") -> None:
    ctx[save_as] = ctx[input].sum(dim=-1)


def _optional_dim(dim: Any) -> int | None:
    if dim is None or (isinstance(dim, str) and dim.strip().lower() in {"", "none", "all"}):
        return None
    return int(dim)


@block(
    id="reduce_sum",
    name="Reduce Sum",
    category="Tensor Operation",
    description="Sum a tensor globally or along one explicit dimension.",
    params={"input": {"type": "slot", "default": "values"}, "dim": {"type": "value", "default": None}, "keepdim": {"type": "bool", "default": False}, "empty": {"type": "enum", "options": ["nan", "zero", "error"], "default": "nan"}, "save_as": {"type": "slot", "default": "sum"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=Σ_dim x", formula_ref="explicit sum reduction", formula_kind="primitive", formula_group="归约",
)
def reduce_sum(ctx: ScratchContext, input: str = "values", dim: Any = None, keepdim: bool = False, empty: str = "nan", save_as: str = "sum") -> None:
    axis = _optional_dim(dim)
    if ctx[input].numel() == 0 and str(empty) == "error":
        raise ValueError("reduce_sum received an empty tensor")
    ctx[save_as] = ctx[input].sum() if axis is None else ctx[input].sum(dim=axis, keepdim=bool(keepdim))


@block(
    id="reduce_mean",
    name="Reduce Mean",
    category="Tensor Operation",
    description="Average a tensor globally or along one explicit dimension.",
    params={"input": {"type": "slot", "default": "values"}, "dim": {"type": "value", "default": None}, "keepdim": {"type": "bool", "default": False}, "empty": {"type": "enum", "options": ["nan", "zero", "error"], "default": "nan"}, "save_as": {"type": "slot", "default": "mean"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=mean_dim(x)", formula_ref="explicit mean reduction", formula_kind="primitive", formula_group="归约",
)
def reduce_mean(ctx: ScratchContext, input: str = "values", dim: Any = None, keepdim: bool = False, empty: str = "nan", save_as: str = "mean") -> None:
    axis = _optional_dim(dim)
    if ctx[input].numel() == 0:
        if str(empty) == "error":
            raise ValueError("reduce_mean received an empty tensor")
        if str(empty) == "zero":
            ctx[save_as] = ctx[input].sum() * 0.0
            return
    ctx[save_as] = ctx[input].mean() if axis is None else ctx[input].mean(dim=axis, keepdim=bool(keepdim))


@block(
    id="numel",
    name="Element Count",
    category="Tensor Operation",
    description="Publish the number of elements in a tensor as a scalar on its device and dtype.",
    params={"input": {"type": "slot", "default": "values"}, "save_as": {"type": "slot", "default": "count"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="n=|x|", formula_ref="tensor element count", formula_kind="primitive", formula_group="归约",
)
def numel(ctx: ScratchContext, input: str = "values", save_as: str = "count") -> None:
    value = ctx[input]
    ctx[save_as] = __import__("torch").as_tensor(float(value.numel()), dtype=value.dtype, device=value.device)


@block(
    id="reduce_max",
    name="Reduce Max",
    category="Tensor Operation",
    description="Take the maximum globally or along one explicit dimension.",
    params={"input": {"type": "slot", "default": "values"}, "dim": {"type": "value", "default": None}, "keepdim": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "maximum"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=max_dim(x)", formula_ref="explicit maximum reduction", formula_kind="primitive", formula_group="归约",
)
def reduce_max(ctx: ScratchContext, input: str = "values", dim: Any = None, keepdim: bool = False, save_as: str = "maximum") -> None:
    axis = _optional_dim(dim)
    ctx[save_as] = ctx[input].amax() if axis is None else ctx[input].amax(dim=axis, keepdim=bool(keepdim))


@block(
    id="reduce_min",
    name="Reduce Min",
    category="Tensor Operation",
    description="Take the minimum globally or along one explicit dimension.",
    params={"input": {"type": "slot", "default": "values"}, "dim": {"type": "value", "default": None}, "keepdim": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "minimum"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=min_dim(x)", formula_ref="explicit minimum reduction", formula_kind="primitive", formula_group="归约",
)
def reduce_min(ctx: ScratchContext, input: str = "values", dim: Any = None, keepdim: bool = False, save_as: str = "minimum") -> None:
    axis = _optional_dim(dim)
    ctx[save_as] = ctx[input].amin() if axis is None else ctx[input].amin(dim=axis, keepdim=bool(keepdim))


@block(
    id="reshape_tensor",
    name="Reshape Tensor",
    category="Tensor Operation",
    description="Reshape a tensor to an explicit shape without changing its values.",
    params={"input": {"type": "slot", "default": "input"}, "shape": {"type": "value", "default": []}, "save_as": {"type": "slot", "default": "reshaped"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=reshape(x,shape)", formula_ref="explicit tensor reshape", formula_kind="primitive", formula_group="形状",
)
def reshape_tensor(ctx: ScratchContext, input: str = "input", shape: Any = (), save_as: str = "reshaped") -> None:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError("reshape_tensor requires a non-empty shape list")
    ctx[save_as] = ctx[input].reshape(tuple(int(value) for value in shape))


@block(
    id="unsqueeze",
    name="Unsqueeze",
    category="Tensor Operation",
    description="Insert a size-one dimension at an explicit axis.",
    params={"input": {"type": "slot", "default": "input"}, "dim": {"type": "int", "default": 0}, "save_as": {"type": "slot", "default": "expanded"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=unsqueeze(x,dim)", formula_ref="explicit dimension insertion", formula_kind="primitive", formula_group="形状",
)
def unsqueeze(ctx: ScratchContext, input: str = "input", dim: int = 0, save_as: str = "expanded") -> None:
    ctx[save_as] = ctx[input].unsqueeze(int(dim))


@block(
    id="squeeze",
    name="Squeeze",
    category="Tensor Operation",
    description="Remove a size-one dimension, or all size-one dimensions when dim is omitted.",
    params={"input": {"type": "slot", "default": "input"}, "dim": {"type": "value", "default": None}, "save_as": {"type": "slot", "default": "squeezed"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=squeeze(x,dim)", formula_ref="explicit dimension removal", formula_kind="primitive", formula_group="形状",
)
def squeeze(ctx: ScratchContext, input: str = "input", dim: Any = None, save_as: str = "squeezed") -> None:
    axis = _optional_dim(dim)
    ctx[save_as] = ctx[input].squeeze() if axis is None else ctx[input].squeeze(axis)


@block(
    id="transpose_dims",
    name="Transpose Dimensions",
    category="Tensor Operation",
    description="Swap two tensor dimensions explicitly.",
    params={"input": {"type": "slot", "default": "input"}, "dim0": {"type": "int", "default": 0}, "dim1": {"type": "int", "default": 1}, "save_as": {"type": "slot", "default": "transposed"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=transpose(x,dim0,dim1)", formula_ref="explicit dimension transpose", formula_kind="primitive", formula_group="形状",
)
def transpose_dims(ctx: ScratchContext, input: str = "input", dim0: int = 0, dim1: int = 1, save_as: str = "transposed") -> None:
    ctx[save_as] = ctx[input].transpose(int(dim0), int(dim1))


@block(
    id="compare",
    name="Compare",
    category="Tensor Operation",
    description="Compare two values elementwise using an explicit relation.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "operator": {"type": "enum", "options": ["lt", "le", "eq", "ne", "ge", "gt"], "default": "gt"}, "save_as": {"type": "slot", "default": "mask"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="m=(x op y)", formula_ref="elementwise comparison", formula_kind="primitive", formula_group="条件",
)
def compare(ctx: ScratchContext, left: str = "left", right: str = "right", operator: str = "gt", save_as: str = "mask") -> None:
    lhs, rhs = ctx[left], ctx[right]
    operations = {"lt": lhs.__lt__, "le": lhs.__le__, "eq": lhs.__eq__, "ne": lhs.__ne__, "ge": lhs.__ge__, "gt": lhs.__gt__}
    try:
        ctx[save_as] = operations[str(operator)](rhs)
    except KeyError as exc:
        raise ValueError(f"unsupported comparison operator: {operator}") from exc


@block(
    id="where",
    name="Where",
    category="Tensor Operation",
    description="Select true or false values elementwise using a boolean condition.",
    params={"condition": {"type": "slot", "default": "mask"}, "when_true": {"type": "slot", "default": "true_values"}, "when_false": {"type": "slot", "default": "false_values"}, "save_as": {"type": "slot", "default": "selected"}},
    requires=("condition", "when_true", "when_false"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=where(m,x,y)", formula_ref="elementwise conditional selection", formula_kind="primitive", formula_group="条件",
)
def where(ctx: ScratchContext, condition: str = "mask", when_true: str = "true_values", when_false: str = "false_values", save_as: str = "selected") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.where(ctx[condition].bool(), ctx[when_true], ctx[when_false])


@block(
    id="logical_not",
    name="Logical Not",
    category="Tensor Operation",
    description="Invert a boolean mask.",
    params={"input": {"type": "slot", "default": "mask"}, "save_as": {"type": "slot", "default": "inverted_mask"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=¬m", formula_ref="boolean inversion", formula_kind="primitive", formula_group="条件",
)
def logical_not(ctx: ScratchContext, input: str = "mask", save_as: str = "inverted_mask") -> None:
    ctx[save_as] = ~ctx[input].bool()


@block(
    id="logical_and",
    name="Logical And",
    category="Tensor Operation",
    description="Combine two boolean masks with logical conjunction.",
    params={"left": {"type": "slot", "default": "left_mask"}, "right": {"type": "slot", "default": "right_mask"}, "save_as": {"type": "slot", "default": "mask"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=m_1∧m_2", formula_ref="boolean conjunction", formula_kind="primitive", formula_group="条件",
)
def logical_and(ctx: ScratchContext, left: str = "left_mask", right: str = "right_mask", save_as: str = "mask") -> None:
    ctx[save_as] = ctx[left].bool() & ctx[right].bool()


@block(
    id="logical_or",
    name="Logical Or",
    category="Tensor Operation",
    description="Combine two boolean masks with logical disjunction.",
    params={"left": {"type": "slot", "default": "left_mask"}, "right": {"type": "slot", "default": "right_mask"}, "save_as": {"type": "slot", "default": "mask"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=m_1∨m_2", formula_ref="boolean disjunction", formula_kind="primitive", formula_group="条件",
)
def logical_or(ctx: ScratchContext, left: str = "left_mask", right: str = "right_mask", save_as: str = "mask") -> None:
    ctx[save_as] = ctx[left].bool() | ctx[right].bool()


@block(
    id="argmax",
    name="Argmax",
    category="Tensor Operation",
    description="Return the index of the largest value along a dimension.",
    params={"input": {"type": "slot", "default": "values"}, "dim": {"type": "int", "default": -1}, "keepdim": {"type": "bool", "default": False}, "save_as": {"type": "slot", "default": "indices"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=argmax_dim(x)", formula_ref="index of maximum value", formula_kind="primitive", formula_group="索引",
)
def argmax(ctx: ScratchContext, input: str = "values", dim: int = -1, keepdim: bool = False, save_as: str = "indices") -> None:
    ctx[save_as] = ctx[input].argmax(dim=int(dim), keepdim=bool(keepdim))


@block(
    id="gather",
    name="Gather",
    category="Tensor Operation",
    description="Gather values along an explicit dimension using integer indices.",
    params={"input": {"type": "slot", "default": "values"}, "indices": {"type": "slot", "default": "indices"}, "dim": {"type": "int", "default": 0}, "save_as": {"type": "slot", "default": "gathered"}},
    requires=("input", "indices"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=gather(x,i,dim)", formula_ref="dimension-indexed gather", formula_kind="primitive", formula_group="索引",
)
def gather(ctx: ScratchContext, input: str = "values", indices: str = "indices", dim: int = 0, save_as: str = "gathered") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.gather(ctx[input], int(dim), ctx[indices].long())


@block(
    id="select_class_column",
    name="Select Class Column",
    category="Tensor Operation",
    description="Select one fixed class column from an [N,C] tensor without changing its batch axis.",
    params={"input": {"type": "slot", "default": "values"}, "class_index": {"type": "int", "default": 0, "min": 0}, "save_as": {"type": "slot", "default": "selected_class"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z_i=x_{i,c}", formula_ref="fixed class-column selection", formula_kind="primitive", formula_group="索引",
)
def select_class_column(ctx: ScratchContext, input: str = "values", class_index: int = 0, save_as: str = "selected_class") -> None:
    values = ctx[input]
    if values.ndim < 2 or not 0 <= int(class_index) < int(values.shape[-1]):
        raise ValueError("select_class_column expects a valid class index for an [N,C] tensor")
    ctx[save_as] = values[..., int(class_index)]


@block(
    id="class_count",
    name="Class Count",
    category="Tensor Operation",
    description="Publish the class dimension of a reference tensor as an integer scalar.",
    params={"reference": {"type": "slot", "default": "logits"}, "save_as": {"type": "slot", "default": "num_classes"}},
    requires=("reference",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="C=shape(x)[-1]", formula_ref="reference class dimension", formula_kind="primitive", formula_group="形状",
)
def class_count(ctx: ScratchContext, reference: str = "logits", save_as: str = "num_classes") -> None:
    value = ctx[reference]
    if getattr(value, "ndim", 0) < 1:
        raise ValueError("class_count expects a tensor with a class dimension")
    ctx[save_as] = int(value.shape[-1])


@block(
    id="index_select",
    name="Index Select",
    category="Tensor Operation",
    description="Select rows or slices along an explicit dimension.",
    params={"input": {"type": "slot", "default": "values"}, "indices": {"type": "slot", "default": "indices"}, "dim": {"type": "int", "default": 0}, "save_as": {"type": "slot", "default": "selected"}},
    requires=("input", "indices"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑥ 样本选择",
    formula="z=index_select(x,i,dim)", formula_ref="dimension-indexed selection", formula_kind="primitive", formula_group="索引",
)
def index_select(ctx: ScratchContext, input: str = "values", indices: str = "indices", dim: int = 0, save_as: str = "selected") -> None:
    torch, _ = _torch()
    ctx[save_as] = torch.index_select(ctx[input], int(dim), ctx[indices].long())


@block(
    id="batched_matmul",
    name="Batched Matrix Multiply",
    category="Tensor Operation",
    description="Multiply matrices per batch, preserving the batch dimension.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "save_as": {"type": "slot", "default": "product"}},
    requires=("left", "right"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="Z_i=X_iY_i", formula_ref="batched matrix product", formula_kind="primitive", formula_group="线性代数",
)
def batched_matmul(ctx: ScratchContext, left: str = "left", right: str = "right", save_as: str = "product") -> None:
    torch, _ = _torch()
    lhs, rhs = ctx[left], ctx[right]
    if lhs.ndim == 2 and rhs.ndim == 3:
        ctx[save_as] = torch.bmm(lhs.unsqueeze(1), rhs).squeeze(1)
    else:
        ctx[save_as] = torch.matmul(lhs, rhs)


@block(
    id="detach",
    name="Detach Tensor",
    category="Tensor Operation",
    description="Publish a detached tensor without changing its values.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "detached"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_kind="primitive", formula_group="函数",
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
    requires=("minuend", "subtrahend"), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式", formula_group="基础",
    formula="z=x-y", formula_ref="elementwise subtraction", formula_kind="primitive",
)
def subtract(ctx: ScratchContext, minuend: str = "first", subtrahend: str = "second", save_as: str = "difference") -> None:
    ctx[save_as] = ctx[minuend] - ctx[subtrahend]


@block(
    id="negate",
    name="Negate",
    category="Tensor Operation",
    description="Multiply a tensor or scalar by -1 without reduction.",
    params={"input": {"type": "slot", "default": "input"}, "save_as": {"type": "slot", "default": "negated"}},
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="train", ui_group="⑤ 损失公式",
    formula="z=-x", formula_ref="explicit negation", formula_kind="primitive", formula_group="基础",
)
def negate(ctx: ScratchContext, input: str = "input", save_as: str = "negated") -> None:
    ctx[save_as] = -ctx[input]


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
    formula="MSE(x,y)=mean((x-y)^2)", formula_ref="builtin/mean_squared_error", formula_kind="composite", formula_group="概率 / Loss",
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
    formula="L=sum_i w_i L_i", formula_ref="builtin/weighted_sum", formula_kind="primitive", formula_group="概率 / Loss",
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
    formula="w=max(0,+/-g)", formula_ref="nonnegative projection", formula_kind="primitive",
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
    params={"weights": {"type": "slot", "default": "nonnegative_values"}, "epsilon": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "weights"}},
    requires=("weights",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑥ 后验与权重",
    formula="w=wbar/sum(wbar)", formula_ref="builtin/normalize_nonnegative_weights", formula_kind="composite",
)
def normalize_nonnegative_weights(ctx: ScratchContext, weights: str = "nonnegative_values", epsilon: float = 1e-12, save_as: str = "weights") -> None:
    values = ctx[weights]
    total = values.sum()
    denominator = total.clamp_min(float(epsilon))
    ctx[save_as] = values / denominator if bool(total > 0) else __import__("torch").zeros_like(values)


@block(
    id="negative_log",
    name="Negative Log",
    category="Tensor Operation",
    description="Apply an explicit finite floor and negative logarithm.",
    params={"input": {"type": "slot", "default": "probabilities"}, "minimum": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "negative_log_values"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula="z=-log(max(x,epsilon))", formula_ref="builtin/negative_log", formula_kind="composite", formula_group="函数",
)
def negative_log(ctx: ScratchContext, input: str = "probabilities", minimum: float = 1e-12, save_as: str = "negative_log_values") -> None:
    ctx[save_as] = -ctx[input].clamp_min(float(minimum)).log()


@block(
    id="safe_divide",
    name="Safe Divide",
    category="Tensor Operation",
    description="Divide two tensors after applying an explicit denominator floor.",
    params={"numerator": {"type": "slot", "default": "numerator"}, "denominator": {"type": "slot", "default": "denominator"}, "minimum": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "quotient"}},
    requires=("numerator", "denominator"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula="z=x/max(y,epsilon)", formula_ref="builtin/safe_divide", formula_kind="composite", formula_group="基础",
)
def safe_divide(ctx: ScratchContext, numerator: str = "numerator", denominator: str = "denominator", minimum: float = 1e-12, save_as: str = "quotient") -> None:
    denominator_value = ctx[denominator]
    if not hasattr(denominator_value, "clamp_min"):
        denominator_value = __import__("torch").as_tensor(denominator_value, dtype=ctx[numerator].dtype, device=ctx[numerator].device)
    ctx[save_as] = ctx[numerator] / denominator_value.clamp_min(float(minimum))


@block(
    id="weighted_blend",
    name="Weighted Blend",
    category="Tensor Operation",
    description="Blend two tensors using a sample-wise weight.",
    params={"left": {"type": "slot", "default": "left"}, "right": {"type": "slot", "default": "right"}, "weight": {"type": "slot", "default": "weight"}, "clamp_weight": {"type": "bool", "default": True}, "save_as": {"type": "slot", "default": "blended"}},
    requires=("left", "right", "weight"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="z=w·left+(1-w)·right", formula_ref="builtin/weighted_blend", formula_kind="composite",
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
    params={"input": {"type": "slot", "default": "targets"}, "temperature": {"type": "float", "default": 0.5, "min": 0.0001}, "epsilon": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "sharpened"}},
    requires=("input",), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="q'_c=q_c^(1/T)/Σ_j q_j^(1/T)", formula_ref="builtin/sharpen_distribution", formula_kind="composite",
)
def sharpen_distribution(ctx: ScratchContext, input: str = "targets", temperature: float = 0.5, epsilon: float = 1e-12, save_as: str = "sharpened") -> None:
    values = ctx[input].clamp_min(0).pow(1.0 / float(temperature))
    ctx[save_as] = values / values.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))


@block(
    id="soft_target_cross_entropy",
    name="Soft-target Cross Entropy",
    category="Loss",
    description="Compute cross entropy against probability targets, optionally restricted by a mask.",
    params={"logits": {"type": "slot", "default": "logits"}, "targets": {"type": "slot", "default": "targets"}, "mask": {"type": "value", "default": None}, "reduction": {"type": "enum", "options": ["per_sample", "mean"], "default": "mean"}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("logits", "targets"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="CE(q,z)=-Σ_c q_c log softmax(z)_c", formula_ref="builtin/soft_target_cross_entropy", formula_kind="composite",
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
    requires=("probabilities", "transition"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式", formula="q'=qT", formula_ref="transition probability application", formula_kind="special", formula_group="概率 / Loss",
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
    requires=("input",), provides=("save_as",), placement=("batch", "top"), stage="setup", ui_group="⑤ 损失公式", formula="z=x/max(sum(x),epsilon)", formula_ref="builtin/row_normalize", formula_kind="composite", formula_group="概率 / Loss",
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
    params={"first": {"type": "slot", "default": "transition_a"}, "second": {"type": "slot", "default": "transition_b"}, "epsilon": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "composed_transition"}},
    requires=("first", "second"), provides=("save_as",), placement=("top", "batch"), stage="setup", ui_group="⑥ 后验与权重",
    formula="T=T_1T_2; row-normalize(T)", formula_ref="builtin/compose_transition", formula_kind="composite",
)
def compose_transition(ctx: ScratchContext, first: str = "transition_a", second: str = "transition_b", epsilon: float = 1e-12, save_as: str = "composed_transition") -> None:
    matrix = ctx[first] @ ctx[second]
    torch = __import__("torch")
    ctx[save_as] = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))


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
    formula="T'=row_normalize(max(0,T(x)+T_revision))", formula_ref="trainable transition revision", formula_kind="special",
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
    params={"probabilities": {"type": "slot", "default": "probabilities"}, "prior": {"type": "slot", "default": "prior"}, "epsilon": {"type": "float", "default": 1e-12, "min": 0.0}, "save_as": {"type": "slot", "default": "loss"}},
    requires=("probabilities", "prior"), provides=("save_as",), placement=("batch",), stage="train", ui_group="⑤ 损失公式",
    formula="D=Σ_cπ_c log(π_c/mean_i p_{i,c})", formula_ref="builtin/prior_kl", formula_kind="composite", formula_group="概率 / Loss",
)
def prior_kl(ctx: ScratchContext, probabilities: str = "probabilities", prior: str = "prior", epsilon: float = 1e-12, save_as: str = "loss") -> None:
    values = ctx[probabilities].mean(dim=0).clamp_min(float(epsilon))
    target = ctx[prior].to(values).clamp_min(float(epsilon))
    ctx[save_as] = (target * (target / values).log()).sum()
