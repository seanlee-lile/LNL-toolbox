from __future__ import annotations

"""Runner-owned, display-neutral experiment planning contracts."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PlanField:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class RunPlan:
    runner: str
    method: str
    training_budget: str
    fields: tuple[PlanField, ...]


def method_name(config: Mapping[str, Any], runner: str) -> str:
    method = config.get("method", "")
    if isinstance(method, Mapping):
        method = method.get("name", "")
    if str(method).strip():
        return str(method).strip().lower()
    algorithm = config.get("algorithm", {}) or {}
    if isinstance(algorithm, Mapping) and str(algorithm.get("name", "")).strip():
        return str(algorithm["name"]).strip().lower()
    loss = config.get("loss", {}) or {}
    if isinstance(loss, Mapping) and str(loss.get("name", "")).strip():
        return str(loss["name"]).strip().lower()
    return runner


def value_at(config: Mapping[str, Any], path: tuple[str, ...], default: Any = "-") -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def generic_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    budget = value_at(config, budget_path, "runner-defined") if budget_path else "runner-defined"
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    trainer = config.get("trainer", {}) or {}
    evaluation = config.get("evaluation", {}) or {}
    return RunPlan(
        runner=runner,
        method=method_name(config, runner),
        training_budget=str(budget),
        fields=(
            PlanField("Dataset", str(data.get("name", "unknown"))),
            PlanField("Model", str(model.get("name", "runner default"))),
            PlanField("Device", str(trainer.get("device", "auto"))),
            PlanField("Selection split", str(evaluation.get("selection_split", "validation"))),
        ),
    )


def supervised_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    base = generic_plan(config, runner, budget_path)
    loss = config.get("loss", {}) or {}
    selector = config.get("selector", {}) or {}
    return RunPlan(
        base.runner,
        base.method,
        f"{base.training_budget} epochs",
        base.fields
        + (
            PlanField("Loss", str(loss.get("name", "ce"))),
            PlanField("Selector", str(selector.get("name", "all"))),
        ),
    )


def coteaching_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    base = generic_plan(config, runner, budget_path)
    values = config.get("coteaching", {}) or {}
    return RunPlan(
        base.runner,
        base.method,
        f"{base.training_budget} epochs",
        base.fields
        + (
            PlanField("Models", "2"),
            PlanField("Gradual epochs", str(values.get("gradual_epochs", "-"))),
            PlanField("Noise rate", str((config.get("noise", {}) or {}).get("rate", "-"))),
        ),
    )


def dividemix_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    values = config.get("dividemix", {}) or {}
    warmup = (values.get("warmup", {}) or {}).get("epochs", "-")
    main = (values.get("training", {}) or {}).get("epochs", "-")
    total = warmup + main if isinstance(warmup, int) and isinstance(main, int) else "-"
    gmm = values.get("gmm", {}) or {}
    mixmatch = values.get("mixmatch", {}) or {}
    base = generic_plan(config, runner, budget_path)
    return RunPlan(
        base.runner,
        base.method,
        f"{warmup}/{main}/{total} (warmup/main/total)",
        base.fields
        + (
            PlanField("Models", "2"),
            PlanField("GMM threshold", str(gmm.get("threshold", "-"))),
            PlanField("MixMatch temperature", str(mixmatch.get("temperature", "-"))),
            PlanField("MixUp alpha", str(mixmatch.get("mixup_alpha", "-"))),
        ),
    )


def upm_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    base = generic_plan(config, runner, budget_path)
    values = config.get("upm", {}) or {}
    stage1 = values.get("stage1", {}) or {}
    main = values.get("main", {}) or {}
    psi = values.get("psi", {}) or {}
    eta = values.get("confusing_probability", {}) or {}
    return RunPlan(
        base.runner,
        base.method,
        f"{stage1.get('epochs', '-')}/{main.get('epochs', '-')} (stage1/main)",
        base.fields
        + (
            PlanField("UPM psi source", str(psi.get("source", "-"))),
            PlanField("UPM eta initial value", str(eta.get("initial_value", "-"))),
            PlanField("UPM eta update start epoch", str(eta.get("update_start_epoch", "-"))),
            PlanField("UPM eta update interval", str(eta.get("update_interval_epochs", "-"))),
        ),
    )


def dld_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    base = generic_plan(config, runner, budget_path)
    values = config.get("dld", {}) or {}
    pre = values.get("precorrection", {}) or {}
    inference = values.get("inference", {}) or {}
    fidelity = values.get("fidelity", {}) or {}
    return RunPlan(
        base.runner,
        base.method,
        f"{value_at(config, ('dld', 'diffusion', 'epochs'))} (diffusion)",
        base.fields
        + (
            PlanField("DLD neighbors", f"K={pre.get('k_neighbors', '-') }"),
            PlanField("DLD neighbor metric", str(fidelity.get("neighbor_metric", "-"))),
            PlanField("DLD artifact", "dld_precorrection.npz"),
            PlanField("DLD fidelity", str(fidelity.get("name", "-"))),
            PlanField("DLD inference steps", str(inference.get("steps", "-"))),
        ),
    )


def lend_plan(config: Mapping[str, Any], runner: str, budget_path: tuple[str, ...] | None) -> RunPlan:
    base = generic_plan(config, runner, budget_path)
    values = config.get("lend", {}) or {}
    graph = values.get("graph", {}) or {}
    dilution = values.get("dilution", {}) or {}
    history = values.get("history", {}) or {}
    selection = values.get("selection", {}) or {}
    return RunPlan(
        base.runner,
        base.method,
        f"{value_at(config, ('lend', 'training', 'epochs'))} (LEND)",
        base.fields
        + (
            PlanField(
                "LEND graph",
                f"k={graph.get('k', '-')} gamma={graph.get('gamma', '-')} "
                f"{graph.get('metric', '-')} normalize_features={graph.get('normalize_features', '-')}",
            ),
            PlanField(
                "LEND dilution",
                f"alpha={dilution.get('alpha', '-')} {dilution.get('policy', '-')} "
                f"steps={dilution.get('steps', '-')}",
            ),
            PlanField("LEND history", f"beta={history.get('beta', '-')}"),
            PlanField("LEND selection rule", str(selection.get("rule", "-"))),
            PlanField("LEND reduction", str(selection.get("reduction", "-"))),
            PlanField("LEND empty batch", str(selection.get("empty_batch", "-"))),
        ),
    )


__all__ = [
    "PlanField",
    "RunPlan",
    "coteaching_plan",
    "dividemix_plan",
    "dld_plan",
    "generic_plan",
    "method_name",
    "lend_plan",
    "supervised_plan",
    "upm_plan",
    "value_at",
]
