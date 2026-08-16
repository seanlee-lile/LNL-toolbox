from __future__ import annotations

"""Strict configuration for the paper-oriented LEND workflow."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class LENDConfig:
    k: int
    gamma: float
    metric: str
    normalize_features: bool
    zero_degree_policy: str
    alpha: float
    dilution_policy: str
    dilution_steps: int
    beta: float
    first_observation: str
    selection_rule: str
    reduction: str
    empty_batch: str
    epochs: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "LENDConfig":
        method = config.get("method", "")
        if isinstance(method, Mapping):
            method = method.get("name", "")
        if str(method).strip().lower() != "lend":
            raise ValueError("LEND requires method: lend")
        values = _mapping(config.get("lend"), "lend configuration")
        graph = _mapping(values.get("graph"), "lend.graph")
        required_graph = {"k", "gamma", "metric", "normalize_features"}
        if set(graph) != required_graph:
            raise ValueError("lend.graph requires k/gamma/metric/normalize_features")
        k = int(graph["k"])
        gamma = float(graph["gamma"])
        metric = str(graph["metric"]).strip().lower()
        normalize_features = graph["normalize_features"]
        # Zero degree is a valid boundary of a sparse directed kNN graph.  It
        # has one fixed normalization meaning rather than a user-tunable policy.
        zero_degree_policy = "zero_inverse"
        if k < 1:
            raise ValueError("lend.graph.k must be at least one")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("lend.graph.gamma must be finite and positive")
        if metric not in {"inner_product", "cosine", "euclidean"}:
            raise ValueError("lend.graph.metric must be inner_product, cosine, or euclidean")
        if type(normalize_features) is not bool:
            raise TypeError("lend.graph.normalize_features must be boolean")
        dilution = _mapping(values.get("dilution"), "lend.dilution")
        if set(dilution) != {"alpha", "policy", "steps"}:
            raise ValueError("lend.dilution requires alpha/policy/steps")
        alpha = float(dilution["alpha"])
        policy = str(dilution["policy"]).strip().lower()
        steps = int(dilution["steps"])
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("lend.dilution.alpha must be finite and in (0,1)")
        if policy != "fixed_steps" or steps < 1:
            raise ValueError("LEND first version requires fixed_steps with steps >= 1")

        history = _mapping(values.get("history"), "lend.history")
        if set(history) != {"beta", "first_observation"}:
            raise ValueError("lend.history requires beta/first_observation")
        beta = float(history["beta"])
        first = str(history["first_observation"]).strip().lower()
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("lend.history.beta must be finite and in [0,1]")
        if first != "current":
            raise ValueError("LEND first observation must be current")

        selection = _mapping(values.get("selection"), "lend.selection")
        if set(selection) != {"rule", "reduction", "empty_batch"}:
            raise ValueError("lend.selection requires rule/reduction/empty_batch")
        rule = str(selection["rule"]).strip().lower()
        reduction = str(selection["reduction"]).strip().lower()
        empty = str(selection["empty_batch"]).strip().lower()
        if rule != "noisy_equals_diluted_argmax":
            raise ValueError("unsupported LEND selection rule")
        if reduction != "batch_mean":
            raise ValueError("LEND requires batch_mean reduction")
        if empty != "skip_update":
            raise ValueError("LEND first version requires empty_batch: skip_update")
        training = _mapping(values.get("training"), "lend.training")
        if set(training) != {"epochs"} or int(training["epochs"]) <= 0:
            raise ValueError("lend.training requires positive epochs")
        if "selector" in config or "target_provider" in config or "weight_provider" in config:
            raise ValueError("LEND cannot combine with generic sample-treatment providers")
        if dict(config.get("loss", {"name": "ce"})) != {"name": "ce"}:
            raise ValueError("LEND first version requires per-sample CE")
        loader = _mapping(config.get("loader"), "loader configuration")
        batch_size = int(loader.get("batch_size", 0))
        if batch_size <= k:
            raise ValueError("LEND loader.batch_size must be greater than graph.k")
        drop_last = loader.get("drop_last", False)
        if type(drop_last) is not bool:
            raise TypeError("LEND loader.drop_last must be boolean")
        data = _mapping(config.get("data"), "data configuration")
        sample_limit = data.get("max_train_samples")
        if sample_limit is not None and not drop_last:
            remainder = int(sample_limit) % batch_size
            if 0 < remainder <= k:
                raise ValueError(
                    "LEND final partial batch must satisfy B > k or use drop_last: true"
                )
        model = _mapping(config.get("model"), "model configuration")
        feature_models = {
            "tiny_cnn", "cifar_cnn8", "resnet14", "resnet32", "resnet18",
            "resnet34", "resnet50", "resnet101", "preact_resnet18",
        }
        if str(model.get("name", "")).strip().lower() not in feature_models:
            raise ValueError("LEND model must be a supported feature-aware model")
        return cls(k, gamma, metric, normalize_features, zero_degree_policy,
                   alpha, policy, steps, beta, first, rule, reduction, empty,
                   int(training["epochs"]))

    def identity(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("epochs")
        return value


__all__ = ["LENDConfig"]
