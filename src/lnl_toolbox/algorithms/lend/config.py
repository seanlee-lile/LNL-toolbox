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
    dilution_tolerance: float = 1e-5
    dilution_max_steps: int = 1000
    implementation_variant: str = "paper_equation"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "LENDConfig":
        method = config.get("method", "")
        if isinstance(method, Mapping):
            method = method.get("name", "")
        if str(method).strip().lower() != "lend":
            raise ValueError("LEND requires method: lend")
        values = _mapping(config.get("lend"), "lend configuration")
        variant = str(values.get("implementation_variant", "paper_equation")).strip().lower()
        if variant not in {"paper_equation", "author_release"}:
            raise ValueError("LEND implementation_variant must be paper_equation or author_release")
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
        alpha = float(dilution["alpha"])
        policy = str(dilution["policy"]).strip().lower()
        if policy not in {"fixed_steps", "until_converged"}:
            raise ValueError("lend.dilution.policy must be fixed_steps or until_converged")
        required = {"alpha", "policy", "steps"} if policy == "fixed_steps" else {"alpha", "policy", "tolerance", "max_steps"}
        if not required <= set(dilution) or set(dilution) - {"alpha", "policy", "steps", "tolerance", "max_steps"}:
            raise ValueError(f"lend.dilution {policy} requires {sorted(required)}")
        steps = int(dilution.get("steps", 10))
        tolerance = float(dilution.get("tolerance", 1e-5))
        max_steps = dilution.get("max_steps", 1000)
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("lend.dilution.alpha must be finite and in (0,1)")
        if policy == "fixed_steps" and steps < 1:
            raise ValueError("LEND fixed_steps requires steps >= 1")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("lend.dilution.tolerance must be finite and positive")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("lend.dilution.max_steps must be a positive integer")

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
        if reduction not in {"batch_mean", "sum"}:
            raise ValueError("LEND reduction must be batch_mean or sum")
        if empty not in {"skip_update", "zero_gradient"}:
            raise ValueError("LEND empty_batch must be skip_update or zero_gradient")
        training = _mapping(values.get("training"), "lend.training")
        if "epochs" not in training or set(training) - {"epochs", "early_stopping"} or int(training["epochs"]) <= 0:
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
            "tiny_cnn", "cnn8", "resnet14", "resnet32", "resnet18",
            "resnet34", "resnet50", "resnet101", "preact_resnet18",
        }
        if str(model.get("name", "")).strip().lower() not in feature_models:
            raise ValueError("LEND model must be a supported feature-aware model")
        return cls(k, gamma, metric, normalize_features, zero_degree_policy,
                   alpha, policy, steps, beta, first, rule, reduction, empty,
                   int(training["epochs"]), tolerance, max_steps, variant)

    def implementation_metadata(self) -> dict[str, Any]:
        """Effective semantics; inherited paper controls are inactive for author release.

        Author constants follow code/main.py:56-69,135-150. Only k/beta and
        the outer training protocol are shared experimental controls.
        """
        if self.implementation_variant == "author_release":
            return {
                "variant": self.implementation_variant, "k": self.k,
                "metric": "squared_euclidean", "gamma": 3,
                "normalize_features": False, "self_policy": "topk_then_clear_diagonal",
                "orientation": "AAT", "propagation": "resolvent_solve",
                "lambda": 0.99, "softmax_scale": 1.0,
                "nan_policy": "whole_batch_input_then_softmax",
                "beta": self.beta, "history": "pre_propagation",
                "history_start_epoch_zero_based": 10,
                "weighting": "noisy_class_probability", "reduction": "batch_mean",
                "inactive_config_fields": [
                    "lend.graph.gamma", "lend.graph.metric", "lend.graph.normalize_features",
                    "lend.dilution", "lend.history.first_observation", "lend.selection",
                ],
            }
        return {
            "variant": self.implementation_variant, "k": self.k,
            "metric": self.metric, "gamma": self.gamma,
            "normalize_features": self.normalize_features, "orientation": "ATA",
            "propagation": self.dilution_policy, "alpha": self.alpha,
            "beta": self.beta, "history": "post_propagation_ema",
            "weighting": self.selection_rule, "reduction": self.reduction,
        }

    def identity(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("epochs")
        return value


__all__ = ["LENDConfig"]
