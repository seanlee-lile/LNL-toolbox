from __future__ import annotations

"""Strict configuration for binary asymmetric-RCN importance reweighting."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ImportanceReweightingConfig:
    """Validated first-version method configuration."""

    data: Mapping[str, Any]
    noise: Mapping[str, Any]
    posterior_stage: Mapping[str, Any]
    model: Mapping[str, Any]
    loss: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    loader: Mapping[str, Any]
    trainer: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    epochs: int
    seed: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ImportanceReweightingConfig":
        if not isinstance(value, Mapping):
            raise TypeError("importance reweighting configuration must be a mapping")
        method = value.get("method")
        method_name = (
            str(method.get("name", "")).strip().lower()
            if isinstance(method, Mapping)
            else str(method or "").strip().lower()
        )
        if method_name != "importance_reweighting":
            raise ValueError(
                "importance reweighting runner requires "
                "method: importance_reweighting"
            )
        if "num_classes" not in value or int(value["num_classes"]) != 2:
            raise ValueError("importance reweighting requires num_classes == 2")
        convention = str(value.get("label_convention", "")).strip().lower()
        if convention != "zero_one":
            raise ValueError(
                "importance reweighting requires label_convention: zero_one"
            )

        data = _mapping(value.get("data"), owner="data")
        data_name = str(data.get("name", "")).strip().lower()
        if data_name not in {
            "synthetic_binary_2d",
            "synthetic_binary_high_dim",
            "uci_statlog_heart",
        }:
            raise ValueError(
                "importance reweighting supports synthetic_binary_2d, "
                "synthetic_binary_high_dim, or uci_statlog_heart"
            )
        dimension = int(data.get("dimension", 2))
        if data_name == "synthetic_binary_2d" and dimension != 2:
            raise ValueError("synthetic_binary_2d requires data.dimension: 2")
        if data_name == "synthetic_binary_high_dim" and dimension <= 2:
            raise ValueError(
                "synthetic_binary_high_dim requires data.dimension greater than 2"
            )
        if data_name == "uci_statlog_heart":
            if dimension != 13:
                raise ValueError("uci_statlog_heart requires data.dimension: 13")
            source = str(data.get("path", "")).strip()
            if not source:
                raise ValueError("uci_statlog_heart requires data.path")
            data["path"] = str(Path(source))
            sha256 = str(data.get("sha256", "")).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError(
                    "uci_statlog_heart requires a 64-character data.sha256"
                )
            data["sha256"] = sha256
            expected_samples = int(data.get("expected_samples", 0))
            if expected_samples < 6:
                raise ValueError(
                    "uci_statlog_heart data.expected_samples must be at least 6"
                )
            data["expected_samples"] = expected_samples
            split = _mapping(data.get("split"), owner="data.split")
            validation_fraction = float(split.get("validation_fraction", 0.0))
            test_fraction = float(split.get("test_fraction", 0.0))
            if (
                not math.isfinite(validation_fraction)
                or not math.isfinite(test_fraction)
                or validation_fraction <= 0.0
                or test_fraction <= 0.0
                or validation_fraction + test_fraction >= 1.0
            ):
                raise ValueError(
                    "UCI validation/test fractions must be positive and sum to less than 1"
                )
            split.update({
                "seed": int(split.get("seed", value.get("seed", 1))),
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
            })
            data["split"] = split
            preprocessing = _mapping(
                data.get("preprocessing"), owner="data.preprocessing"
            )
            if str(preprocessing.get("format", "")).strip().lower() != "whitespace":
                raise ValueError(
                    "uci_statlog_heart requires whitespace preprocessing"
                )
            labels = tuple(str(item) for item in preprocessing.get("label_values", ()))
            if labels != ("1", "2"):
                raise ValueError(
                    "uci_statlog_heart label_values must map 1 -> 0 and 2 -> 1"
                )
            preprocessing["label_values"] = labels
            data["preprocessing"] = preprocessing
            schema = _mapping(data.get("schema"), owner="data.schema")
            feature_columns = schema.get("feature_columns")
            if (
                not isinstance(feature_columns, list)
                or len(feature_columns) != 13
                or any(not str(name).strip() for name in feature_columns)
                or len({str(name) for name in feature_columns}) != 13
                or str(schema.get("target", "")).strip() != "heart_disease"
            ):
                raise ValueError(
                    "uci_statlog_heart schema requires 13 unique features and "
                    "target: heart_disease"
                )
            data["schema"] = schema
        data["dimension"] = dimension
        if data_name != "uci_statlog_heart":
            for name in ("train_size", "validation_size", "test_size"):
                size = int(data.get(name, 0))
                if size < 2 or size % 2:
                    raise ValueError(f"data.{name} must be a positive even integer")

        noise = _mapping(value.get("noise"), owner="noise")
        if str(noise.get("name", "")).strip().lower() != "binary_asymmetric_rcn":
            raise ValueError(
                "importance reweighting requires "
                "noise.name: binary_asymmetric_rcn"
            )
        positive = float(noise.get("rho_positive", -1.0))
        negative = float(noise.get("rho_negative", -1.0))
        if not 0.0 <= positive < 1.0 or not 0.0 <= negative < 1.0:
            raise ValueError("binary RCN noise rates must be within [0, 1)")
        if positive + negative >= 1.0:
            raise ValueError("rho_positive + rho_negative must be less than 1")
        if str(noise.get("validation_targets", "noisy")).strip().lower() != "noisy":
            raise ValueError(
                "importance reweighting checkpoint selection requires "
                "noise.validation_targets: noisy"
            )

        posterior = _mapping(
            value.get("posterior_stage"), owner="posterior_stage"
        )
        posterior_name = str(posterior.get("name", "")).strip().lower()
        if posterior_name not in {"kde", "kliep"}:
            raise ValueError(
                "posterior_stage.name must be kde or kliep"
            )
        posterior["name"] = posterior_name
        bandwidth = float(posterior.get("bandwidth", 0.0))
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("posterior_stage.bandwidth must be positive")
        posterior["bandwidth"] = bandwidth
        if posterior_name == "kliep":
            max_centers = int(posterior.get("max_centers", 0))
            max_iterations = int(posterior.get("max_iterations", 0))
            learning_rate = float(posterior.get("learning_rate", 0.0))
            tolerance = float(posterior.get("tolerance", 0.0))
            epsilon = float(posterior.get("epsilon", 0.0))
            if max_centers <= 0:
                raise ValueError("posterior_stage.max_centers must be positive")
            if max_iterations <= 0:
                raise ValueError(
                    "posterior_stage.max_iterations must be positive"
                )
            for name, parameter in (
                ("learning_rate", learning_rate),
                ("tolerance", tolerance),
                ("epsilon", epsilon),
            ):
                if not math.isfinite(parameter) or parameter <= 0.0:
                    raise ValueError(
                        f"posterior_stage.{name} must be finite and positive"
                    )
            posterior.update({
                "max_centers": max_centers,
                "max_iterations": max_iterations,
                "learning_rate": learning_rate,
                "tolerance": tolerance,
                "epsilon": epsilon,
                "seed": int(posterior.get("seed", value.get("seed", 1))),
            })
        rate_name = str(
            posterior.get("rate_estimator", "paper_raw_min")
        ).strip().lower()
        if rate_name != "paper_raw_min":
            raise ValueError(
                "importance reweighting first version requires "
                "rate_estimator: paper_raw_min"
            )

        model = _mapping(value.get("model", {"name": "linear"}), owner="model")
        if str(model.get("name", "linear")).strip().lower() != "linear":
            raise ValueError(
                "importance reweighting first version only supports model: linear"
            )
        if int(model.get("in_features", dimension)) != dimension:
            raise ValueError(
                "importance reweighting model in_features must match data.dimension"
            )
        model["in_features"] = dimension
        if int(model.get("num_classes", 2)) != 2:
            raise ValueError("importance reweighting model output must contain 2 classes")

        loss = _mapping(value.get("loss", {"name": "ce"}), owner="loss")
        if str(loss.get("name", "")).strip().lower() != "ce":
            raise ValueError(
                "importance reweighting first version requires loss.name: ce"
            )
        optimizer = _mapping(value.get("optimizer"), owner="optimizer")
        scheduler = _mapping(
            value.get("scheduler", {"name": "none"}), owner="scheduler"
        )
        loader = _mapping(value.get("loader"), owner="loader")
        if int(loader.get("batch_size", 0)) <= 0:
            raise ValueError("loader.batch_size must be positive")
        trainer = _mapping(value.get("trainer"), owner="trainer")
        epochs = int(trainer.get("epochs", 0))
        if epochs <= 0:
            raise ValueError("trainer.epochs must be positive")
        diagnostics = _mapping(
            value.get("diagnostics", {"max_gradient_norm": 1.0e6}),
            owner="diagnostics",
        )
        max_gradient_norm = float(diagnostics.get("max_gradient_norm", 1.0e6))
        if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0.0:
            raise ValueError("diagnostics.max_gradient_norm must be positive")
        diagnostics["max_gradient_norm"] = max_gradient_norm
        evaluation = _mapping(
            value.get("evaluation", {"selection_split": "validation"}),
            owner="evaluation",
        )
        if str(
            evaluation.get("selection_split", "validation")
        ).strip().lower() != "validation":
            raise ValueError(
                "importance reweighting checkpoint selection must use validation"
            )
        return cls(
            data=data,
            noise=noise,
            posterior_stage=posterior,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            trainer=trainer,
            diagnostics=diagnostics,
            epochs=epochs,
            seed=int(value.get("seed", 1)),
        )
