from __future__ import annotations

"""Strict configuration contract for the standalone VolMinNet method."""

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return dict(value)


def _positive(value: Any, *, owner: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner} must be finite and positive")
    return result


@dataclass(frozen=True)
class VolMinNetConfig:
    num_classes: int
    fidelity: str
    parameterization: str
    convention: str
    normalization_axis: str
    initialization: str
    lambda_volume: float
    classifier_optimizer: Mapping[str, Any]
    transition_optimizer: Mapping[str, Any]
    classifier_scheduler: Mapping[str, Any]
    transition_scheduler: Mapping[str, Any]
    checkpoint_metric: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VolMinNetConfig":
        method = value.get("method", "")
        if isinstance(method, Mapping):
            method = method.get("name", "")
        method = str(method).strip().lower()
        if method not in {"volmin", "volminnet"}:
            raise ValueError("VolMin requires method: volmin (or legacy alias volminnet)")
        execution = _mapping(value.get("execution"), owner="execution")
        runner = str(execution.get("runner", "")).strip().lower()
        if runner not in {"volmin", "volminnet"}:
            raise ValueError(
                "VolMin requires execution.runner: volmin "
                "(or legacy alias volminnet)"
            )
        data = _mapping(value.get("data"), owner="data")
        dataset = str(data.get("name", "")).strip().lower()
        expected_classes = {"cifar10": 10, "cifar100": 100}
        if dataset not in expected_classes:
            raise ValueError("VolMinNet first version supports CIFAR-10 and CIFAR-100")
        num_classes = int(data.get("num_classes", expected_classes[dataset]))
        if num_classes != expected_classes[dataset] or num_classes < 3:
            raise ValueError("VolMinNet class count must match the CIFAR dataset and be >= 3")
        trainer = _mapping(value.get("trainer"), owner="trainer")
        if int(trainer.get("epochs", 0)) <= 0:
            raise ValueError("trainer.epochs must be positive")
        noise = _mapping(value.get("noise"), owner="noise")
        if str(noise.get("validation_targets", "")).strip().lower() != "noisy":
            raise ValueError("VolMinNet checkpoint selection requires noisy validation targets")

        config = _mapping(value.get("volminnet"), owner="volminnet")
        fidelity = str(config.get("fidelity", "")).strip().lower()
        if fidelity != "paper_positive_logdet":
            raise ValueError("VolMinNet fidelity must be paper_positive_logdet")
        transition = _mapping(config.get("transition"), owner="volminnet.transition")
        parameterization = str(transition.get("parameterization", "")).strip().lower()
        convention = str(transition.get("convention", "")).strip().lower()
        normalization_axis = str(transition.get("normalization_axis", "")).strip().lower()
        initialization_config = _mapping(
            transition.get("initialization"), owner="volminnet.transition.initialization"
        )
        initialization = str(initialization_config.get("mode", "")).strip().lower()
        if parameterization != "fixed_diagonal_sigmoid_offdiag":
            raise ValueError("VolMinNet requires the paper transition parameterization")
        if convention != "clean_to_noisy_row" or normalization_axis != "row":
            raise ValueError("VolMinNet requires clean_to_noisy_row row normalization")
        if initialization != "paper":
            raise ValueError("VolMinNet requires paper transition initialization")

        objective = _mapping(config.get("objective"), owner="volminnet.objective")
        if str(objective.get("classification", "")).strip().lower() != "noisy_nll":
            raise ValueError("VolMinNet classification objective must be noisy_nll")
        volume = _mapping(objective.get("volume"), owner="volminnet.objective.volume")
        if str(volume.get("mode", "")).strip().lower() != "positive_logdet":
            raise ValueError("VolMinNet volume mode must be positive_logdet")
        lambda_volume = _positive(volume.get("coefficient"), owner="volume coefficient")

        optimizer = _mapping(config.get("optimizer"), owner="volminnet.optimizer")
        classifier_optimizer = _mapping(optimizer.get("classifier"), owner="classifier optimizer")
        transition_optimizer = _mapping(optimizer.get("transition"), owner="transition optimizer")
        scheduler = _mapping(config.get("scheduler", {}), owner="volminnet.scheduler")
        classifier_scheduler = _mapping(
            scheduler.get("classifier", {"name": "none"}), owner="classifier scheduler"
        )
        transition_scheduler = _mapping(
            scheduler.get("transition", {"name": "none"}), owner="transition scheduler"
        )
        for owner, scheduler_config in (
            ("classifier", classifier_scheduler),
            ("transition", transition_scheduler),
        ):
            name = str(scheduler_config.get("name", "none")).strip().lower()
            if name not in {"none", "multistep", "cosine"}:
                raise ValueError(f"VolMinNet {owner} scheduler is unsupported")
            if name == "cosine" and "t_max" not in scheduler_config:
                raise ValueError(
                    f"VolMinNet {owner} cosine scheduler requires explicit t_max"
                )
        selection = _mapping(config.get("checkpoint_selection"), owner="checkpoint_selection")
        if (
            str(selection.get("split", "")).strip().lower() != "noisy_validation"
            or str(selection.get("metric", "")).strip().lower() != "loss"
            or str(selection.get("mode", "")).strip().lower() != "min"
        ):
            raise ValueError("VolMinNet best checkpoint must minimize noisy-validation loss")
        return cls(
            num_classes=num_classes,
            fidelity=fidelity,
            parameterization=parameterization,
            convention=convention,
            normalization_axis=normalization_axis,
            initialization=initialization,
            lambda_volume=lambda_volume,
            classifier_optimizer=classifier_optimizer,
            transition_optimizer=transition_optimizer,
            classifier_scheduler=classifier_scheduler,
            transition_scheduler=transition_scheduler,
            checkpoint_metric="noisy_validation_loss",
        )
