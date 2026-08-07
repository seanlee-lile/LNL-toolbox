from __future__ import annotations

"""Method-owned loss history and peer co-divide construction."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from lnl_toolbox.estimators import (
    DivideMixGMMCleanProbabilityEstimator,
    DivideMixGMMLossInput,
)

from .config import DivideMixConfig


@dataclass(frozen=True)
class CoDivideResult:
    sample_indices: Tensor
    normalized_loss_a: Tensor
    normalized_loss_b: Tensor
    clean_probability_a: Tensor
    clean_probability_b: Tensor
    labeled_for_a: Tensor
    labeled_for_b: Tensor
    metrics: Mapping[str, float]


def _artifact_hash(arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        value = torch.as_tensor(arrays[name]).detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii")); digest.update(str(value.shape).encode("ascii")); digest.update(value.tobytes())
    return digest.hexdigest()


def save_co_divide_artifact(path: str | Path, result: CoDivideResult, *, epoch: int, metadata: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    arrays = {
        "sample_indices": result.sample_indices,
        "normalized_loss_a": result.normalized_loss_a,
        "normalized_loss_b": result.normalized_loss_b,
        "clean_probability_a": result.clean_probability_a,
        "clean_probability_b": result.clean_probability_b,
        "labeled_for_a": result.labeled_for_a.to(torch.uint8),
        "labeled_for_b": result.labeled_for_b.to(torch.uint8),
    }
    full_metadata = {**dict(metadata), "method": "dividemix", "epoch": int(epoch), "producer_a_consumer": "b", "producer_b_consumer": "a"}
    content_hash = _artifact_hash(arrays, full_metadata)
    import numpy as np
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: torch.as_tensor(value).detach().cpu().numpy() for key, value in arrays.items()}, metadata_json=np.asarray(json.dumps(full_metadata, sort_keys=True)), artifact_hash=np.asarray(content_hash))
    loaded, loaded_metadata, loaded_hash = load_co_divide_artifact(temporary)
    if loaded_hash != content_hash or loaded_metadata != full_metadata:
        temporary.unlink(missing_ok=True)
        raise ValueError("DivideMix co-divide artifact reload validation failed")
    temporary.replace(destination)
    return content_hash


def load_co_divide_artifact(path: str | Path) -> tuple[CoDivideResult, dict[str, Any], str]:
    import numpy as np
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"sample_indices", "normalized_loss_a", "normalized_loss_b", "clean_probability_a", "clean_probability_b", "labeled_for_a", "labeled_for_b", "metadata_json", "artifact_hash"}
        if set(payload.files) != required:
            raise ValueError("DivideMix co-divide artifact fields are invalid")
        metadata = json.loads(str(payload["metadata_json"].item()))
        content_hash = str(payload["artifact_hash"].item())
        arrays = {key: payload[key].copy() for key in required - {"metadata_json", "artifact_hash"}}
    if metadata.get("method") != "dividemix" or metadata.get("producer_a_consumer") != "b" or metadata.get("producer_b_consumer") != "a":
        raise ValueError("DivideMix co-divide producer/consumer identity is invalid")
    indices = torch.from_numpy(arrays["sample_indices"]).to(torch.long)
    normalized_a = torch.from_numpy(arrays["normalized_loss_a"]).to(torch.float64)
    normalized_b = torch.from_numpy(arrays["normalized_loss_b"]).to(torch.float64)
    probability_a = torch.from_numpy(arrays["clean_probability_a"]).to(torch.float64)
    probability_b = torch.from_numpy(arrays["clean_probability_b"]).to(torch.float64)
    mask_a = torch.from_numpy(arrays["labeled_for_a"]).to(torch.bool)
    mask_b = torch.from_numpy(arrays["labeled_for_b"]).to(torch.bool)
    if indices.ndim != 1 or torch.unique(indices).numel() != indices.numel() or any(value.shape != indices.shape for value in (normalized_a, normalized_b, probability_a, probability_b, mask_a, mask_b)):
        raise ValueError("DivideMix co-divide artifact arrays are not sample aligned")
    for probability in (probability_a, probability_b):
        if not bool(torch.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError("DivideMix co-divide probabilities are invalid")
    if _artifact_hash(arrays, metadata) != content_hash:
        raise ValueError("DivideMix co-divide artifact hash mismatch")
    result = CoDivideResult(indices, normalized_a, normalized_b, probability_a, probability_b, mask_a, mask_b, {})
    return result, metadata, content_hash


def normalized_loss(losses: Tensor) -> Tensor:
    if losses.ndim != 1 or not torch.is_floating_point(losses) or not bool(torch.isfinite(losses).all().item()):
        raise ValueError("DivideMix losses must be finite floating point [N]")
    minimum, maximum = losses.min(), losses.max()
    if not bool((maximum > minimum).item()):
        raise ValueError("DivideMix losses must have positive range")
    return ((losses - minimum) / (maximum - minimum)).detach()


def append_loss_history(history: list[dict[str, Any]], indices: Tensor, losses: Tensor, window: int) -> None:
    history.append({"indices": indices.detach().cpu(), "losses": normalized_loss(losses).detach().cpu()})
    del history[:-window]


def history_input(history: Iterable[Mapping[str, Any]], current_indices: Tensor, *, use_average: bool) -> Tensor:
    entries = list(history)
    if not entries:
        raise ValueError("DivideMix loss history is empty")
    sorted_current, order = torch.sort(current_indices.detach().cpu().to(torch.long))
    aligned: list[Tensor] = []
    selected = entries if use_average else entries[-1:]
    for entry in selected:
        indices = torch.as_tensor(entry["indices"], dtype=torch.long)
        losses = torch.as_tensor(entry["losses"], dtype=torch.float64)
        sorted_indices, positions = torch.sort(indices)
        if not torch.equal(sorted_indices, sorted_current):
            raise ValueError("DivideMix loss history stable indices changed")
        aligned.append(losses[positions])
    canonical = torch.stack(aligned).mean(0)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    return canonical[inverse].to(current_indices.device)


def build_co_divide(
    indices: Tensor,
    history_a: list[dict[str, Any]],
    history_b: list[dict[str, Any]],
    config: DivideMixConfig,
    noise_rate: float,
) -> CoDivideResult:
    use_average = (
        config.gmm.history_name == "official_auto"
        and abs(float(noise_rate) - config.gmm.high_noise_rate) <= 1e-12
    )
    estimator = DivideMixGMMCleanProbabilityEstimator(
        random_seed=config.gmm.random_seed,
        max_iter=config.gmm.max_iterations,
        tolerance=config.gmm.tolerance,
        covariance_regularization=config.gmm.covariance_regularization,
        minimum_mean_separation=config.gmm.minimum_mean_separation,
    )
    input_a = normalized_loss(history_input(history_a, indices, use_average=use_average))
    input_b = normalized_loss(history_input(history_b, indices, use_average=use_average))
    result_a = estimator.estimate(DivideMixGMMLossInput(input_a, indices))
    result_b = estimator.estimate(DivideMixGMMLossInput(input_b, indices))
    # Cross ownership: B selects data consumed by A; A selects data consumed by B.
    labeled_for_a = result_b.scores >= config.gmm.threshold
    labeled_for_b = result_a.scores >= config.gmm.threshold
    for owner, mask in (("A", labeled_for_a), ("B", labeled_for_b)):
        selected = int(mask.sum().item())
        if selected == 0 or selected == mask.numel():
            raise ValueError(f"DivideMix split for network {owner} must contain labeled and unlabeled samples")
    metrics = {
        "clean_probability_mean_a": float(result_a.scores.mean().item()),
        "clean_probability_mean_b": float(result_b.scores.mean().item()),
        "labeled_ratio_for_a": float(labeled_for_a.double().mean().item()),
        "labeled_ratio_for_b": float(labeled_for_b.double().mean().item()),
        "loss_history_depth": float(len(history_a) if use_average else 1),
    }
    for peer, input_values, result in (("a", input_a, result_a), ("b", input_b, result_b)):
        for name, value in result.metrics.items(): metrics[f"gmm_{peer}_{name}"] = float(value)
        clean_mean = float(result.metrics["clean_component_mean"])
        noisy_mean = float(result.metrics["noisy_component_mean"])
        clean_weight, noisy_weight = result.scores, 1.0 - result.scores
        metrics[f"gmm_{peer}_clean_component_covariance"] = float((clean_weight * (input_values - clean_mean).square()).sum().item() / clean_weight.sum().item() + config.gmm.covariance_regularization)
        metrics[f"gmm_{peer}_noisy_component_covariance"] = float((noisy_weight * (input_values - noisy_mean).square()).sum().item() / noisy_weight.sum().item() + config.gmm.covariance_regularization)
    return CoDivideResult(indices.detach(), input_a.detach(), input_b.detach(), result_a.scores, result_b.scores, labeled_for_a.detach(), labeled_for_b.detach(), metrics)
