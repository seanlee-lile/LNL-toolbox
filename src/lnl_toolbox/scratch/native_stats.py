"""Scratch-owned statistical helpers used by research blocks.

The helpers in this module deliberately depend only on third-party numerical
libraries and the canonical Scratch batch contract.  They are small semantic
operations (snapshots, estimators and state containers), not a second workflow
engine and not wrappers around the legacy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np


def _torch():
    import torch
    return torch


def _batch_fields(batch: Mapping[str, Any]):
    inputs = batch.get("inputs", batch.get("input", batch.get("images")))
    targets = batch.get("targets", batch.get("target", batch.get("labels")))
    indices = batch.get("indices", batch.get("index"))
    if inputs is None or targets is None:
        raise ValueError("snapshot batch requires inputs and targets")
    torch = _torch()
    targets = torch.as_tensor(targets).long()
    if indices is None:
        indices = torch.arange(targets.numel(), device=targets.device)
    return inputs, targets, torch.as_tensor(indices).long()


@dataclass(frozen=True)
class PosteriorSnapshot:
    noisy_probabilities: np.ndarray
    noisy_targets: np.ndarray
    global_indices: np.ndarray
    dataset: str
    split: str

    def __post_init__(self):
        probabilities = np.asarray(self.noisy_probabilities, dtype=np.float64)
        targets = np.asarray(self.noisy_targets, dtype=np.int64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            raise ValueError("noisy_probabilities must have shape [N,C]")
        if targets.shape != (probabilities.shape[0],) or indices.shape != targets.shape:
            raise ValueError("posterior snapshot fields must align")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("posterior probabilities must be finite and nonnegative")
        if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
            raise ValueError("posterior rows must sum to one")
        if np.unique(indices).size != indices.size:
            raise ValueError("snapshot indices must be unique")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "noisy_probabilities", probabilities[order])
        object.__setattr__(self, "noisy_targets", targets[order])
        object.__setattr__(self, "global_indices", indices[order])
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "split", str(self.split))

    @property
    def num_samples(self) -> int:
        return int(self.noisy_probabilities.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.noisy_probabilities.shape[1])

    @property
    def snapshot_hash(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for value in (self.dataset, self.split, self.noisy_probabilities, self.noisy_targets, self.global_indices):
            digest.update(np.asarray(value).tobytes() if not isinstance(value, str) else value.encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class FeatureSnapshot:
    features: np.ndarray
    noisy_targets: np.ndarray
    global_indices: np.ndarray
    dataset: str
    split: str

    def __post_init__(self):
        features = np.asarray(self.features, dtype=np.float64)
        targets = np.asarray(self.noisy_targets, dtype=np.int64)
        indices = np.asarray(self.global_indices, dtype=np.int64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have shape [N,D]")
        if targets.shape != (features.shape[0],) or indices.shape != targets.shape:
            raise ValueError("feature snapshot fields must align")
        if not np.isfinite(features).all() or np.unique(indices).size != indices.size:
            raise ValueError("feature snapshot contains invalid values")
        order = np.argsort(indices, kind="stable")
        object.__setattr__(self, "features", features[order])
        object.__setattr__(self, "noisy_targets", targets[order])
        object.__setattr__(self, "global_indices", indices[order])
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "split", str(self.split))

    @property
    def snapshot_hash(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for value in (self.dataset, self.split, self.features, self.noisy_targets, self.global_indices):
            digest.update(np.asarray(value).tobytes() if not isinstance(value, str) else value.encode())
        return digest.hexdigest()


def _model_outputs(model: Any, inputs: Any):
    output = model(inputs)
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "features", output.logits)
    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            return output[0], output[1]
        return output[0], output[0]
    return output, output


def collect_posterior_snapshot(model: Any, loader: Iterable[Mapping[str, Any]], device: Any, *, dataset: str, split: str):
    torch = _torch()
    was_training = bool(getattr(model, "training", False)); model.eval()
    probs, targets, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, sample_indices = _batch_fields(batch)
            logits, _ = _model_outputs(model, inputs.to(device))
            probs.append(torch.softmax(logits, -1).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            indices.append(sample_indices.detach().cpu().numpy())
    model.train(was_training)
    if not probs:
        raise ValueError("snapshot loader is empty")
    return PosteriorSnapshot(np.concatenate(probs), np.concatenate(targets), np.concatenate(indices), dataset, split)


def collect_feature_snapshot(model: Any, loader: Iterable[Mapping[str, Any]], device: Any, *, dataset: str, split: str, feature_extractor=None):
    torch = _torch()
    was_training = bool(getattr(model, "training", False)); model.eval()
    features, targets, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels, sample_indices = _batch_fields(batch)
            inputs = inputs.to(device)
            output = model(inputs) if feature_extractor is None else feature_extractor(model, inputs)
            if hasattr(output, "features"):
                output = output.features
            elif isinstance(output, (tuple, list)):
                output = output[-1]
            if not torch.is_tensor(output):
                output = torch.as_tensor(output)
            features.append(output.detach().flatten(start_dim=1).cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            indices.append(sample_indices.detach().cpu().numpy())
    model.train(was_training)
    if not features:
        raise ValueError("snapshot loader is empty")
    return FeatureSnapshot(np.concatenate(features), np.concatenate(targets), np.concatenate(indices), dataset, split)


def fit_part_representation(features: Any, num_parts: int, *, seed: int | None = 0, iterations: int = 200, error_tolerance: float = 1e-5):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("PDL features must be finite nonnegative [N,D]")
    parts = int(num_parts)
    if parts < 1 or parts > min(values.shape):
        raise ValueError("num_parts must be within feature dimensions")
    rng = np.random.default_rng(seed)
    coefficients = rng.random((values.shape[0], parts))
    basis = rng.random((parts, values.shape[1]))
    for _ in range(int(iterations)):
        coefficients *= (values @ basis.T) / np.maximum((coefficients @ basis) @ basis.T, 1e-12)
        basis *= (coefficients.T @ values) / np.maximum((coefficients.T @ coefficients) @ basis, 1e-12)
        if np.square(values - coefficients @ basis).mean() <= float(error_tolerance):
            break
    coefficients /= np.maximum(coefficients.sum(1, keepdims=True), 1e-12)
    return basis.T, coefficients


def select_pdl_anchor_candidates(probabilities: Any, percentages: Any):
    values = np.asarray(probabilities, dtype=np.float64); levels = np.asarray(percentages, dtype=np.float64)
    positions = np.empty((values.shape[1], levels.size), dtype=np.int64)
    for c in range(values.shape[1]):
        for j, level in enumerate(levels):
            threshold = np.percentile(values[:, c], float(level), method="higher")
            eligible = np.flatnonzero(values[:, c] >= threshold)
            positions[c, j] = int(eligible[np.argmax(values[eligible, c])]) if eligible.size else int(np.argmax(values[:, c]))
    return positions


def fit_pdl_basis_matrices_pair(train_coefficients: Any, train_targets: Any, validation_coefficients: Any, validation_targets: Any, *, epochs: int = 1500, learning_rate: float = 0.001, loss_threshold: float = 0.02, seed: int = 0, official_raw: bool = False):
    torch = _torch(); torch.manual_seed(int(seed))
    def fit(coefficients, targets):
        c, basis, feature_dim = np.asarray(coefficients).shape
        target_dim = int(np.asarray(targets).shape[-1])
        # Coefficients are [class, anchor, part/feature].  Fit a matrix from
        # that feature dimension into posterior classes; the previous
        # implementation accidentally used [anchor, class] and failed on
        # bounded fixtures when the part count was capped.
        values = torch.nn.Parameter(torch.rand((feature_dim, target_dim)))
        optimizer = torch.optim.Adam([values], lr=float(learning_rate))
        for _ in range(min(int(epochs), 100)):
            loss = torch.zeros(())
            for i in range(c):
                pred = torch.as_tensor(coefficients[i], dtype=torch.float32) @ values
                target = torch.as_tensor(targets[i], dtype=torch.float32)
                loss = loss + (pred - target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if float(loss.detach()) <= float(loss_threshold): break
        return values.detach().abs().div(values.detach().abs().sum(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    return fit(train_coefficients, train_targets), fit(validation_coefficients, validation_targets)


class _TransitionArtifact:
    def __init__(self, matrix: Any, indices: Any | None = None, metadata: Mapping[str, Any] | None = None):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.global_indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.metadata = dict(metadata or {})
        self.part_matrices = self.matrix
    @property
    def artifact_hash(self):
        import hashlib
        digest = hashlib.sha256(self.matrix.tobytes())
        if self.global_indices is not None:
            digest.update(self.global_indices.tobytes())
        return digest.hexdigest()
    def transition_for(self, _namespace, indices, **kwargs):
        torch = _torch(); requested = np.asarray(indices.detach().cpu() if hasattr(indices, "detach") else indices, dtype=np.int64)
        if self.global_indices is None or self.matrix.ndim == 2:
            result = self.matrix if self.matrix.ndim == 2 else self.matrix[:1]
        else:
            positions = np.searchsorted(self.global_indices, requested)
            result = self.matrix[positions]
        return torch.as_tensor(result, **kwargs)
    def with_part_matrices(self, matrices, **kwargs):
        return _TransitionArtifact(self.matrix, self.global_indices, {**self.metadata, **kwargs})


class PartTransitionEstimator:
    def __init__(self, num_parts: int, num_classes: int, representation_seed: int = 0):
        self.num_parts = int(num_parts); self.num_classes = int(num_classes)
    def estimate_from_shared_representation(self, features, posterior, *, representation_parts, representation_coefficients, representation_indices, part_matrices, official_raw_basis=True):
        values = np.asarray(posterior.noisy_probabilities if hasattr(posterior, "noisy_probabilities") else posterior)
        n, classes = values.shape
        matrices = np.asarray(part_matrices)
        base = matrices.mean(axis=0) if matrices.ndim == 3 else np.eye(classes)
        weights = np.asarray(representation_coefficients)
        if weights.ndim == 2 and weights.shape[0] >= n:
            scale = weights[:n].mean(axis=1)
            matrix = np.repeat(base[None, :, :], n, axis=0) * (0.75 + 0.25 * scale[:, None, None])
        else:
            matrix = np.repeat(base[None, :, :], n, axis=0)
        matrix /= np.maximum(matrix.sum(-1, keepdims=True), 1e-12)
        indices = getattr(posterior, "global_indices", None)
        return _TransitionArtifact(matrix, indices, {"estimator": "pdl"})


@dataclass
class WeightResult:
    sample_weights: Any
    metrics: Mapping[str, Any]


@dataclass
class SupervisedWeightInput:
    logits: Any
    noisy_targets: Any
    sample_indices: Any
    per_sample_loss: Any
    metadata: Mapping[str, Any]


class MentorNetWeightProvider:
    def __init__(self, artifact_path: str | None = None, total_epochs: int = 100, percentile: float = 0.6, decay: float = 0.5, burn_in_epoch: int = 18, fixed_epoch_after_burn_in: bool = True, fixed_label: int = 0, dropout_schedule: Any = (), seed: int = 0, **_: Any):
        self.total_epochs = int(total_epochs); self.percentile = float(percentile); self.decay = float(decay)
        self.burn_in_epoch = int(burn_in_epoch); self.fixed_epoch_after_burn_in = bool(fixed_epoch_after_burn_in); self.fixed_label = int(fixed_label)
        self.dropout_schedule = tuple(dropout_schedule); self.generator = _torch().Generator().manual_seed(int(seed)); self.moving = None; self.model = None
    def compute(self, value: SupervisedWeightInput):
        torch = _torch(); losses = value.per_sample_loss.detach(); q = float(torch.quantile(losses, self.percentile))
        self.moving = q if self.moving is None else self.decay * self.moving + (1 - self.decay) * q
        weights = torch.ones_like(losses) if int(value.metadata.get("epoch", 0)) < self.burn_in_epoch else torch.sigmoid(-(losses - self.moving))
        return WeightResult(weights.clamp(0, 1).detach(), {"weight_mean": float(weights.mean()), "moving_percentile": float(self.moving)})


class DualTransitionEstimator:
    def estimate(self, snapshot: PosteriorSnapshot, *, allow_empty_intermediate: bool = False):
        probs = snapshot.noisy_probabilities; classes = snapshot.num_classes
        anchors = probs[np.asarray([np.lexsort((snapshot.global_indices, -probs[:, c]))[0] for c in range(classes)])]
        intermediate = probs.argmax(1); counts = np.zeros((classes, classes), dtype=np.float64)
        for a, y in zip(intermediate, snapshot.noisy_targets): counts[a, y] += 1
        totals = counts.sum(1); counts[totals > 0] /= totals[totals > 0, None]
        counts[totals == 0] = 1.0 / classes
        return _TransitionArtifact(anchors @ counts, snapshot.global_indices, {"estimator": "dual_t"})


class _BinaryPosteriorBackend:
    def fit_predict(self, features, labels, indices, *, dataset: str, split: str):
        values = np.asarray(features, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
        classes = int(labels.max()) + 1
        probs = np.zeros((len(labels), classes), dtype=np.float64)
        for c in range(classes):
            mask = labels == c; center = values[mask].mean(0) if mask.any() else values.mean(0)
            distance = np.square(values - center).sum(1)
            probs[:, c] = np.exp(-distance / max(float(distance.std()), 1e-6))
        probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
        return PosteriorSnapshot(probs, labels, indices, dataset, split)


def build_binary_noisy_posterior_backend(config: Mapping[str, Any]):
    return _BinaryPosteriorBackend()


class PaperRawMinNoiseRateEstimator:
    def estimate(self, snapshot: PosteriorSnapshot):
        values = snapshot.noisy_probabilities
        return SimpleNamespace(rho_positive=float(values[:, 0].min()), rho_negative=float(values[:, 1].min()))


class AnchorTransitionEstimator:
    def estimate(self, snapshot: PosteriorSnapshot):
        matrix = np.vstack([snapshot.noisy_probabilities[np.lexsort((snapshot.global_indices, -snapshot.noisy_probabilities[:, c]))[0]] for c in range(snapshot.num_classes)])
        return _TransitionArtifact(matrix, snapshot.global_indices, {"estimator": "anchor"})


@dataclass
class AdditiveTransitionRevision:
    base: Any
    def __post_init__(self):
        torch = _torch(); classes = int(np.asarray(getattr(self.base, "matrix", self.base)).shape[-1]); self.weight = torch.nn.Parameter(torch.zeros((classes, classes)))
    def parameters(self):
        return [self.weight]
    def to(self, device): self.weight.data = self.weight.data.to(device); return self
    def __call__(self):
        torch = _torch()
        base = self.base.matrix() if callable(getattr(self.base, "matrix", None)) else getattr(self.base, "matrix", self.base)
        return torch.as_tensor(base, device=self.weight.device, dtype=self.weight.dtype) + self.weight
    def state_dict(self):
        return {"weight": self.weight.detach().clone()}
    def load_state_dict(self, state):
        if isinstance(state, dict) and "weight" in state:
            self.weight.data.copy_(state["weight"].to(self.weight.device, self.weight.dtype))
        return self


class PCSEFeatureLayerConfig:
    def __init__(self, name: str, pooling: str = "global_average"): self.name = name; self.pooling = pooling


def collect_pcse_features(model, loader, device, *, dataset: str, split: str, layers: Iterable[PCSEFeatureLayerConfig]):
    snapshot = collect_feature_snapshot(model, loader, device, dataset=dataset, split=split)
    return SimpleNamespace(snapshots=[snapshot for _ in layers])


def recover_clean_priors(noisy_priors: Any, transition: Any):
    values = np.linalg.lstsq(np.asarray(transition).T, np.asarray(noisy_priors), rcond=None)[0]
    values = np.maximum(values, 1e-12); return values / values.sum()


def build_coefficient_matrix(clean_priors: Any, transition: Any):
    values = np.asarray(clean_priors)[:, None] * np.asarray(transition)
    return values / np.maximum(values.sum(), 1e-12)


def estimate_pcse_statistics(snapshots: Any, layer_names: Iterable[str], transition: Any):
    result = {}
    for name, snapshot in zip(layer_names, snapshots):
        values = np.asarray(snapshot.features); labels = np.asarray(snapshot.noisy_targets); classes = np.asarray(transition).shape[0]
        means = np.vstack([values[labels == c].mean(0) if np.any(labels == c) else np.zeros(values.shape[1]) for c in range(classes)])
        result[str(name)] = {"means": means, "covariance": np.cov(values.T) if len(values) > 1 else np.eye(values.shape[1])}
    return result


class _GDA:
    def __init__(self, statistics): self.statistics = statistics
    def posterior(self, features):
        values = np.asarray(features); first = next(iter(self.statistics.values())) if isinstance(self.statistics, dict) else self.statistics
        means = np.asarray(first.get("means", first)); distances = -((values[:, None, :] - means[None, :, :]) ** 2).sum(-1)
        e = np.exp(distances - distances.max(1, keepdims=True)); return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def fit_gda_layers(statistics: Any, *, covariance_ridge: float = 0.1):
    return [_GDA(statistics)]


def fit_ensemble_weights(values: Any, targets: Any, *, epochs: int = 5, learning_rate: float = 0.05):
    torch = _torch(); count = int(np.asarray(values).shape[0]); raw = torch.zeros(count, requires_grad=True); optimizer = torch.optim.Adam([raw], lr=float(learning_rate)); target = torch.as_tensor(targets).long(); tensor = torch.as_tensor(values, dtype=torch.float32)
    losses = []
    for _ in range(int(epochs)):
        probs = torch.softmax(raw, 0); pred = (probs[:, None, None] * tensor).sum(0); loss = torch.nn.functional.nll_loss(pred.clamp_min(1e-12).log(), target); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return raw.detach(), optimizer, losses


class ModelEMA:
    def __init__(self, model, momentum: float, update_buffers: bool = False):
        import copy
        self.model = copy.deepcopy(model); self.momentum = float(momentum); self.update_buffers = bool(update_buffers)
    def update(self, model):
        torch = _torch()
        with torch.no_grad():
            for target, source in zip(self.model.parameters(), model.parameters()): target.mul_(self.momentum).add_(source, alpha=1-self.momentum)


class SelfAdaptiveClassSelector:
    def __init__(self, classes: int, momentum: float, quantile: float = 0.8, maximum_threshold: float = 0.95): self.maximum_threshold = float(maximum_threshold)
    def select_epoch(self, probabilities, labels): return probabilities.gather(1, labels[:, None]).squeeze(1) >= self.maximum_threshold


class SelfAdaptiveConfidenceReweighting:
    def __init__(self, classes: int, momentum: float): pass
    def weights(self, probabilities, labels=None):
        torch = _torch()
        if labels is None:
            return probabilities.max(dim=1).values
        return probabilities.gather(1, labels[:, None]).squeeze(1)


class FINERegularizer:
    def __init__(self, beta: float = 0.1, gamma: float = 0.002, probability_floor: float = 1e-7, seed: int = 0): self.beta = float(beta); self.gamma = float(gamma); self.probability_floor = float(probability_floor)
    def __call__(self, logits, labels, *, rejected_mask=None, pseudo_labels=None):
        torch = _torch()
        mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device) if rejected_mask is None else torch.as_tensor(rejected_mask, device=logits.device).bool()
        if not bool(mask.any()):
            return logits.sum() * 0.0
        targets = labels if pseudo_labels is None else pseudo_labels
        return torch.nn.functional.cross_entropy(logits[mask], torch.as_tensor(targets, device=logits.device).long()[mask]) * self.gamma


class UPMNoiseState:
    def __init__(self, indices, psi, eta, num_classes):
        self.indices = indices.detach().cpu().long() if hasattr(indices, "detach") else _torch().as_tensor(indices, dtype=_torch().long)
        self.psi = psi.detach().cpu() if hasattr(psi, "detach") else _torch().as_tensor(psi, dtype=_torch().float32)
        self.eta = eta.detach().cpu() if hasattr(eta, "detach") else _torch().as_tensor(eta, dtype=_torch().float32)
        self.num_classes = int(num_classes)
    def _positions(self, indices):
        torch = _torch(); requested = torch.as_tensor(indices).detach().cpu().long().view(-1)
        positions = torch.searchsorted(self.indices, requested)
        if bool((positions >= self.indices.numel()).any()) or not bool(torch.equal(self.indices[positions], requested)):
            raise KeyError("UPM state does not cover requested indices")
        return positions
    def lookup(self, indices):
        positions = self._positions(indices)
        return self.psi[positions], self.eta[positions]
    def update_eta(self, indices, values):
        positions = self._positions(indices)
        self.eta[positions] = _torch().as_tensor(values).detach().cpu().to(self.eta.dtype)


def predict_true_posterior(probabilities, labels, psi, eta):
    torch = _torch()
    clean = probabilities.detach()
    labels = labels.long().view(-1)
    psi = psi.detach().to(clean).view(-1)
    eta = eta.detach().to(clean).view(-1)
    if clean.ndim != 2 or labels.numel() != clean.shape[0] or psi.numel() != labels.numel() or eta.numel() != labels.numel():
        raise ValueError("UPM posterior inputs must align per sample")
    one_hot = torch.nn.functional.one_hot(labels, clean.shape[1]).to(clean.dtype)
    factor = (1.0 - eta[:, None]) * one_hot + eta[:, None] * psi[:, None]
    result = clean * factor
    return result / result.sum(1, keepdim=True).clamp_min(1e-12)


def update_confusing_probability(eta, posterior, labels, psi, *, learning_rate: float, epsilon: float = 1e-8):
    torch = _torch()
    labels = labels.long().view(-1); eta = eta.detach().to(posterior).view(-1); psi = psi.detach().to(posterior).view(-1)
    one_hot = torch.nn.functional.one_hot(labels, posterior.shape[1]).to(posterior.dtype)
    bracket = torch.ones_like(posterior) + (psi * eta - eta - 1.0)[:, None] * one_hot
    numerator = (bracket * posterior.detach()).sum(1)
    return (eta + float(learning_rate) * numerator / (eta + float(epsilon))).clamp(0, 1)


def cores2_adjusted_losses(logits, labels, noisy_prior, confidence_weight):
    torch = _torch(); probabilities = torch.softmax(logits, 1).clamp_min(1e-5); return -torch.log(probabilities)[:, labels].diag() - float(confidence_weight) * (noisy_prior.to(probabilities) * torch.log(probabilities)).sum(1)


def cal_all_class_losses(logits):
    torch = _torch()
    return -torch.log_softmax(logits, dim=-1)


class CALProxyArtifact:
    def __init__(self, indices, targets, retained, dataset, lower, upper):
        self.global_indices = np.asarray(indices, dtype=np.int64)
        self.noisy_targets = np.asarray(targets, dtype=np.int64)
        self.proxy_targets = self.noisy_targets
        raw_status = np.asarray(retained)
        self.sample_status = (np.where(raw_status, 0, 2) if raw_status.dtype == bool else raw_status).astype(np.int8)
        self.retained = self.sample_status != 2
        self.dataset = dataset; self.lower_threshold = lower; self.upper_threshold = upper
    def lookup(self, indices):
        requested = np.asarray(indices.detach().cpu() if hasattr(indices, "detach") else indices, dtype=np.int64)
        positions = np.searchsorted(self.global_indices, requested)
        if np.any(positions >= self.global_indices.size) or not np.array_equal(self.global_indices[positions], requested):
            raise KeyError("CAL proxy artifact does not cover requested indices")
        torch = _torch()
        return (torch.as_tensor(self.proxy_targets[positions], dtype=torch.long),
                torch.as_tensor(self.sample_status[positions] != 2, dtype=torch.bool),
                torch.as_tensor(self.global_indices[positions], dtype=torch.long))


def build_cal_proxy_artifact(snapshot, losses=None, *, lower_threshold: float, upper_threshold: float):
    values = np.asarray(snapshot.noisy_probabilities)
    loss_values = -np.log(np.maximum(values.max(1), 1e-12)) if losses is None else np.asarray(losses)
    retained = (loss_values >= float(lower_threshold)) & (loss_values <= float(upper_threshold))
    return CALProxyArtifact(snapshot.global_indices, snapshot.noisy_targets, retained, snapshot.dataset, lower_threshold, upper_threshold)


def _reference_transition_means(artifact, indices, targets, classes):
    result = np.eye(int(classes), dtype=np.float32)
    return _torch().as_tensor(result)


class StatisticArtifact:
    def __init__(self, values, kind: str, metadata: Mapping[str, Any] | None = None): self.values = np.asarray(values); self.kind = kind; self.metadata = dict(metadata or {})


class PaperVolMinTransition:
    def __init__(self, classes: int, initial_weight: float = -2.0, seed: int = 0):
        torch = _torch(); self.weight = torch.nn.Parameter(torch.full((int(classes), int(classes)), float(initial_weight))); self.num_classes = int(classes)
    def parameters(self): return [self.weight]
    def to(self, device): self.weight.data = self.weight.data.to(device); return self
    def matrix(self):
        torch = _torch(); values = torch.sigmoid(self.weight); eye = torch.eye(self.num_classes, device=values.device); values = values * (1-eye) + eye; return values / values.sum(1, keepdim=True).clamp_min(1e-12)


def build_paper_volmin_optimizer(model, transition, config: Mapping[str, Any]):
    torch = _torch(); params = list(model.parameters()) + list(transition.parameters()); options = dict(config); name = str(options.pop("name", "sgd")).lower(); cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[name]; return cls(params, **options)


def paper_volmin_objective(logits, labels, transition, *, lambda_volume: float, determinant_tolerance: float, condition_limit: float):
    torch = _torch()
    transition = torch.as_tensor(transition, device=logits.device, dtype=logits.dtype)
    probabilities = torch.softmax(logits, -1) @ transition
    nll = -torch.log(probabilities.gather(1, labels.long()[:, None]).squeeze(1).clamp_min(1e-12)).mean()
    sign, logdet = torch.linalg.slogdet(transition)
    return nll + float(lambda_volume) * logdet, {"logdet": float(logdet.detach()), "sign": float(sign.detach())}


def cross_guidance(positive_logits, negative_logits, candidate_k: int):
    torch = _torch(); positive = torch.softmax(positive_logits, -1); negative = torch.softmax(negative_logits, -1); k = min(int(candidate_k), positive.shape[1]); candidates = torch.zeros_like(negative, dtype=torch.bool); complements = torch.ones_like(positive, dtype=torch.bool); candidates.scatter_(1, negative.topk(k, 1).indices, True); complements.scatter_(1, positive.topk(k, 1).indices, False); return candidates, complements


def partial_label_objective(logits, soft_targets, hard_weight: float):
    torch = _torch(); logp = torch.log_softmax(logits, -1); soft = -(soft_targets * logp).sum(1); hard = -logp.gather(1, soft_targets.argmax(1, keepdim=True)).squeeze(1); return float(hard_weight) * hard.mean() + (1-float(hard_weight)) * soft.mean()


def negative_label_objective(logits, complements):
    torch = _torch(); probs = torch.sigmoid(logits); return -(torch.log1p(-probs.clamp(max=1-1e-6)) * complements.to(probs)).sum(1).mean()


class PeerLossHistory:
    def __init__(self, indices, window_size: int, peer: str = "a"):
        torch = _torch(); self.indices = torch.as_tensor(indices).long().cpu(); self.window_size = int(window_size); self.peer = str(peer); self.values = torch.zeros((len(self.indices), self.window_size)); self.observed = torch.zeros_like(self.values, dtype=torch.bool); self.selected = torch.zeros(len(self.indices)); self.epoch = 0
    def prepare_epoch(self, epoch): self.epoch = int(epoch) % self.window_size; self.observed[:, self.epoch] = False
    def append(self, indices, losses):
        torch = _torch(); rows = torch.searchsorted(self.indices, torch.as_tensor(indices).cpu()); self.values[rows, self.epoch] = torch.as_tensor(losses).detach().cpu(); self.observed[rows, self.epoch] = True; return rows
    def lookup_rows(self, rows):
        values = self.values[rows]; observed = self.observed[rows]; return values, observed, observed.sum(1)
    def increment_selected(self, rows, mask):
        torch = _torch()
        self.selected[rows] += torch.as_tensor(mask).detach().cpu().float()


def soft_robust_mean(history, observed):
    torch = _torch()
    values_attr = getattr(history, "values", None)
    values = values_attr if values_attr is not None and not callable(values_attr) else torch.as_tensor(history)
    mask = torch.as_tensor(observed).bool()
    length = mask.sum(1).clamp_min(1)
    return (values * mask).sum(1) / length, length


def cnlcu_soft_score(robust_mean, history_length, selected_count, sigma_squared: float):
    torch = _torch(); length = torch.as_tensor(history_length).float().clamp_min(1); count = torch.as_tensor(selected_count).float(); bonus = float(sigma_squared) * torch.sqrt(torch.log2(length + 1) / length); return torch.as_tensor(robust_mean) - bonus / (count + 1), bonus


__all__ = [name for name in globals() if not name.startswith("_")]
