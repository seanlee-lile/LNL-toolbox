from __future__ import annotations

"""UPM's warm-up, posterior snapshot and alternating training lifecycle."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lnl_toolbox.algorithms.upm import estimate_clean_posterior, update_confusion_probabilities_, upm_soft_target_objective
from lnl_toolbox.data.multiclass_synthetic import MulticlassTensorDataset, generate_synthetic_multiclass
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.noise.generators import generate_symmetric
from lnl_toolbox.noise.upm import UPMNoiseState
from lnl_toolbox.runtime import seed_everything
from lnl_toolbox.training.checkpoint import atomic_save, capture_rng_state, restore_rng_state
from lnl_toolbox.training.interfaces import RunContext
from lnl_toolbox.training.reproduction_data import build_reproduction_model, prepare_noisy_classification


class _UPMMLP(nn.Module):
    def __init__(self, dimension: int, width: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.ReLU(), nn.Linear(width, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dir(config: Mapping[str, Any], output_dir: str | Path | None, resume: str | Path | None) -> Path:
    path = Path(resume).resolve().parent if resume else (Path(output_dir).expanduser().resolve() if output_dir else Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data(config: Mapping[str, Any], run_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, int, np.ndarray]:
    d = config.get("data", {}); classes, dimension = int(d.get("num_classes", 3)), int(d.get("dimension", 6)); n = int(d.get("train_size", 90)); val_n = int(d.get("validation_size", 30)); test_n = int(d.get("test_size", 30)); seed = int(config.get("seed", 1))
    if str(d.get("name", "synthetic_multiclass")).lower() in {"cifar10", "cifar100"}:
        prepared = prepare_noisy_classification(config, run_dir, seed)
        return prepared.train_loader, prepared.snapshot_loader, prepared.validation_loader, prepared.test_loader, 0, prepared.num_classes, prepared.noisy_targets
    train = generate_synthetic_multiclass(n, dimension, classes, seed, start_index=0, split="train")
    val = generate_synthetic_multiclass(val_n, dimension, classes, seed + 1, start_index=n, split="validation")
    test = generate_synthetic_multiclass(test_n, dimension, classes, seed + 2, start_index=n + val_n, split="test")
    noise = config.get("noise", {}); manifest = generate_symmetric(train.labels, classes, float(noise.get("rate", 0.2)), int(noise.get("seed", seed + 10)), "synthetic_multiclass", sampling="per_class", rng="default_rng")
    snapshot = DataLoader(MulticlassTensorDataset(train, manifest.noisy_targets), batch_size=int(config.get("loader", {}).get("batch_size", 30)), shuffle=False)
    return DataLoader(MulticlassTensorDataset(train, manifest.noisy_targets), batch_size=int(config.get("loader", {}).get("batch_size", 30)), shuffle=True), snapshot, DataLoader(MulticlassTensorDataset(val), batch_size=30), DataLoader(MulticlassTensorDataset(test), batch_size=30), dimension, classes, manifest.noisy_targets


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(device)); correct += int(logits.argmax(1).eq(batch["target"].to(device)).sum()); total += logits.shape[0]
    return correct / max(total, 1)


def _snapshot(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> PosteriorSnapshot:
    rows: dict[int, tuple[np.ndarray, int]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(model(batch["input"].to(device)), dim=1).cpu().numpy()
            for index, probability, target in zip(batch["index"].tolist(), probs, batch["target"].tolist()):
                rows[int(index)] = (probability, int(target))
    indices = np.asarray(sorted(rows), dtype=np.int64)
    probabilities = np.stack([rows[int(i)][0] for i in indices])
    targets = np.asarray([rows[int(i)][1] for i in indices], dtype=np.int64)
    return PosteriorSnapshot(probabilities, targets, indices, "synthetic_multiclass", "train")


def _run_legacy_upm_experiment(config: dict[str, Any], output_dir: str | Path | None = None, resume: str | Path | None = None, *, context: RunContext | None = None) -> Path:
    run_dir = _dir(config, output_dir, resume); seed_everything(int(config.get("seed", 1))); train_loader, snapshot_loader, val_loader, test_loader, dimension, classes, noisy_targets = _data(config, run_dir)
    device = torch.device(str(config.get("trainer", {}).get("device", "cpu"))); model_cfg = config.get("model", {}); warm = (_UPMMLP(dimension, int(model_cfg.get("hidden_width", 16)), classes) if dimension else build_reproduction_model(model_cfg, config["data"], classes)).to(device); warm_opt = torch.optim.SGD(warm.parameters(), lr=float(config.get("optimizer", {}).get("lr", 0.05)), momentum=0.9)
    pre_epochs = int(config.get("warmup", {}).get("epochs", 1))
    for _ in range(pre_epochs):
        warm.train()
        for batch in train_loader:
            loss = nn.functional.cross_entropy(warm(batch["input"].to(device)), batch["target"].to(device)); warm_opt.zero_grad(); loss.backward(); warm_opt.step()
    snapshot = _snapshot(warm, snapshot_loader, device, classes)
    state = UPMNoiseState.from_snapshot(snapshot, torch.as_tensor(noisy_targets), eta_init=float(config.get("upm", {}).get("eta_init", 0.01)))
    model = ((_UPMMLP(dimension, int(model_cfg.get("hidden_width", 16)), classes) if dimension else build_reproduction_model(model_cfg, config["data"], classes)).to(device)); optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("optimizer", {}).get("lr", 0.05)), momentum=0.9)
    scheduler_cfg = config.get("scheduler", {}) or {}; scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(value) for value in scheduler_cfg.get("milestones", [])], gamma=float(scheduler_cfg.get("gamma", 0.1))) if scheduler_cfg.get("milestones") else None
    epochs = int(config.get("trainer", {}).get("epochs", 1)); start = 0; checkpoint = run_dir / "last.pt"
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); state.load_state_dict(payload["upm_state"]); start = int(payload["epoch"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
    upm_cfg = config.get("upm", {}); interval = max(1, int(upm_cfg.get("eta_update_interval", 1))); metrics = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    session = context.session if context is not None and context.state.get("lifecycle_active") else None
    if session is not None:
        session.start_phase("alternating_training", total_units=epochs)
    record: dict[str, Any] = {}
    with metrics:
        for epoch in range(start, epochs):
            model.train(); total = 0.0
            for batch in train_loader:
                inputs, targets, indices = batch["input"].to(device), batch["target"].to(device), batch["index"]
                logits = model(inputs); psi, eta = state.lookup(indices); q = estimate_clean_posterior(logits, targets, psi.to(device), eta.to(device)); loss = upm_soft_target_objective(logits, q); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * inputs.shape[0]
                if (epoch + 1) >= int(upm_cfg.get("eta_update_start_epoch", 1)) and (epoch + 1) % interval == 0:
                    update_confusion_probabilities_(state, indices, q, targets, learning_rate=float(upm_cfg.get("eta_lr", 0.7)))
            if scheduler is not None:
                scheduler.step()
            record = {"epoch": epoch + 1, "train_loss": total / len(train_loader.dataset), "validation_accuracy": _accuracy(model, val_loader, device), "test_accuracy": _accuracy(model, test_loader, device), "eta_mean": float(state.confusion_probability.mean())}
            if session is not None:
                session.log_epoch(
                    epoch + 1,
                    phase="alternating_training",
                    **{key: value for key, value in record.items() if key != "epoch"},
                )
            else:
                metrics.write(json.dumps(record) + "\n"); metrics.flush()
            checkpoint_payload = {
                "method": "upm",
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "upm_state": state.state_dict(),
                "config": config,
                "rng_state": capture_rng_state(),
            }
            if session is not None:
                session.save_checkpoint(
                    checkpoint_payload,
                    checkpoint,
                    phase="alternating_training",
                    completed_epoch=epoch + 1,
                    component_states={
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": {} if scheduler is None else scheduler.state_dict(),
                        "upm_state": state.state_dict(),
                    },
                    best_metric={"test_accuracy": record["test_accuracy"]},
                )
            else:
                atomic_save(checkpoint_payload, checkpoint)
    if session is not None:
        session.end_phase("alternating_training", completed_units=max(0, epochs - start))
        session.emit("final", phase="evaluation", method="upm", completed_epochs=epochs,
                     test_accuracy=record.get("test_accuracy") if epochs else None)
    return run_dir

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.supervised import SupervisedClassificationAlgorithm
from lnl_toolbox.algorithms.upm import (
    ConfusingProbabilityState,
    UPMAlgorithm,
    UPMConfig,
    UPMPhase,
    UPMState,
)
from lnl_toolbox.core import Batch, ExperimentContext, RunState
from lnl_toolbox.data import NoisyTargetDataset
from lnl_toolbox.data.cifar import load_cifar10, load_cifar100
from lnl_toolbox.data.torch_cifar import (
    TorchCifarDataset,
    build_cifar_transform,
    cifar_pixel_mean,
    stratified_split,
)
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.noise.estimators import PosteriorSnapshot
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    _resolved_noise_config,
    _subset,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
    prepare_noise_manifest,
)
from lnl_toolbox.training.snapshots import collect_posterior_snapshot


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _train_epoch(
    algorithm: SupervisedClassificationAlgorithm,
    loader: Any,
    state: RunState,
    epoch: int,
) -> dict[str, float]:
    state.cycle = epoch
    algorithm.on_cycle_start(state)
    loss_sum = all_loss_sum = correct_sum = samples = selected = 0.0
    for raw_batch in loader:
        result = algorithm.step(Batch(raw_batch), state)
        count = result.metrics["samples"]
        chosen = result.metrics["selected_samples"]
        loss_sum += result.metrics["loss"] * chosen
        all_loss_sum += result.metrics["all_sample_loss"] * count
        correct_sum += result.metrics["accuracy"] * count
        samples += count
        selected += chosen
    algorithm.on_cycle_end(state)
    if samples <= 0 or selected <= 0:
        raise RuntimeError("UPM training epoch contains no selected samples")
    return {
        "train_loss": loss_sum / selected,
        "train_all_sample_loss": all_loss_sum / samples,
        "train_accuracy": correct_sum / samples,
        "selected_ratio": selected / samples,
    }


def _save_eta(path: Path, state: ConfusingProbabilityState) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    values = state.state_dict()
    metadata = {
        "sample_index_mapping_hash": values["sample_index_mapping_hash"],
        "initial_value": values["initial_value"],
    }
    try:
        np.savez_compressed(
            temporary,
            canonical_sample_indices=values["canonical_sample_indices"].numpy(),
            eta=values["eta"].numpy(),
            update_count=values["update_count"].numpy(),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        with np.load(temporary, allow_pickle=False) as data:
            loaded_meta = json.loads(str(data["metadata_json"].item()))
            if loaded_meta != metadata:
                raise ValueError("temporary UPM eta metadata mismatch")
            if not np.array_equal(data["canonical_sample_indices"], values["canonical_sample_indices"].numpy()):
                raise ValueError("temporary UPM eta sample mapping mismatch")
            if not np.array_equal(data["eta"], values["eta"].numpy()):
                raise ValueError("temporary UPM eta values mismatch")
            if not np.array_equal(data["update_count"], values["update_count"].numpy()):
                raise ValueError("temporary UPM eta counts mismatch")
        temporary.replace(path)
        return _file_sha256(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_snapshot(path: Path, snapshot: PosteriorSnapshot) -> PosteriorSnapshot:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    try:
        snapshot.save(temporary)
        loaded = PosteriorSnapshot.load(temporary)
        if loaded.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("temporary UPM psi snapshot hash mismatch")
        temporary.replace(path)
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()


class UPMWorkflow:
    """Own Stage 1, psi publication, UPM Stage 2, and strict resume."""

    def __init__(
        self,
        *,
        stage1_model,
        stage1_optimizer,
        stage1_scheduler,
        main_model,
        main_optimizer,
        main_scheduler,
        loss,
        train_loader,
        posterior_loader,
        noisy_validation_loader,
        clean_test_loader,
        canonical_train_indices,
        device,
        run_dir,
        config,
        dataset,
        noise_metadata,
    ) -> None:
        self.config = dict(config)
        self.method_config = UPMConfig.from_mapping(config)
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.dataset = str(dataset)
        self.noise_metadata = dict(noise_metadata)
        self.loss = loss.to(self.device)
        self.train_loader = train_loader
        self.posterior_loader = posterior_loader
        self.noisy_validation_loader = noisy_validation_loader
        self.clean_test_loader = clean_test_loader
        self.stage1_model = stage1_model.to(self.device)
        self.stage1_optimizer = stage1_optimizer
        self.stage1_scheduler = stage1_scheduler
        self.main_model = main_model.to(self.device)
        self.main_optimizer = main_optimizer
        self.main_scheduler = main_scheduler
        if self.stage1_model is self.main_model or self.stage1_optimizer is self.main_optimizer:
            raise ValueError("UPM Stage 1 and main model/optimizer must be distinct")
        self.eta_state = ConfusingProbabilityState(
            torch.as_tensor(canonical_train_indices, dtype=torch.int64),
            self.method_config.confusing_probability.initial_value,
            device=self.device,
        )
        self.state = UPMState()
        self.stage1_run_state = RunState(phase="upm_stage1")
        self.main_run_state = RunState(phase="upm_main")
        self.stage1_algorithm = SupervisedClassificationAlgorithm(
            self.stage1_model, self.stage1_optimizer, self.loss, self.device
        )
        self.stage1_algorithm.setup(ExperimentContext(self.run_dir, self.config))
        self.main_algorithm: UPMAlgorithm | None = None
        self.snapshot: PosteriorSnapshot | None = None
        self.best_main_model_state: dict[str, torch.Tensor] | None = None
        self.best_eta_state: dict[str, Any] | None = None
        self.psi_provenance: dict[str, Any] = {}

    @property
    def last_path(self) -> Path: return self.run_dir / "last.pt"
    @property
    def best_path(self) -> Path: return self.run_dir / "best.pt"
    @property
    def stage1_best_path(self) -> Path: return self.run_dir / "stage1_best.pt"
    @property
    def snapshot_path(self) -> Path: return self.run_dir / "psi_snapshot.npz"
    @property
    def metrics_path(self) -> Path: return self.run_dir / "metrics.jsonl"

    def _checkpoint_payload(self, role: str) -> dict[str, Any]:
        return {
            "format_version": 2,
            "method": "upm",
            "checkpoint_role": role,
            "config": self.config,
            "upm_config_identity_hash": self.method_config.identity_hash,
            "upm_state": self.state.state_dict(),
            "stage1": {
                "model": self.stage1_model.state_dict(),
                "optimizer": self.stage1_optimizer.state_dict(),
                "scheduler": None if self.stage1_scheduler is None else self.stage1_scheduler.state_dict(),
                "run_state": {"cycle": self.stage1_run_state.cycle, "step": self.stage1_run_state.step, "phase": self.stage1_run_state.phase},
            },
            "main": {
                "model": self.main_model.state_dict(),
                "optimizer": self.main_optimizer.state_dict(),
                "scheduler": None if self.main_scheduler is None else self.main_scheduler.state_dict(),
                "run_state": {"cycle": self.main_run_state.cycle, "step": self.main_run_state.step, "phase": self.main_run_state.phase},
            },
            "eta_state": self.eta_state.state_dict(),
            "best_main_model_state": self.best_main_model_state,
            "best_eta_state": self.best_eta_state,
            "psi_provenance": self.psi_provenance,
            "noise": self.noise_metadata,
            "rng_state": capture_rng_state(),
        }

    def _save_last(self) -> None:
        atomic_save(self._checkpoint_payload("run_state"), self.last_path)

    def _append(self, row: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")

    def _seed_loader(self, epoch: int, offset: int) -> None:
        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(int(self.config.get("seed", 1)) + offset + epoch)

    def train_stage1(self, *, max_epochs: int | None = None) -> None:
        target = self.method_config.stage1.epochs
        if max_epochs is not None:
            target = min(target, self.state.stage1_completed_epochs + int(max_epochs))
        while self.state.stage1_completed_epochs < target:
            epoch = self.state.stage1_completed_epochs
            self._seed_loader(epoch, 0)
            train = _train_epoch(self.stage1_algorithm, self.train_loader, self.stage1_run_state, epoch)
            validation = evaluate_classification(
                self.stage1_model, self.noisy_validation_loader, self.loss, self.device
            )
            self.state.stage1_completed_epochs = epoch + 1
            self.state.stage1_global_step = self.stage1_run_state.step
            if validation["accuracy"] > self.state.stage1_best_validation_accuracy:
                self.state.stage1_best_epoch = epoch
                self.state.stage1_best_validation_accuracy = validation["accuracy"]
                atomic_save({
                    "method": "upm", "checkpoint_role": "stage1_best",
                    "epoch": epoch, "validation_accuracy": validation["accuracy"],
                    "model": self.stage1_model.state_dict(),
                }, self.stage1_best_path)
            if self.stage1_scheduler is not None:
                self.stage1_scheduler.step()
            self._append({"event": "stage1_epoch", "epoch": epoch, **train,
                          "noisy_validation_loss": validation["loss"],
                          "noisy_validation_accuracy": validation["accuracy"]})
            self._save_last()
        if self.state.stage1_completed_epochs < self.method_config.stage1.epochs:
            return
        best = read_checkpoint(self.stage1_best_path, self.device)
        self.stage1_model.load_state_dict(best["model"])
        self.state.stage1_best_checkpoint_sha256 = _file_sha256(self.stage1_best_path)
        self.state.advance(UPMPhase.STAGE1_READY)
        self._save_last()

    def publish_psi(self) -> None:
        if _file_sha256(self.stage1_best_path) != self.state.stage1_best_checkpoint_sha256:
            raise ValueError("UPM Stage 1 best checkpoint changed before psi collection")
        snapshot = collect_posterior_snapshot(
            self.stage1_model, self.posterior_loader, self.device,
            dataset=self.dataset, split="train",
        )
        expected = self.eta_state.canonical_sample_indices.detach().cpu().numpy()
        if not np.array_equal(snapshot.global_indices, expected):
            raise ValueError("UPM psi snapshot does not match the eta sample mapping")
        persisted = _atomic_snapshot(self.snapshot_path, snapshot)
        self.snapshot = persisted
        self.state.psi_snapshot_hash = persisted.snapshot_hash
        self.state.psi_file_sha256 = _file_sha256(self.snapshot_path)
        self.psi_provenance = {
            "source": "stage1_best",
            "source_checkpoint_sha256": self.state.stage1_best_checkpoint_sha256,
            "snapshot_hash": persisted.snapshot_hash,
            "snapshot_file_sha256": self.state.psi_file_sha256,
            "upm_config_identity_hash": self.method_config.identity_hash,
            "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
            "collection": "eval_inference_no_random_augmentation_stable_index",
        }
        _save_eta(self.run_dir / "eta_initial.npz", self.eta_state)
        self.state.advance(UPMPhase.PSI_READY)
        self._save_last()

    def _load_snapshot_strict(self) -> PosteriorSnapshot:
        if not self.snapshot_path.is_file():
            raise FileNotFoundError("UPM psi-ready checkpoint is missing psi_snapshot.npz")
        if _file_sha256(self.snapshot_path) != self.state.psi_file_sha256:
            raise ValueError("UPM psi snapshot file hash mismatch")
        snapshot = PosteriorSnapshot.load(self.snapshot_path)
        if snapshot.snapshot_hash != self.state.psi_snapshot_hash:
            raise ValueError("UPM psi snapshot content hash mismatch")
        expected = self.eta_state.canonical_sample_indices.detach().cpu().numpy()
        if not np.array_equal(snapshot.global_indices, expected):
            raise ValueError("UPM psi snapshot sample mapping changed")
        return snapshot

    def _prepare_main(self) -> None:
        if self.snapshot is None:
            self.snapshot = self._load_snapshot_strict()
        if self.main_algorithm is None:
            self.main_algorithm = UPMAlgorithm(
                self.main_model, self.main_optimizer, self.loss, self.device,
                snapshot=self.snapshot, eta_state=self.eta_state, config=self.config,
            )
            self.main_algorithm.setup(ExperimentContext(self.run_dir, self.config))
        if self.state.phase is UPMPhase.PSI_READY:
            self.state.advance(UPMPhase.MAIN_TRAINING)
            self._save_last()

    def train_main(self, *, max_epochs: int | None = None) -> None:
        self._prepare_main()
        assert self.main_algorithm is not None
        target = self.method_config.main.epochs
        if max_epochs is not None:
            target = min(target, self.state.main_completed_epochs + int(max_epochs))
        while self.state.main_completed_epochs < target:
            epoch = self.state.main_completed_epochs
            self._seed_loader(epoch, 100000)
            train = _train_epoch(self.main_algorithm, self.train_loader, self.main_run_state, epoch)
            validation = evaluate_classification(
                self.main_model, self.noisy_validation_loader, self.loss, self.device
            )
            self.state.main_completed_epochs = epoch + 1
            self.state.main_global_step = self.main_run_state.step
            if validation["accuracy"] > self.state.main_best_validation_accuracy:
                self.state.main_best_epoch = epoch
                self.state.main_best_validation_accuracy = validation["accuracy"]
                self.best_main_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.main_model.state_dict().items()
                }
                self.best_eta_state = self.eta_state.state_dict()
                atomic_save(self._checkpoint_payload("main_best"), self.best_path)
                _save_eta(self.run_dir / "eta_best.npz", self.eta_state)
            if self.main_scheduler is not None:
                self.main_scheduler.step()
            eta = self.eta_state.eta
            counts = self.eta_state.update_count
            self._append({
                "event": "main_epoch", "epoch": epoch, **train,
                "noisy_validation_loss": validation["loss"],
                "noisy_validation_accuracy": validation["accuracy"],
                "eta_mean": float(eta.mean().item()),
                "eta_min": float(eta.min().item()),
                "eta_max": float(eta.max().item()),
                "eta_update_count": float(counts.sum().item()),
            })
            _save_eta(self.run_dir / "eta_last.npz", self.eta_state)
            self._save_last()

    def complete(self) -> None:
        if self.best_main_model_state is None or self.best_eta_state is None:
            raise RuntimeError("UPM cannot complete without a paired best model and eta")
        current_model = {
            key: value.detach().cpu().clone()
            for key, value in self.main_model.state_dict().items()
        }
        self.main_model.load_state_dict(self.best_main_model_state)
        test = evaluate_classification(
            self.main_model, self.clean_test_loader, self.loss, self.device
        )
        self.main_model.load_state_dict(current_model)
        self.state.advance(UPMPhase.COMPLETED)
        eta_best_tensor = torch.as_tensor(self.best_eta_state["eta"])
        final = {
            "method": "upm",
            "phase": self.state.phase.value,
            "completed_epochs": self.state.main_completed_epochs,
            "global_step": self.state.main_global_step,
            "stage1_best_epoch": self.state.stage1_best_epoch,
            "stage1_best_noisy_validation_accuracy": self.state.stage1_best_validation_accuracy,
            "best_epoch": self.state.main_best_epoch,
            "best_noisy_validation_accuracy": self.state.main_best_validation_accuracy,
            "clean_test_loss": test["loss"],
            "clean_test_accuracy": test["accuracy"],
            "psi_snapshot_hash": self.state.psi_snapshot_hash,
            "eta_best_mean": float(eta_best_tensor.mean().item()),
            "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
            "test_selection_leakage": False,
        }
        (self.run_dir / "final_metrics.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        self._save_last()

    def run(self) -> None:
        if self.state.phase is UPMPhase.COMPLETED:
            if self.state.main_completed_epochs >= self.method_config.main.epochs:
                return
            self.state.phase = UPMPhase.MAIN_TRAINING
        if self.state.phase is UPMPhase.STAGE1_TRAINING:
            self.train_stage1()
        if self.state.phase is UPMPhase.STAGE1_READY:
            self.publish_psi()
        if self.state.phase in {UPMPhase.PSI_READY, UPMPhase.MAIN_TRAINING}:
            self.train_main()
            if self.state.main_completed_epochs >= self.method_config.main.epochs:
                self.complete()

    def resume(self, checkpoint: str | Path) -> None:
        payload = read_checkpoint(checkpoint, self.device)
        if payload.get("method") != "upm":
            raise ValueError("resume checkpoint is not a UPM run")
        saved_config = UPMConfig.from_mapping(payload["config"])
        if saved_config.identity_hash != self.method_config.identity_hash:
            raise ValueError("UPM identity settings changed on resume")
        if self.method_config.stage1.epochs != saved_config.stage1.epochs:
            raise ValueError("UPM Stage 1 epoch target changed on resume")
        if self.method_config.main.epochs < saved_config.main.epochs:
            raise ValueError("UPM main epoch target cannot be reduced")
        saved_noise = payload.get("noise", {})
        for key in ("mapping_hash", "manifest_sha256", "dataset", "num_classes"):
            if saved_noise.get(key) != self.noise_metadata.get(key):
                raise ValueError(f"UPM resume noise identity changed: {key}")
        self.state = UPMState.from_state_dict(payload["upm_state"])
        self.stage1_model.load_state_dict(payload["stage1"]["model"])
        self.stage1_optimizer.load_state_dict(payload["stage1"]["optimizer"])
        _move_optimizer_state(self.stage1_optimizer, self.device)
        self.main_model.load_state_dict(payload["main"]["model"])
        self.main_optimizer.load_state_dict(payload["main"]["optimizer"])
        _move_optimizer_state(self.main_optimizer, self.device)
        for scheduler, saved, owner in (
            (self.stage1_scheduler, payload["stage1"]["scheduler"], "Stage 1"),
            (self.main_scheduler, payload["main"]["scheduler"], "main"),
        ):
            if (scheduler is None) != (saved is None):
                raise ValueError(f"UPM {owner} scheduler configuration changed")
            if scheduler is not None:
                scheduler.load_state_dict(saved)
        self.stage1_run_state = RunState(**payload["stage1"]["run_state"])
        self.main_run_state = RunState(**payload["main"]["run_state"])
        self.eta_state.load_state_dict(payload["eta_state"])
        self.best_main_model_state = payload.get("best_main_model_state")
        self.best_eta_state = payload.get("best_eta_state")
        self.psi_provenance = dict(payload.get("psi_provenance", {}))
        if self.state.phase in {UPMPhase.PSI_READY, UPMPhase.MAIN_TRAINING, UPMPhase.COMPLETED}:
            self.snapshot = self._load_snapshot_strict()
            expected = {
                "source_checkpoint_sha256": self.state.stage1_best_checkpoint_sha256,
                "snapshot_hash": self.state.psi_snapshot_hash,
                "snapshot_file_sha256": self.state.psi_file_sha256,
                "upm_config_identity_hash": self.method_config.identity_hash,
                "noise_mapping_hash": self.noise_metadata.get("mapping_hash", ""),
                "manifest_sha256": self.noise_metadata.get("manifest_sha256", ""),
            }
            if any(self.psi_provenance.get(key) != value for key, value in expected.items()):
                raise ValueError("UPM psi artifact provenance mismatch")
        restore_rng_state(payload["rng_state"])

    def close(self) -> None:
        self.stage1_algorithm.close()
        if self.main_algorithm is not None:
            self.main_algorithm.close()


def _run_upm_paper_workflow(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    config = deepcopy(config)
    method = UPMConfig.from_mapping(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    trainer = config.get("trainer", {}) or {}
    device = resolve_device(trainer.get("device", "auto"))
    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(resume, "cpu") if resume is not None else None
    if checkpoint is not None and checkpoint.get("method") != "upm":
        raise ValueError("resume checkpoint is not a UPM run")
    extension_requested = False
    if checkpoint is not None:
        saved_method = UPMConfig.from_mapping(checkpoint["config"])
        extension_requested = method.main.epochs > saved_method.main.epochs

    data_config = config["data"]
    dataset = str(data_config.get("name", "cifar10")).lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError("UPM first version supports CIFAR-10 and CIFAR-100")
    loader_fn = load_cifar10 if dataset == "cifar10" else load_cifar100
    classes = 10 if dataset == "cifar10" else 100
    train_data = loader_fn(data_config.get("root"), "train")
    test_data = loader_fn(data_config.get("root"), "test")
    train_indices, validation_indices = stratified_split(
        train_data.labels, int(data_config["validation_size"]), seed
    )
    manifest_indices = np.sort(np.concatenate((train_indices, validation_indices)))
    manifest, manifest_path = prepare_noise_manifest(
        config, dataset=dataset, clean_targets=train_data.labels[manifest_indices],
        global_indices=manifest_indices, num_classes=classes, run_dir=run_dir,
        checkpoint_payload=checkpoint, dataset_targets=train_data.labels,
    )
    if manifest is None or manifest_path is None:
        raise ValueError("UPM requires noisy labels and a NoiseManifest")
    train_indices = _subset(train_indices, train_data.labels, data_config.get("max_train_samples"), seed + 1)
    validation_indices = _subset(validation_indices, train_data.labels, data_config.get("max_validation_samples"), seed + 2)
    test_indices = _subset(np.arange(len(test_data)), test_data.labels, data_config.get("max_test_samples"), seed + 3)
    preprocessing = str(data_config.get("preprocessing", "standard")).lower()
    pixel_mean = cifar_pixel_mean(train_data.images) if preprocessing == "gce2018" else None
    transform_options = {"preprocessing": preprocessing, "pixel_mean": pixel_mean}
    noisy_train = NoisyTargetDataset(
        TorchCifarDataset(train_data, train_indices, transform=build_cifar_transform(True, bool(data_config.get("augment", True)), **transform_options)),
        manifest.global_indices, manifest.noisy_targets,
    )
    deterministic_train = NoisyTargetDataset(
        TorchCifarDataset(train_data, train_indices, transform=build_cifar_transform(False, **transform_options)),
        manifest.global_indices, manifest.noisy_targets,
    )
    noisy_validation = NoisyTargetDataset(
        TorchCifarDataset(train_data, validation_indices, transform=build_cifar_transform(False, **transform_options)),
        manifest.global_indices, manifest.noisy_targets,
    )
    clean_test = TorchCifarDataset(test_data, test_indices, transform=build_cifar_transform(False, **transform_options))
    loader_config = config["loader"]
    train_loader = _loader(noisy_train, loader_config, shuffle=True, seed=seed)
    posterior_loader = _loader(deterministic_train, loader_config, shuffle=False, seed=seed)
    validation_loader = _loader(noisy_validation, loader_config, shuffle=False, seed=seed)
    test_loader = _loader(clean_test, loader_config, shuffle=False, seed=seed)
    noise_metadata = checkpoint_noise_metadata(
        manifest, manifest_path, run_dir,
        effective_subset_actual_rate(manifest, train_indices),
        mode=noise_mode(config), validation_targets="noisy",
        effective_validation_rate=effective_subset_actual_rate(manifest, validation_indices),
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    stage1_model = build_model(method.stage1.model, classes)
    stage1_optimizer = build_optimizer(stage1_model, method.stage1.optimizer)
    stage1_scheduler = build_scheduler(stage1_optimizer, method.stage1.scheduler, method.stage1.epochs)
    main_model = build_model(method.main.model, classes)
    main_optimizer = build_optimizer(main_model, method.main.optimizer)
    main_scheduler = build_scheduler(main_optimizer, method.main.scheduler, method.main.epochs)
    loss = build_builtin_loss({"name": "ce"})
    if resume is None:
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "environment.json").write_text(json.dumps(_environment(seed, device), indent=2), encoding="utf-8")
        (run_dir / "noise_summary.json").write_text(json.dumps(noise_metadata, indent=2), encoding="utf-8")
    workflow = UPMWorkflow(
        stage1_model=stage1_model, stage1_optimizer=stage1_optimizer, stage1_scheduler=stage1_scheduler,
        main_model=main_model, main_optimizer=main_optimizer, main_scheduler=main_scheduler,
        loss=loss, train_loader=train_loader, posterior_loader=posterior_loader,
        noisy_validation_loader=validation_loader, clean_test_loader=test_loader,
        canonical_train_indices=np.sort(train_indices), device=device, run_dir=run_dir,
        config=config, dataset=dataset, noise_metadata=noise_metadata,
    )
    try:
        if resume is not None:
            workflow.resume(resume)
            if extension_requested:
                resolved_path = run_dir / "resolved_config.yaml"
                temporary_path = resolved_path.with_suffix(".yaml.tmp")
                temporary_path.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                temporary_path.replace(resolved_path)
        workflow.run()
    finally:
        workflow.close()
    return run_dir


def run_upm_experiment(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    context: RunContext | None = None,
) -> Path:
    """Run the canonical paper workflow through the existing public entry point."""

    settings = config.get("upm", {}) if isinstance(config, Mapping) else {}
    if isinstance(settings, Mapping) and any(key in settings for key in ["stage1","main"]):
        return _run_upm_paper_workflow(dict(config), output_dir=output_dir, resume=resume)
    return _run_legacy_upm_experiment(config, output_dir=output_dir, resume=resume, context=context)


__all__ = ["run_upm_experiment"]
