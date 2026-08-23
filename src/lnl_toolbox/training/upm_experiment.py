from __future__ import annotations

"""Two-stage, resumable experiment runner for UPM."""

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
from lnl_toolbox.data import DataRequirements, DataRole
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
    _resolved_noise_config,
    build_model,
    build_optimizer,
    build_scheduler,
)
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.noisy_labels import (
    checkpoint_noise_metadata,
    effective_subset_actual_rate,
    noise_mode,
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


def run_upm_experiment(
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
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir, seed=seed, checkpoint_payload=checkpoint,
    )
    dataset, classes = prepared.dataset, prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None:
        raise ValueError("UPM requires noisy labels and a NoiseManifest")
    train_loader = prepared.loader(DataRole.TRAIN)
    posterior_loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    noise_metadata = checkpoint_noise_metadata(
        manifest, manifest_path, run_dir,
        effective_subset_actual_rate(manifest, prepared.train_indices),
        mode=noise_mode(config), validation_targets="noisy",
        effective_validation_rate=effective_subset_actual_rate(manifest, prepared.validation_indices),
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
        canonical_train_indices=np.sort(prepared.train_indices), device=device, run_dir=run_dir,
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


__all__ = ["UPMWorkflow", "run_upm_experiment"]
