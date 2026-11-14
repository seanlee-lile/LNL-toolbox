from __future__ import annotations

"""Standalone CWD lifecycle for the CIFAR airplane/automobile benchmark."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lnl_toolbox.algorithms.cwd import CWDGlobalObjective
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.estimators.cwd import CWDEstimator
from lnl_toolbox.evaluation.classification import evaluate_classification
from lnl_toolbox.losses.torch_losses import CrossEntropyLoss
from lnl_toolbox.models.cifar_resnet import cifar_resnet34
from lnl_toolbox.models.feature_output import forward_with_features
from lnl_toolbox.models.feature_output import FeatureOutput
from lnl_toolbox.models.tiny_cnn import TinyCNN
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
)
from lnl_toolbox.training.progress import standardize_epoch_row, write_training_curves_svg
from lnl_toolbox.training.data_service import prepare_experiment_data
from lnl_toolbox.training.experiment import (
    bind_model_input, build_model, build_optimizer, build_scheduler,
)
from lnl_toolbox.training.snapshots import collect_feature_snapshot


class _FeatureMLP(torch.nn.Module):
    """Expose the same feature/classifier contract as image backbones."""

    def __init__(self, input_dim: int, hidden_width: int, outputs: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_width), torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(hidden_width, outputs)

    def forward_with_features(self, inputs):
        features = self.encoder(inputs)
        return FeatureOutput(self.classifier(features), features)

    def forward(self, inputs):
        return self.forward_with_features(inputs).logits


def _build_model(config: Mapping[str, Any]):
    name = str(config.get("name", "resnet34")).strip().lower()
    if str(config.get("variant", "standard")).lower() != "standard" or str(config.get("weights") or "none").lower() != "none":
        raise ValueError("CWD's dedicated model supports only standard architecture without pretrained weights")
    if name in {"resnet34", "cifar_resnet34"}:
        outputs = int(config.get("num_outputs", 2))
        model = cifar_resnet34(
            outputs,
            int(config.get("base_width", 64)),
            bias=bool(config.get("bias", True)),
            stem_padding=int(config.get("stem_padding", 1)),
            initialization=str(config.get("initialization", "kaiming")),
        )
        probability = float(config.get("feature_dropout", 0.0))
        if not 0.0 <= probability < 1.0:
            raise ValueError("feature_dropout must be in [0, 1)")
        model.feature_dropout.p = probability
        return model
    if name == "tiny_cnn":
        return TinyCNN(int(config.get("num_outputs", 2)), int(config.get("width", 8)))
    if name == "feature_mlp":
        input_dim = int(config.get("input_dim", 0))
        hidden_width = int(config.get("hidden_width", 16))
        if input_dim <= 0 or hidden_width <= 0:
            raise ValueError("CWD feature_mlp requires positive input dimensions")
        return _FeatureMLP(input_dim, hidden_width, int(config.get("num_outputs", 2)))
    raise ValueError(f"Unsupported CWD model: {name}")


def _classification_from_cwd_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("CWD logits must have shape [B, C]")
    if logits.shape[1] == 1:
        return (logits[:, 0] >= 0.0).long()
    return logits.argmax(1)


@torch.no_grad()
def _evaluate_cwd(model, loader, device, *, scalar_binary: bool) -> dict[str, float]:
    model.eval()
    loss_sum = correct = samples = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        if scalar_binary:
            signed = targets.to(dtype=output.logits.dtype).mul(2.0).sub(1.0)
            margins = output.logits[:, 0]
            loss = (1.0 - signed * margins).square()
        else:
            loss = torch.nn.functional.cross_entropy(output.logits, targets, reduction="none")
        predictions = _classification_from_cwd_logits(output.logits)
        count = int(targets.numel())
        loss_sum += float(loss.sum())
        correct += float(predictions.eq(targets).sum())
        samples += count
    if samples == 0:
        raise ValueError("CWD evaluation loader must not be empty")
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def _apply_cwd_beta1_schedule(optimizer, epoch: int, *, enabled: bool = False) -> None:
    """Apply the reference release's zero-based epoch-80 Adam beta1 switch.

    Leave the loaded optimizer untouched when disabled, including on resume.
    LR, beta2, optimizer state and scheduler ownership remain unchanged.
    """
    if enabled:
        beta1 = 0.9 if epoch < 80 else 0.1
        for group in optimizer.param_groups:
            group["betas"] = (beta1, group["betas"][1])


def _train_epoch(model, optimizer, loader, objective, device) -> tuple[float, float]:
    model.train()
    loss_sum = correct = samples = 0.0
    base_loss = CrossEntropyLoss().to(device)
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        indices = batch["index"].to(device, non_blocking=True)
        output = forward_with_features(model, inputs)
        loss = objective.compute(
            model=model,
            logits=output.logits,
            features=output.features,
            noisy_targets=targets,
            sample_indices=indices,
            base_loss=base_loss,
            metadata={},
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count = int(targets.numel())
        loss_sum += float(loss.detach()) * count
        correct += float(_classification_from_cwd_logits(output.logits).eq(targets).sum())
        samples += count
    return loss_sum / samples, correct / samples


def _estimate_cwd_transition(prepared, config, run_dir, device, *, publish_last=True):
    """Own the VolMin algorithm stage; never load another experiment's result."""
    from lnl_toolbox.algorithms.volminnet import VolMinNetAlgorithm, VolMinTransition

    stage = config["cwd"]["transition_stage"]
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(stage["epochs"])
    validation_size = int(stage.get("validation_size", 0))
    selection = str(stage.get("selection", "final"))
    if selection not in {"final", "noisy_validation_loss"}:
        raise ValueError("CWD transition selection must be final or noisy_validation_loss")
    select_validation = selection == "noisy_validation_loss"
    if validation_size < 0 or validation_size >= len(prepared.train_indices):
        raise ValueError("CWD transition validation_size must leave training samples")
    if select_validation and validation_size == 0:
        raise ValueError("CWD transition noisy validation selection requires held-out samples")
    identity = {
        "config": config,
        "data_fingerprint": prepared.data_fingerprint,
        "noise_mapping_hash": prepared.manifest.mapping_hash,
        "train_indices_sha256": hashlib.sha256(
            np.asarray(prepared.train_indices, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    positions = {int(index): offset for offset, index in enumerate(prepared.manifest.global_indices)}
    observed = np.asarray(prepared.manifest.noisy_targets)[
        [positions[int(index)] for index in prepared.train_indices]
    ]
    observed_prior = np.bincount(observed, minlength=prepared.num_classes) / len(observed)
    fit_indices = np.asarray(prepared.train_indices)
    validation_loader = None
    if validation_size:
        permutation = np.random.default_rng(int(config.get("seed", 1))).permutation(fit_indices)
        validation_indices = np.sort(permutation[:validation_size])
        fit_indices = np.sort(permutation[validation_size:])
        # Both subsets use observed labels; never fall back to raw clean labels.
        targets_by_index = {
            int(index): int(prepared.manifest.noisy_targets[offset])
            for index, offset in positions.items()
        }
        validation_loader = prepared.subset_loader(
            validation_indices, targets_by_index=targets_by_index,
            training=False, shuffle=False, drop_last=False, stream=1001,
        )
        np.savez(run_dir / "cwd_transition_split.npz", train=fit_indices, noisy_validation=validation_indices)
    path = run_dir / "cwd_transition.pt"
    saved = read_checkpoint(path, "cpu") if path.is_file() else None
    if saved is not None and saved.get("identity") != identity:
        raise ValueError("CWD internal transition checkpoint data/config mismatch")
    if saved is not None and int(saved["completed_epochs"]) == epochs:
        key = "selected_matrix" if select_validation else "matrix"
        if saved.get(key) is None:
            raise ValueError("CWD internal transition has no valid selected matrix")
        return np.asarray(saved[key], dtype=np.float64).copy()

    # Keep CWD's model/dropout RNG independent of the auxiliary lifecycle.
    outer_rng = capture_rng_state()
    try:
        # An optional independent auxiliary stream derives from the one public
        # experiment seed. Restore the main stream in finally, including errors.
        if bool(stage.get("independent_rng", False)):
            seed_everything(int(config.get("seed", 1)))
        model_config = bind_model_input(stage["model"], prepared.input_spec)
        auxiliary = build_model(model_config, prepared.num_classes).to(device)
        transition_model = VolMinTransition(prepared.num_classes).to(device)
        if "initial_raw_value" in stage:
            initial = float(stage["initial_raw_value"])
            if not np.isfinite(initial):
                raise ValueError("CWD transition initial_raw_value must be finite")
            with torch.no_grad():
                transition_model.off_diagonal_logits.fill_(initial)
        classifier_optimizer = build_optimizer(auxiliary, stage["optimizer"]["classifier"])
        transition_optimizer = build_optimizer(transition_model, stage["optimizer"]["transition"])
        algorithm = VolMinNetAlgorithm(
            model=auxiliary,
            transition=transition_model,
            classifier_optimizer=classifier_optimizer,
            transition_optimizer=transition_optimizer,
            classifier_scheduler=build_scheduler(
                classifier_optimizer, stage["scheduler"]["classifier"], epochs
            ),
            transition_scheduler=build_scheduler(
                transition_optimizer, stage["scheduler"]["transition"], epochs
            ),
            lambda_volume=float(stage["lambda_volume"]),
            device=device,
        )
        completed = 0
        rows = []
        selected_matrix = None
        selected_epoch = None
        selected_loss = float("inf")
        if saved is not None:
            algorithm.load_state_dict(saved["algorithm"])
            restore_rng_state(saved["rng_state"])
            completed = int(saved["completed_epochs"])
            rows = list(saved["metrics"])
            selected_matrix = saved.get("selected_matrix")
            selected_epoch = saved.get("selected_epoch")
            selected_loss = float(saved.get("selected_loss", float("inf")))

        def persist():
            payload = {
                "format_version": 1,
                "method": "cwd",
                "phase": "cwd_transition",
                "config": config,
                "identity": identity,
                "algorithm": algorithm.state_dict(),
                "completed_epochs": completed,
                "rng_state": capture_rng_state(),
                "matrix": transition_model.matrix().detach().cpu().numpy().copy(),
                "metrics": rows,
                "selection": selection,
                "selected_matrix": selected_matrix,
                "selected_epoch": selected_epoch,
                "selected_loss": selected_loss,
            }
            atomic_save(payload, path)
            if publish_last:
                atomic_save(payload, run_dir / "last.pt")

        if saved is None:
            persist()
        for epoch in range(completed, epochs):
            started = time.perf_counter()
            loader = (
                prepared.subset_loader(
                    fit_indices, targets_by_index=targets_by_index,
                    training=True, shuffle=True, drop_last=False, epoch=epoch, stream=1000,
                ) if validation_size else
                prepared.loader(DataRole.TRAIN, epoch=epoch, stream=1000)
            )
            loss_sum = classification_sum = samples = 0.0
            for batch in loader:
                metrics = algorithm.train_batch(batch)
                samples += metrics["samples"]
                loss_sum += metrics["objective"] * metrics["samples"]
                classification_sum += metrics["classification_loss"] * metrics["samples"]
            if samples == 0:
                raise ValueError("CWD transition training data is empty")
            completed = epoch + 1
            algorithm.state.completed_epochs = completed
            algorithm.state.completed = completed == epochs
            algorithm.step_schedulers()
            row = {
                "event": "epoch", "phase": "cwd_transition", "epoch": completed,
                "global_step": algorithm.state.global_step,
                "objective": loss_sum / samples,
                "classification_loss": classification_sum / samples,
                "epoch_seconds": time.perf_counter() - started,
                **transition_model.diagnostics(),
            }
            # Diagnostics only: never use clean labels, or alter the learned matrix/prior.
            matrix = transition_model.matrix().detach().cpu().numpy()
            try:
                prior = np.linalg.solve(matrix.T, observed_prior)
                row.update({
                    "inferred_prior_min": float(prior.min()),
                    "inferred_prior_negative_classes": int((prior < -1e-8).sum()),
                    "inferred_prior_residual": float(np.linalg.norm(matrix.T @ prior - observed_prior)),
                })
            except np.linalg.LinAlgError:
                row["inferred_prior_error"] = "singular_matrix"
            if validation_loader is not None:
                auxiliary.eval()
                validation_loss = validation_samples = 0.0
                with torch.no_grad():
                    transition_tensor = transition_model.matrix(dtype=torch.float32)
                    for batch in validation_loader:
                        logits = auxiliary(batch["input"].to(device, non_blocking=True))
                        targets = batch["target"].to(device, non_blocking=True).long()
                        log_probability = torch.logsumexp(
                            torch.log_softmax(logits, dim=1)[:, :, None]
                            + transition_tensor.log()[None, :, :], dim=1,
                        )
                        validation_loss += float(torch.nn.functional.nll_loss(
                            log_probability, targets, reduction="sum"
                        ))
                        validation_samples += int(targets.numel())
                if validation_samples == 0:
                    raise ValueError("CWD transition noisy validation data is empty")
                row["noisy_validation_loss"] = validation_loss / validation_samples
                feasible = (
                    "inferred_prior_error" not in row
                    and row.get("inferred_prior_negative_classes", 1) == 0
                )
                row["selection_eligible"] = feasible or not bool(stage.get("require_valid_prior", False))
                if (select_validation and row["selection_eligible"]
                        and np.isfinite(row["noisy_validation_loss"])
                        and row["noisy_validation_loss"] < selected_loss):
                    selected_loss = row["noisy_validation_loss"]
                    selected_epoch = completed
                    selected_matrix = matrix.copy()
            rows.append(row)
            persist()
            (run_dir / "transition_metrics.jsonl").write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
                encoding="utf-8",
            )
            print(json.dumps(row, sort_keys=True), flush=True)
        if select_validation:
            if selected_matrix is None:
                raise ValueError("CWD transition stage found no valid noisy-validation-selected matrix")
            return np.asarray(selected_matrix, dtype=np.float64).copy()
        return transition_model.matrix().detach().cpu().numpy().copy()
    finally:
        restore_rng_state(outer_rng)


def run_cwd_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    *,
    requirements: DataRequirements | None = None,
) -> Path:
    """Run one configured fold of the paper's five-fold CIFAR-binary protocol."""

    config = deepcopy(config)
    # Persist the resolved algorithm choice. Old checkpoints which omitted
    # this setting used a static centroid and must not silently change it.
    config.setdefault("cwd", {}).setdefault("dynamic_centroid", True)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = resolve_device(str(config["trainer"].get("device", "auto")))
    run_dir = (
        Path(resume).resolve().parent
        if resume is not None
        else Path(
            output_dir
            or Path(config.get("output_root", "artifacts/runs"))
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None if resume is None else read_checkpoint(resume, "cpu")
    if checkpoint is not None and checkpoint.get("phase") == "cwd_transition":
        # The helper resumes its own algorithm/optimizers/RNG from cwd_transition.pt.
        if not (run_dir / "cwd_transition.pt").is_file():
            raise FileNotFoundError("CWD internal transition checkpoint is missing")
        checkpoint = None
    if requirements is None:
        from lnl_toolbox.training.runners import resolve_data_requirements

        requirements = resolve_data_requirements(config, expected_runner="cwd")

    data_config = config["data"]
    folds = int(data_config.get("folds", 5))
    fold_index = int(data_config.get("fold_index", 0))
    noise_config = config["noise"]
    rho_positive = float(noise_config.get("rho_positive", 0.0))
    rho_negative = float(noise_config.get("rho_negative", 0.0))
    prepared = prepare_experiment_data(
        config,
        requirements=requirements,
        run_dir=run_dir, seed=seed, checkpoint_payload=checkpoint,
    )
    # Recreate workers each epoch so resumed runs consume the same loader RNG
    # sequence and initialize worker RNGs identically to uninterrupted runs.
    prepared.loader_config = {**prepared.loader_config, "persistent_workers": False}
    multiclass = str(config.get("cwd", {}).get("variant", "")).lower() == "multiclass"
    if (not multiclass and prepared.num_classes != 2) or prepared.manifest is None or prepared.manifest_path is None:
        raise ValueError("CWD requires a noisy indexed dataset matching the selected variant")
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if not multiclass:
        manifest.metadata.update({"folds": folds, "fold_index": fold_index})
    if checkpoint is not None and checkpoint.get("noise_mapping_hash") != manifest.mapping_hash:
        raise ValueError("CWD resume noise manifest mismatch")
    train_loader = prepared.loader(DataRole.TRAIN)
    snapshot_loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)
    loaders = {"train": train_loader, "snapshot": snapshot_loader, "test": test_loader}
    validation_loader = (prepared.validation_loader(shuffle=False)
                         if {DataRole.CLEAN_VALIDATION, DataRole.NOISY_VALIDATION}.intersection(prepared.available_roles)
                         else None)
    if validation_loader is not None:
        loaders["validation"] = validation_loader

    cwd_config = config.get("cwd", {})
    cwd_variant = str(cwd_config.get("variant", "multiclass")).strip().lower()
    model_config = dict(config["model"])
    if multiclass:
        model_config["num_outputs"] = prepared.num_classes
    if str(model_config.get("name", "")).lower() == "feature_mlp" and "input_dim" not in model_config:
        if prepared.input_spec.feature_dim is None:
            raise ValueError("CWD feature_mlp requires a known feature dimension")
        model_config["input_dim"] = prepared.input_spec.feature_dim
    if cwd_variant == "binary_scalar":
        model_config["num_outputs"] = 1
    model = _build_model(model_config).to(device)
    optimizer_config = config["optimizer"]
    beta1_schedule = optimizer_config.get("beta1_schedule", False)
    if not isinstance(beta1_schedule, bool):
        raise ValueError("optimizer.beta1_schedule must be a boolean")
    if str(optimizer_config.get("name", "adam")).lower() != "adam":
        raise ValueError("CWD paper runner requires Adam")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    scheduler_config = dict(config.get("scheduler", {}))
    scheduler_config.setdefault("name", "multistep")
    scheduler = build_scheduler(optimizer, scheduler_config, int(config["trainer"]["epochs"]))
    start_epoch = 0
    rows: list[dict[str, Any]] = []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        rows = list(checkpoint.get("metrics", []))
        # PreparedData loaders own generators independent of the global RNG.
        # Restoring only rng_state restarts their sample order on resume.
        loader_states = checkpoint.get("loader_rng_state", {})
        if start_epoch < int(config["trainer"]["epochs"]) and set(loader_states) != set(loaders):
            raise ValueError("CWD checkpoint is missing data-loader RNG states; exact resume is unavailable")
        for name, state in loader_states.items():
            loaders[name].generator.set_state(state.cpu())

    if multiclass:
        if "transition_artifact" in cwd_config:
            raise ValueError("CWD no longer accepts an external transition_artifact")
        if checkpoint is not None and not (run_dir / "cwd_transition.pt").is_file():
            raise FileNotFoundError("CWD internal transition checkpoint is missing")
        transition = _estimate_cwd_transition(
            prepared, config, run_dir, device, publish_last=checkpoint is None
        )
        if transition.shape != (prepared.num_classes, prepared.num_classes):
            raise ValueError("CWD learned transition class count mismatch")
        (run_dir / "transition-source.json").write_text(json.dumps({
            "path": "cwd_transition.pt", "source": "cwd_internal_volmin_algorithm",
            "noise_mapping_hash": manifest.mapping_hash,
            "selection": cwd_config["transition_stage"].get("selection", "final"),
            "selected_epoch": read_checkpoint(run_dir / "cwd_transition.pt", "cpu").get("selected_epoch"),
        }, indent=2), encoding="utf-8")
    else:
        transition = np.asarray([
            [1.0 - rho_negative, rho_negative],
            [rho_positive, 1.0 - rho_positive],
        ], dtype=np.float64)
    pinv_rcond = config.get("cwd", {}).get("pinv_rcond")
    if pinv_rcond is None and "ridge" in config.get("cwd", {}):
        # Backward-compatible parsing for old smoke configurations.  The
        # reproduction configuration uses the paper's unregularized pinv.
        pinv_rcond = float(config["cwd"]["ridge"])
    estimator = CWDEstimator(transition, pinv_rcond=pinv_rcond)
    criterion = CrossEntropyLoss().to(device)
    epochs = int(config["trainer"]["epochs"])
    metrics_path = run_dir / "metrics.jsonl"
    from lnl_toolbox.training.early_stopping import StageEarlyStopping
    stopping = StageEarlyStopping(config, "trainer.early_stopping", "validation_accuracy", run_dir=run_dir)
    if stopping.enabled and validation_loader is None:
        raise ValueError("trainer.early_stopping requires data.num_val or data.validation_size")
    stopping.replay(rows, completed=start_epoch)
    for epoch in range(start_epoch, epochs):
        if stopping.stopped:
            break
        epoch_started = time.perf_counter()
        _apply_cwd_beta1_schedule(optimizer, epoch, enabled=beta1_schedule)
        snapshot = collect_feature_snapshot(
            model,
            snapshot_loader,
            device,
            dataset=prepared.dataset,
            split="train" if multiclass else f"train_fold_{fold_index}",
            feature_extractor=lambda current, inputs: forward_with_features(current, inputs).features,
        )
        artifact = estimator.estimate(snapshot)
        snapshot.save(run_dir / "feature_snapshot.npz")
        artifact.save(run_dir / "statistic_artifact.npz")
        objective = CWDGlobalObjective(
            artifact,
            variant=str(config.get("cwd", {}).get("variant", "multiclass")),
            dynamic_centroid=bool(
                config.get("cwd", {}).get("dynamic_centroid", True)
            ),
        )
        train_loss, train_accuracy = _train_epoch(
            model, optimizer, train_loader, objective, device
        )
        test = _evaluate_cwd(
            model,
            test_loader,
            device,
            scalar_binary=cwd_variant == "binary_scalar",
        )
        row = standardize_epoch_row(
            {
                "epoch": epoch + 1,
                "phase": "cwd",
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "statistic_hash": artifact.artifact_hash,
                "feature_snapshot_hash": snapshot.snapshot_hash,
                "fold_index": fold_index,
            },
            require_validation=False,
        )
        rows.append(row)
        if beta1_schedule:
            row["adam_beta1"] = float(optimizer.param_groups[0]["betas"][0])
        if validation_loader is not None:
            validation = _evaluate_cwd(model, validation_loader, device, scalar_binary=cwd_variant == "binary_scalar")
            row.update({"validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]})
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        print(json.dumps(row, sort_keys=True), flush=True)
        metrics_path.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
            encoding="utf-8",
        )
        if scheduler is not None:
            scheduler.step()
        atomic_save(
            {
                "format_version": 1,
                "method": "cwd",
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "completed_epoch": epoch,
                "metrics": rows,
                "rng_state": capture_rng_state(),
                "loader_rng_state": {
                    name: loader.generator.get_state() for name, loader in loaders.items()
                },
                "noise_mapping_hash": manifest.mapping_hash,
                "statistic_hash": artifact.artifact_hash,
                "feature_snapshot_hash": snapshot.snapshot_hash,
            },
            run_dir / "last.pt",
        )
        if stopping.observe(row, {"model": model, "optimizer": optimizer, "scheduler": scheduler}):
            break
    stopping.restore_best({"model": model})
    final_test = _evaluate_cwd(model, test_loader, device, scalar_binary=cwd_variant == "binary_scalar")
    (run_dir / "final_metrics.json").write_text(json.dumps({
        "event": "final", "method": "cwd", "completed_epochs": len(rows),
        "test_loss": final_test["loss"], "test_accuracy": final_test["accuracy"],
        "selection_split": "validation" if validation_loader is not None else "none",
    }, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if rows:
        write_training_curves_svg(rows, run_dir / "training_curves.svg")
    return run_dir


__all__ = ["run_cwd_experiment"]
