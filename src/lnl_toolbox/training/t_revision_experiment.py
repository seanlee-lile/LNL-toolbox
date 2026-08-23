from __future__ import annotations

"""CIFAR assembly for T-Revision Reweight-R."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.t_revision import TRevisionAlgorithm, TRevisionConfig
from lnl_toolbox.data import DataRequirements, DataRole
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.checkpoint import read_checkpoint
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


def _preflight_model_output(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
) -> None:
    """Validate the public model contract without creating inference parameters."""

    was_training = model.training
    model.eval()
    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("T-Revision training split is empty") from exc
    with torch.inference_mode():
        probe = first_batch["input"].to(device)
        probe_logits = model(probe)
    model.train(was_training)
    if not torch.is_tensor(probe_logits) or probe_logits.ndim != 2:
        raise ValueError("T-Revision model output must have shape [B, C]")
    if probe_logits.shape != (probe.shape[0], num_classes):
        raise ValueError(
            "T-Revision model output class dimension does not match dataset "
            f"class count {num_classes}: got {tuple(probe_logits.shape)}"
        )
    if not bool(torch.isfinite(probe_logits).all().item()):
        raise ValueError("T-Revision model preflight logits must be finite")


def run_t_revision_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run the paper-faithful Reweight-R workflow with explicit code choices."""

    config = deepcopy(config)
    method_config = TRevisionConfig.from_mapping(config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    trainer_config = config.get("trainer", {}) or {}
    if not isinstance(trainer_config, Mapping):
        raise TypeError("trainer configuration must be a mapping")
    device = resolve_device(trainer_config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = None
    if resume is not None:
        checkpoint_payload = read_checkpoint(resume, "cpu")
        if checkpoint_payload.get("method") != "t_revision":
            raise ValueError("Resume checkpoint is not a T-Revision run")

    data_config = config["data"]
    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.TRAIN_EVAL, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir,
        seed=seed,
        checkpoint_payload=checkpoint_payload,
    )
    dataset_name, num_classes = prepared.dataset, prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None:
        raise ValueError("T-Revision requires an enabled noisy-label manifest")
    if manifest.num_classes != num_classes:
        raise ValueError("T-Revision noise manifest class count mismatch")
    if manifest.transition_matrix is None or manifest.per_sample_transition is not None:
        raise ValueError(
            "T-Revision requires one fixed class-dependent transition matrix; "
            "sample-dependent transitions are unsupported"
        )

    manifest_position = {
        int(index): position
        for position, index in enumerate(manifest.global_indices.tolist())
    }
    observed_train_targets = np.asarray(
        [manifest.noisy_targets[manifest_position[int(index)]] for index in prepared.train_indices],
        dtype=np.int64,
    )
    observed_classes = np.unique(observed_train_targets)
    if not np.array_equal(observed_classes, np.arange(num_classes)):
        missing = np.setdiff1d(np.arange(num_classes), observed_classes).tolist()
        raise ValueError(
            "T-Revision noisy training split is missing observed classes: "
            f"{missing}"
        )
    train_loader = prepared.loader(DataRole.TRAIN)
    posterior_loader = prepared.loader(DataRole.TRAIN_EVAL, shuffle=False)
    validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    test_loader = prepared.loader(DataRole.TEST, shuffle=False)

    effective_train_rate = effective_subset_actual_rate(manifest, prepared.train_indices)
    effective_validation_rate = effective_subset_actual_rate(manifest, prepared.validation_indices)
    noise_metadata = checkpoint_noise_metadata(
        manifest,
        manifest_path,
        run_dir,
        effective_train_rate,
        mode=noise_mode(config),
        validation_targets="noisy",
        effective_validation_rate=effective_validation_rate,
    )
    config["noise"] = _resolved_noise_config(config["noise"], noise_metadata)

    model = build_model(method_config.model, num_classes).to(device)
    _preflight_model_output(model, posterior_loader, device, num_classes)
    stage1_optimizer = build_optimizer(model, method_config.stage1.optimizer)
    stage1_scheduler = build_scheduler(
        stage1_optimizer,
        method_config.stage1.scheduler,
        method_config.stage1.epochs,
    )

    def classifier_optimizer_factory(value: nn.Module) -> torch.optim.Optimizer:
        return build_optimizer(value, method_config.classifier_initialization.optimizer)

    def classifier_scheduler_factory(optimizer: torch.optim.Optimizer) -> Any:
        return build_scheduler(
            optimizer,
            method_config.classifier_initialization.scheduler,
            method_config.classifier_initialization.epochs,
        )

    def revision_optimizer_factory(
        value: nn.Module, revision: nn.Module
    ) -> torch.optim.Optimizer:
        container = nn.ModuleList([value, revision])
        return build_optimizer(container, method_config.revision.optimizer)

    def revision_scheduler_factory(optimizer: torch.optim.Optimizer) -> Any:
        return build_scheduler(
            optimizer,
            method_config.revision.scheduler,
            method_config.revision.epochs,
        )

    criterion = build_builtin_loss({"name": "ce"}).to(device)
    if resume is None:
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (run_dir / "environment.json").write_text(
            json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
        )
        (run_dir / "noise_summary.json").write_text(
            json.dumps(noise_metadata, indent=2), encoding="utf-8"
        )

    algorithm = TRevisionAlgorithm(
        model=model,
        stage1_optimizer=stage1_optimizer,
        stage1_scheduler=stage1_scheduler,
        classifier_optimizer_factory=classifier_optimizer_factory,
        classifier_scheduler_factory=classifier_scheduler_factory,
        revision_optimizer_factory=revision_optimizer_factory,
        revision_scheduler_factory=revision_scheduler_factory,
        loss=criterion,
        train_loader=train_loader,
        posterior_loader=posterior_loader,
        noisy_validation_loader=validation_loader,
        clean_test_loader=test_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        dataset=dataset_name,
        num_classes=num_classes,
        noise_metadata=noise_metadata,
        diagnostic_transition=manifest.transition_matrix,
    )
    try:
        if resume is not None:
            algorithm.resume(resume)
        algorithm.run()
    finally:
        algorithm.close()
    return run_dir
