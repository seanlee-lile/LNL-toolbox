from __future__ import annotations

"""CIFAR experiment assembly for the paper-specific Dual-T + Forward method."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from lnl_toolbox.algorithms.dual_t import (
    DualTAlgorithm,
    DualTConfig,
)
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


def run_dual_t_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Build and run the first paper-specific Dual-T + Forward workflow."""

    config = deepcopy(config)
    method_config = DualTConfig.from_mapping(config)
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
        run_dir = Path(
            config.get("output_root", "artifacts/runs")
        ) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = None
    if resume is not None:
        checkpoint_payload = read_checkpoint(resume, "cpu")
        if checkpoint_payload.get("method") == "dual_t_forward":
            raise ValueError("method 'dual_t_forward' was renamed to 'dual_t'")
        if checkpoint_payload.get("method") != "dual_t":
            raise ValueError("Resume checkpoint is not a Dual-T run")

    prepared = prepare_experiment_data(
        config,
        requirements=DataRequirements(
            roles=frozenset({DataRole.TRAIN, DataRole.NOISY_VALIDATION, DataRole.TEST}),
            validation_targets="noisy",
        ),
        run_dir=run_dir,
        seed=seed,
        checkpoint_payload=checkpoint_payload,
    )
    dataset_name = prepared.dataset
    num_classes = prepared.num_classes
    manifest, manifest_path = prepared.manifest, prepared.manifest_path
    if manifest is None or manifest_path is None:
        raise ValueError("Dual-T requires an enabled noisy-label manifest")
    train_loader = prepared.loader(DataRole.TRAIN)
    noisy_validation_loader = prepared.loader(DataRole.NOISY_VALIDATION, shuffle=False)
    clean_test_loader = prepared.loader(DataRole.TEST, shuffle=False)

    effective_train_rate = effective_subset_actual_rate(
        manifest, prepared.train_indices
    )
    effective_validation_rate = effective_subset_actual_rate(
        manifest, prepared.validation_indices
    )
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

    posterior_model = build_model(
        method_config.posterior_stage.model, num_classes
    )
    posterior_optimizer = build_optimizer(
        posterior_model, method_config.posterior_stage.optimizer
    )
    posterior_scheduler = build_scheduler(
        posterior_optimizer,
        method_config.posterior_stage.scheduler,
        method_config.posterior_stage.epochs,
    )
    final_model = build_model(method_config.final_stage.model, num_classes)
    final_optimizer = build_optimizer(
        final_model, method_config.final_stage.optimizer
    )
    final_scheduler = build_scheduler(
        final_optimizer,
        method_config.final_stage.scheduler,
        method_config.final_stage.epochs,
    )
    posterior_loss = build_builtin_loss(
        dict(config["posterior_stage"]).get("loss", {"name": "ce"})
    ).to(device)
    final_loss = build_builtin_loss(
        dict(config["final_stage"]).get("loss", {"name": "ce"})
    ).to(device)

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(seed, device), indent=2), encoding="utf-8"
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps(noise_metadata, indent=2), encoding="utf-8"
    )

    algorithm = DualTAlgorithm(
        posterior_model=posterior_model,
        posterior_optimizer=posterior_optimizer,
        posterior_scheduler=posterior_scheduler,
        final_model=final_model,
        final_optimizer=final_optimizer,
        final_scheduler=final_scheduler,
        posterior_loss=posterior_loss,
        final_loss=final_loss,
        train_loader=train_loader,
        noisy_validation_loader=noisy_validation_loader,
        clean_test_loader=clean_test_loader,
        device=device,
        run_dir=run_dir,
        config=config,
        dataset=dataset_name,
        noise_metadata=noise_metadata,
    )
    try:
        if resume is not None:
            algorithm.resume(resume)
        algorithm.run()
    finally:
        algorithm.close()
    return run_dir
