from __future__ import annotations

"""Experiment assembly for binary asymmetric-RCN Importance Reweighting."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from lnl_toolbox.algorithms.importance_reweighting import (
    ImportanceReweightingAlgorithm,
    ImportanceReweightingConfig,
)
from lnl_toolbox.data.binary_synthetic import (
    BinaryTensorDataset,
    generate_synthetic_binary_2d,
    generate_synthetic_binary_high_dim,
)
from lnl_toolbox.data.binary_benchmarks import BinaryBenchmarkTensorDataset
from lnl_toolbox.data.preprocessing import (
    BinaryPreprocessingConfig,
    BinaryPreprocessor,
)
from lnl_toolbox.noise.binary_rcn import (
    generate_binary_asymmetric_rcn,
    validate_binary_rcn_manifest,
)
from lnl_toolbox.noise.manifest import NoiseManifest
from lnl_toolbox.plugins.builtin import build_builtin_loss
from lnl_toolbox.runtime import resolve_device, seed_everything
from lnl_toolbox.training.experiment import (
    _environment,
    _loader,
    build_optimizer,
    build_scheduler,
)


def run_importance_reweighting_experiment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run the paper-scoped binary KDE/raw-min/weighted-CE workflow."""

    config = deepcopy(config)
    method_config = ImportanceReweightingConfig.from_mapping(config)
    seed_everything(method_config.seed)
    device = resolve_device(str(method_config.trainer.get("device", "cpu")))

    if resume is not None:
        run_dir = Path(resume).resolve().parent
    elif output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        run_dir = Path(config.get("output_root", "artifacts/runs")) / (
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    sizes = method_config.data
    data_name = str(sizes["name"]).strip().lower()
    dimension = int(sizes["dimension"])
    preprocessing_identity: dict[str, Any] = {}
    split_identity: dict[str, Any] = {}
    raw_sha256 = ""
    source_stat: tuple[int, int] | None = None
    if data_name == "uci_statlog_heart":
        source = Path(str(sizes["path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"UCI Statlog Heart file is missing: {source}; run "
                "python scripts/prepare_uci_statlog_heart.py"
            )
        raw_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if raw_sha256 != str(sizes["sha256"]):
            raise ValueError("UCI Statlog Heart raw-file SHA-256 mismatch")
        initial_stat = source.stat()
        source_stat = (int(initial_stat.st_size), int(initial_stat.st_mtime_ns))
        preprocessing_config = BinaryPreprocessingConfig.from_mapping(
            sizes["preprocessing"]
        )
        probe = BinaryPreprocessor(preprocessing_config)
        targets = probe.read_targets(source)
        if targets.size != int(sizes["expected_samples"]):
            raise ValueError("UCI Statlog Heart sample count mismatch")
        rng = np.random.default_rng(int(sizes["split"]["seed"]))
        split_rows: dict[str, list[np.ndarray]] = {
            "train": [], "validation": [], "test": []
        }
        for label in (0, 1):
            rows = np.flatnonzero(targets == label)
            rows = rows[rng.permutation(rows.size)]
            validation_count = max(1, int(round(
                rows.size * float(sizes["split"]["validation_fraction"])
            )))
            test_count = max(1, int(round(
                rows.size * float(sizes["split"]["test_fraction"])
            )))
            if validation_count + test_count >= rows.size:
                raise ValueError("UCI split fractions leave no training samples")
            split_rows["validation"].append(rows[:validation_count])
            split_rows["test"].append(
                rows[validation_count:validation_count + test_count]
            )
            split_rows["train"].append(rows[validation_count + test_count:])
        train_rows = np.sort(np.concatenate(split_rows["train"]))
        validation_rows = np.sort(np.concatenate(split_rows["validation"]))
        test_rows = np.sort(np.concatenate(split_rows["test"]))
        processor = BinaryPreprocessor(preprocessing_config).fit(
            source, row_indices=train_rows
        )
        train = processor.transform(
            source, dataset=data_name, split="train", row_indices=train_rows
        )
        validation = processor.transform(
            source, dataset=data_name, split="validation", row_indices=validation_rows
        )
        test = processor.transform(
            source, dataset=data_name, split="test", row_indices=test_rows
        )
        preprocessing_identity = json.loads(json.dumps(processor.state_dict()))
        state_hash = hashlib.sha256(json.dumps(
            preprocessing_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        preprocessing_identity["state_hash"] = state_hash
        split_payload = {
            "seed": int(sizes["split"]["seed"]),
            "validation_fraction": float(sizes["split"]["validation_fraction"]),
            "test_fraction": float(sizes["split"]["test_fraction"]),
            "train_indices": train_rows.tolist(),
            "validation_indices": validation_rows.tolist(),
            "test_indices": test_rows.tolist(),
        }
        split_identity = {
            **split_payload,
            "split_hash": hashlib.sha256(json.dumps(
                split_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        }
        identity_files = {
            run_dir / "preprocessing_state.json": preprocessing_identity,
            run_dir / "split_manifest.json": split_identity,
        }
        for identity_path, identity in identity_files.items():
            if resume is not None:
                if not identity_path.is_file():
                    raise FileNotFoundError(
                        f"resume requires {identity_path.name}"
                    )
                try:
                    stored_identity = json.loads(
                        identity_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    raise ValueError(
                        f"resume {identity_path.name} is invalid"
                    ) from exc
                if stored_identity != identity:
                    raise ValueError(
                        f"resume {identity_path.name} identity mismatch"
                    )
            else:
                temporary = identity_path.with_suffix(
                    identity_path.suffix + ".pending"
                )
                temporary.write_text(
                    json.dumps(identity, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if json.loads(temporary.read_text(encoding="utf-8")) != identity:
                    raise ValueError(f"failed to validate {identity_path.name}")
                temporary.replace(identity_path)
    else:
        generator = (
            generate_synthetic_binary_2d
            if data_name == "synthetic_binary_2d"
            else generate_synthetic_binary_high_dim
        )
        dimension_args = () if data_name == "synthetic_binary_2d" else (dimension,)
        train = generator(
            int(sizes["train_size"]), *dimension_args, method_config.seed + 11,
            start_index=0, split="train",
        )
        validation = generator(
            int(sizes["validation_size"]), *dimension_args, method_config.seed + 12,
            start_index=len(train.labels), split="validation",
        )
        test = generator(
            int(sizes["test_size"]), *dimension_args, method_config.seed + 13,
            start_index=len(train.labels) + len(validation.labels), split="test",
        )
    train_clean = train.targets if data_name == "uci_statlog_heart" else train.labels
    validation_clean = (
        validation.targets
        if data_name == "uci_statlog_heart"
        else validation.labels
    )
    population_clean = np.concatenate((train_clean, validation_clean))
    population_indices = np.concatenate((
        train.global_indices, validation.global_indices
    ))
    manifest_path = run_dir / "noise_manifest.npz"
    if resume is None:
        manifest = generate_binary_asymmetric_rcn(
            population_clean,
            population_indices,
            rho_positive=float(method_config.noise["rho_positive"]),
            rho_negative=float(method_config.noise["rho_negative"]),
            seed=int(method_config.noise.get("seed", method_config.seed + 21)),
            dataset=data_name,
        )
        manifest.save(manifest_path)
    else:
        if not manifest_path.exists():
            raise FileNotFoundError("resume requires noise_manifest.npz")
        manifest = NoiseManifest.load(manifest_path)
    validate_binary_rcn_manifest(
        manifest,
        expected_indices=population_indices,
        rho_positive=float(method_config.noise["rho_positive"]),
        rho_negative=float(method_config.noise["rho_negative"]),
    )
    if not np.array_equal(manifest.clean_targets, population_clean):
        raise ValueError("importance reweighting manifest clean-label identity mismatch")
    positive_rows = manifest.clean_targets == 1
    negative_rows = manifest.clean_targets == 0
    realized_rho_positive = float(np.mean(
        manifest.noisy_targets[positive_rows] == 0
    ))
    realized_rho_negative = float(np.mean(
        manifest.noisy_targets[negative_rows] == 1
    ))

    position = {
        int(index): row
        for row, index in enumerate(manifest.global_indices.tolist())
    }
    def noisy_targets(indices: np.ndarray) -> np.ndarray:
        try:
            rows = np.asarray([position[int(index)] for index in indices])
        except KeyError as exc:
            raise ValueError("noise manifest is missing a stable sample index") from exc
        return manifest.noisy_targets[rows]

    train_noisy = noisy_targets(train.global_indices)
    validation_noisy = noisy_targets(validation.global_indices)
    dataset_class = (
        BinaryBenchmarkTensorDataset
        if data_name == "uci_statlog_heart"
        else BinaryTensorDataset
    )
    train_dataset = dataset_class(train, train_noisy)
    validation_dataset = dataset_class(validation, validation_noisy)
    test_dataset = dataset_class(test)
    loader_config = method_config.loader

    def train_loader_factory(epoch: int):
        return _loader(
            train_dataset,
            loader_config,
            shuffle=True,
            seed=method_config.seed + 1000 + int(epoch),
        )

    validation_loader = _loader(
        validation_dataset, loader_config, shuffle=False,
        seed=method_config.seed + 2000,
    )
    test_loader = _loader(
        test_dataset, loader_config, shuffle=False,
        seed=method_config.seed + 3000,
    )

    def model_factory() -> nn.Module:
        model = nn.Linear(dimension, 2)
        if model.out_features != 2:
            raise ValueError("importance reweighting model must output two logits")
        return model

    def optimizer_factory(model: nn.Module):
        return build_optimizer(model, method_config.optimizer)

    def scheduler_factory(optimizer):
        return build_scheduler(
            optimizer, method_config.scheduler, method_config.epochs
        )

    manifest_identity = {
        "dataset": manifest.dataset,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "mapping_hash": manifest.mapping_hash,
        "noise_type": manifest.noise_type,
        "num_classes": manifest.num_classes,
        "label_convention": manifest.metadata.get("label_convention"),
        "raw_sha256": raw_sha256,
        "preprocessing_state_hash": preprocessing_identity.get("state_hash", ""),
        "split_hash": split_identity.get("split_hash", ""),
        "realized_rho_positive": realized_rho_positive,
        "realized_rho_negative": realized_rho_negative,
    }
    algorithm = ImportanceReweightingAlgorithm(
        config=config,
        run_dir=run_dir,
        manifest_identity=manifest_identity,
        posterior_features=train.features,
        posterior_targets=train_noisy,
        posterior_indices=train.global_indices,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        train_loader_factory=train_loader_factory,
        validation_loader=validation_loader,
        test_loader=test_loader,
        loss=build_builtin_loss(method_config.loss),
        device=device,
    )
    if resume is not None:
        algorithm.resume(resume)

    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(_environment(method_config.seed, device), indent=2),
        encoding="utf-8",
    )
    (run_dir / "noise_summary.json").write_text(
        json.dumps({
            "rho_positive": float(method_config.noise["rho_positive"]),
            "rho_negative": float(method_config.noise["rho_negative"]),
            "manifest_actual_rate": manifest.actual_rate,
            "mapping_hash": manifest.mapping_hash,
            "raw_sha256": raw_sha256,
            "preprocessing_state_hash": preprocessing_identity.get("state_hash", ""),
            "split_hash": split_identity.get("split_hash", ""),
            "realized_rho_positive": realized_rho_positive,
            "realized_rho_negative": realized_rho_negative,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = algorithm.run()
    if data_name == "uci_statlog_heart":
        final_stat = source.stat()
        if (
            source_stat != (int(final_stat.st_size), int(final_stat.st_mtime_ns))
            or hashlib.sha256(source.read_bytes()).hexdigest() != raw_sha256
        ):
            raise ValueError("UCI Statlog Heart source changed during the run")
    return result
