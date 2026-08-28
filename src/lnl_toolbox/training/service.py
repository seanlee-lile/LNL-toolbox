from __future__ import annotations

"""Public experiment service shared by CLI, Python API, and sweeps."""

import json
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any, Mapping

from lnl_toolbox.core.config_schema import (
    normalize_experiment_config,
    runtime_experiment_config,
)
from lnl_toolbox.training.results import finalize_result, is_completed_result
from lnl_toolbox.data.profile import NoiseRateInfo, NoiseRateStatus
from lnl_toolbox.training.compatibility import (
    CompatibilityReason,
    CompatibilityResult,
    CompatibilityStatus,
    build_input_guidance,
    requirements_unavailable_result,
    resolve_compatibility,
)
from lnl_toolbox.training.runners import resolve_runner, runner_specs

if TYPE_CHECKING:
    from lnl_toolbox.training.data_service import DataService


class ExperimentService:
    def __init__(self, data_service: "DataService | None" = None) -> None:
        if data_service is None:
            from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE

            data_service = DEFAULT_DATA_SERVICE
        self.data_service = data_service
        self.last_compatibility: CompatibilityResult | None = None
        self._last_compatibility_config: dict[str, Any] | None = None

    @staticmethod
    def _config_value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = config
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _config_value_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return bool(value)
        return True

    def inspect_dataset(self, source: object, *, seed: int = 0):
        """Inspect without persisting catalog state or creating artifacts."""

        return self.data_service.inspect(source, seed=seed, persist=False)

    def _resolve_for_capabilities(
        self,
        capabilities,
        runner,
        config: Mapping[str, Any],
        *,
        method_noise_rate_prior: float | None = None,
    ) -> CompatibilityResult:
        requirements = runner.requirements(config)
        if requirements is None:
            result = requirements_unavailable_result(runner.name, capabilities.dataset)
            return replace(result, input_guidance=build_input_guidance(result))

        prior = method_noise_rate_prior
        prior_source = "compatibility API input"
        if prior is None:
            for path in requirements.method_noise_prior_paths:
                value = self._config_value(config, path)
                if value is not None:
                    prior = float(value)
                    prior_source = "experiment config:" + ".".join(path)
                    break
        prior_info = None if prior is None else NoiseRateInfo(
            NoiseRateStatus.KNOWN, prior, prior_source
        )
        pretrained_roles = self._available_pretrained_roles(config, requirements)
        result = resolve_compatibility(
            capabilities,
            requirements,
            method_noise_rate_prior=prior_info,
            available_pretrained_roles=pretrained_roles,
        )
        missing_inputs = []
        for required in requirements.required_config_inputs:
            present = [
                self._config_value_present(self._config_value(config, path))
                for path in required.paths
            ]
            satisfied = all(present) if required.mode == "all" else any(present)
            if not satisfied:
                missing_inputs.append(required)
        if missing_inputs:
            status = (
                result.status
                if result.status is CompatibilityStatus.INCOMPATIBLE
                else CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS
            )
            result = CompatibilityResult(
                status=status,
                method=result.method,
                dataset=result.dataset,
                reasons=result.reasons + tuple(
                    CompatibilityReason(item.code, item.description)
                    for item in missing_inputs
                ),
                warnings=result.warnings,
                required_user_inputs=tuple(sorted(
                    set(result.required_user_inputs).union(
                        f"config:{item.code}" for item in missing_inputs
                    )
                )),
                required_input_paths=tuple(
                    list(result.required_input_paths)
                    + [
                        (f"config:{item.code}", item.paths)
                        for item in missing_inputs
                    ]
                ),
            )
        environments = {
            f"pretrained:{role}": str(self._config_value(config, path)).strip()
            for role, path in requirements.pretrained_role_paths
            if path[-1] == "run_directory_env"
            and self._config_value(config, path) is not None
        }
        return replace(
            result,
            input_guidance=build_input_guidance(
                result, environment_variables=environments,
            ),
        )

    @staticmethod
    def _set_config_value(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        current = config
        for part in path[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[path[-1]] = value

    @classmethod
    def _inject_method_noise_prior(
        cls,
        config: Mapping[str, Any],
        runner,
        prior: float | None,
    ) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        if prior is None:
            return candidate
        requirements = runner.requirements(candidate)
        if requirements is None or not requirements.requires_method_noise_prior:
            return candidate
        if not requirements.method_noise_prior_paths:
            raise ValueError(
                f"method {requirements.method!r} declares a noise-rate prior "
                "but does not declare method_noise_prior_paths"
            )
        for path in requirements.method_noise_prior_paths:
            cls._set_config_value(candidate, path, float(prior))
        return candidate

    @staticmethod
    def _available_pretrained_roles(
        config: Mapping[str, Any], requirements,
    ) -> frozenset[str]:
        available: set[str] = set()
        for role, path in requirements.pretrained_role_paths:
            value = ExperimentService._config_value(config, path)
            if isinstance(value, Mapping):
                value = value.get("path") or value.get("checkpoint")
            if value is None:
                continue
            text = str(value).strip()
            if not text or text == role:
                continue
            if path[-1] == "run_directory_env":
                text = os.environ.get(text, "").strip()
                if not text:
                    continue
            candidate = Path(text).expanduser()
            if candidate.is_file() or candidate.is_dir():
                available.add(role)
        return frozenset(available)

    def resolve_method_compatibility(
        self,
        dataset: object,
        method: str | Mapping[str, Any],
        *,
        method_noise_rate_prior: float | None = None,
    ) -> CompatibilityResult:
        """Resolve one dataset/method pair without model or runner execution."""

        config = (
            dict(method)
            if isinstance(method, Mapping)
            else {"execution": {"runner": str(method)}}
        )
        source = config if isinstance(method, Mapping) else dataset
        capabilities = self.data_service.capabilities(
            source, seed=int(config.get("seed", 0)), persist=False
        )
        runner = resolve_runner(config)
        config = self._inject_method_noise_prior(config, runner, method_noise_rate_prior)
        return self._resolve_for_capabilities(
            capabilities,
            runner,
            config,
            method_noise_rate_prior=method_noise_rate_prior,
        )

    def list_compatible_methods(
        self,
        dataset: object,
        *,
        method_noise_rate_prior: float | None = None,
    ) -> tuple[CompatibilityResult, ...]:
        """Compare one inspected dataset with every central registry runner."""

        capabilities = self.data_service.capabilities(dataset, persist=False)
        return tuple(
            self._resolve_for_capabilities(
                capabilities,
                runner,
                {},
                method_noise_rate_prior=method_noise_rate_prior,
            )
            for runner in runner_specs()
        )

    def list_config_compatibility(
        self,
        dataset: object,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        method_noise_rate_prior: float | None = None,
    ) -> tuple[tuple[str, CompatibilityResult], ...]:
        """Resolve concrete configurations while loading dataset capabilities once."""

        capabilities = self.data_service.capabilities(dataset, persist=False)
        results: list[tuple[str, CompatibilityResult]] = []
        for key, config in configs.items():
            candidate = self.data_service.apply(config, dataset)
            runner = resolve_runner(candidate)
            candidate = self._inject_method_noise_prior(
                candidate, runner, method_noise_rate_prior
            )
            results.append((
                str(key),
                self._resolve_for_capabilities(
                    capabilities,
                    runner,
                    candidate,
                    method_noise_rate_prior=method_noise_rate_prior,
                ),
            ))
        return tuple(results)

    @staticmethod
    def _enforce_compatibility(result: CompatibilityResult) -> None:
        if result.status is CompatibilityStatus.COMPATIBLE:
            return
        details = "; ".join(
            f"{item.code}: {item.message}" for item in result.reasons
        )
        required = (
            "; required_user_inputs=" + ",".join(result.required_user_inputs)
            if result.required_user_inputs else ""
        )
        raise ValueError(
            f"method {result.method!r} is not ready for dataset {result.dataset!r}: "
            f"{result.status.value}; {details}{required}"
        )

    def _compatibility_preflight(
        self,
        candidate: Mapping[str, Any],
        runner,
    ) -> CompatibilityResult | None:
        requirements = getattr(runner, "requirements", None)
        if not callable(requirements) or requirements(candidate) is None:
            self.last_compatibility = None
            self._last_compatibility_config = dict(candidate)
            return None
        if self._last_compatibility_config == dict(candidate):
            result = self.last_compatibility
        else:
            result = self.resolve_method_compatibility(candidate, candidate)
            self.last_compatibility = result
            self._last_compatibility_config = dict(candidate)
        assert result is not None
        self._enforce_compatibility(result)
        return result

    def preflight(
        self,
        config: Mapping[str, Any],
        *,
        check_data: bool = True,
    ):
        """Validate one resolved experiment without creating run artifacts."""

        from lnl_toolbox.catalog import validate_config

        candidate = runtime_experiment_config({"kind": "experiment", **dict(config)})
        if candidate.get("kind") != "experiment":
            raise ValueError("ExperimentService requires kind: experiment")
        runner = validate_config(candidate, check_data=False)
        if check_data:
            self.data_service.validate_config(candidate)
            self._compatibility_preflight(candidate, runner)
        else:
            self.last_compatibility = None
            self._last_compatibility_config = None
        return runner

    def _ensure_metadata(self, run_dir: Path, config: Mapping[str, Any]) -> None:
        resolved = run_dir / "resolved_config.yaml"
        if not resolved.is_file():
            import yaml

            resolved.write_text(
                yaml.safe_dump(
                    normalize_experiment_config(config), sort_keys=False
                ),
                encoding="utf-8",
            )
        environment = run_dir / "environment.json"
        if not environment.is_file():
            environment.write_text(
                json.dumps(
                    {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "seed": int(config.get("seed", 1)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def run(
        self,
        config: Mapping[str, Any],
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
        *,
        recipe: str | None = None,
        completed_noop: bool = True,
    ) -> Path:
        # Preserve the public dispatch contract even for otherwise incomplete
        # configurations: an unknown explicit method must fail as an unknown
        # method before schema validation reports missing dataset fields.
        explicit_method = config.get("method")
        if isinstance(explicit_method, str):
            resolve_runner({"method": explicit_method})
        candidate = runtime_experiment_config({"kind": "experiment", **dict(config)})
        if candidate.get("kind") != "experiment":
            raise ValueError("ExperimentService requires kind: experiment")
        runner = resolve_runner(candidate)
        expected_root = None
        if resume is not None:
            expected_root = Path(resume).expanduser().resolve().parent
        elif output_dir is not None:
            expected_root = Path(output_dir).expanduser().resolve()
        final_path = expected_root / "final_metrics.json" if expected_root else None
        before_final = final_path.read_bytes() if final_path and final_path.is_file() else None
        returned = Path(runner.invoke(candidate, output_dir, resume))
        result = returned.resolve()
        if not result.is_dir():
            return returned
        result_final = result / "final_metrics.json"
        if (
            completed_noop
            and before_final is not None
            and result_final.is_file()
            and result_final.read_bytes() == before_final
        ):
            return returned
        self._ensure_metadata(result, candidate)
        try:
            finalize_result(result, candidate, runner=runner.name, recipe=recipe)
        except ValueError as exc:
            if "does not expose a finite primary test metric" not in str(exc):
                raise
        return returned

    def resume(self, run_dir: str | Path, checkpoint: str = "last") -> Path:
        root = Path(run_dir).expanduser().resolve()
        config_path = root / "resolved_config.yaml"
        if not config_path.is_file():
            raise ValueError(f"run directory is missing resolved_config.yaml: {root}")
        if is_completed_result(root):
            return root
        checkpoint_path = root / f"{checkpoint}.pt"
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
        import yaml

        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"resolved configuration must be a mapping: {config_path}")
        return self.run(
            dict(value), root, checkpoint_path, completed_noop=True
        )


__all__ = ["ExperimentService"]
