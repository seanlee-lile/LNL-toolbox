from __future__ import annotations

"""Public experiment service shared by CLI, Python API, and sweeps."""

import json
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any, Mapping

from lnl_toolbox.training.results import finalize_result, is_completed_result
from lnl_toolbox.training.runners import resolve_runner

if TYPE_CHECKING:
    from lnl_toolbox.training.data_service import DataService


class ExperimentService:
    def __init__(self, data_service: "DataService | None" = None) -> None:
        if data_service is None:
            from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE

            data_service = DEFAULT_DATA_SERVICE
        self.data_service = data_service

    def preflight(
        self,
        config: Mapping[str, Any],
        *,
        check_data: bool = True,
    ):
        """Validate one resolved experiment without creating run artifacts."""

        from lnl_toolbox.catalog import validate_config

        runner = validate_config(config, check_data=False)
        if check_data:
            self.data_service.validate_config(config)
        return runner

    def _ensure_metadata(self, run_dir: Path, config: Mapping[str, Any]) -> None:
        resolved = run_dir / "resolved_config.yaml"
        if not resolved.is_file():
            import yaml

            resolved.write_text(
                yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
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
        candidate = dict(config)
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
