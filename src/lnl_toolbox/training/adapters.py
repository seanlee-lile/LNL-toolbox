from __future__ import annotations

"""Compatibility adapters that expose legacy functions through one protocol."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .interfaces import EvaluationResult, RunContext, RunResult
from .reporting import RunSession, load_metric_events, write_run_report


class LegacyRunnerAdapter:
    """Wrap one existing runner function without changing its training semantics."""

    def __init__(self, spec: Any, *, method: str | None = None) -> None:
        self.spec = spec
        self.method = method or spec.name
        self._loaded_resume: Path | None = None

    def prepare(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunContext:
        if context is not None:
            return context
        if config is None:
            raise TypeError("config is required when context is not supplied")
        from lnl_toolbox.catalog import resolve_config_paths, find_project_root

        raw = dict(config)
        resolved = resolve_config_paths(raw, find_project_root())
        run_dir = self._run_dir(resolved, output_dir, resume)
        return RunContext(
            config=raw,
            resolved_config=resolved,
            run_dir=run_dir,
            session=RunSession(
                run_dir,
                config=resolved,
                runner=self.spec.name,
                method=self.method,
                resumed=resume is not None,
            ),
            device=resolved.get("trainer", {}).get("device", "auto")
            if isinstance(resolved.get("trainer", {}), Mapping)
            else "auto",
            seed=int(resolved.get("seed", 0)),
        )

    @staticmethod
    def _run_dir(
        config: Mapping[str, Any],
        output_dir: str | Path | None,
        resume: str | Path | None,
    ) -> Path:
        if resume is not None:
            path = Path(resume).expanduser().resolve()
            return path.parent if path.suffix else path
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve()
        output_root = Path(str(config.get("output_root", "artifacts/runs")))
        return (output_root / "unified-run").expanduser().resolve()

    def fit(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult:
        ctx = self.prepare(context, config=config, output_dir=output_dir, resume=resume)
        effective_resume = Path(resume).expanduser().resolve() if resume is not None else self._loaded_resume
        runner = self.spec.load()
        try:
            if effective_resume is not None and self.spec.supports_resume:
                self._prepare_resume_log(ctx.run_dir, effective_resume, ctx.resolved_config)
            result = (
                runner(dict(ctx.resolved_config), ctx.run_dir, effective_resume)
                if self.spec.supports_resume
                else runner(dict(ctx.resolved_config), ctx.run_dir)
            )
            run_dir = Path(result).expanduser()
            self._normalize_legacy_metrics(run_dir)
            write_run_report(
                run_dir,
                config=ctx.resolved_config,
                runner=self.spec.name,
                method=self.method,
                status="completed",
            )
            return RunResult.from_run_dir(run_dir, resolve=False)
        except BaseException as exc:
            try:
                ctx.session.fail_run(exc)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            raise

    @staticmethod
    def _prepare_resume_log(
        run_dir: Path,
        checkpoint: Path,
        config: Mapping[str, Any],
    ) -> None:
        """Remove a stale final event only when the legacy runner will continue."""

        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            return
        trainer = config.get("trainer", {})
        target_epochs = trainer.get("epochs") if isinstance(trainer, Mapping) else None
        if target_epochs is None:
            target_epochs = config.get("epochs")
        if target_epochs is None:
            return
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            completed_epoch = int(payload.get("completed_epoch", -1))
            if completed_epoch + 1 >= int(target_epochs):
                return
        except (ImportError, OSError, TypeError, ValueError, RuntimeError):
            return
        events = load_metric_events(metrics_path)
        while events and events[-1].get("event") == "final":
            events.pop()
        metrics_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events),
            encoding="utf-8",
        )

    @staticmethod
    def _prepare_native_resume_log(run_dir: Path) -> None:
        """Keep committed training events and remove the prior phase tail."""

        metrics_path = run_dir / "metrics.jsonl"
        events = load_metric_events(metrics_path)
        while events and events[-1].get("event") in {"phase_end", "final"}:
            events.pop()
        metrics_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_legacy_metrics(run_dir: Path) -> None:
        """Bridge legacy ``metrics.json`` outputs into authoritative JSONL."""

        target = run_dir / "metrics.jsonl"
        source = run_dir / "metrics.json"
        events: list[dict[str, Any]] = []
        if target.is_file():
            try:
                events = load_metric_events(target)
            except (OSError, ValueError, json.JSONDecodeError):
                return
        elif source.is_file():
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if not isinstance(value, list):
                return
            for sequence, item in enumerate(value):
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("event", "epoch")
                    row.setdefault("seq", sequence)
                    row.setdefault("phase", "train")
                    events.append(row)
        if not events:
            return
        normalized: list[dict[str, Any]] = []
        for row in events:
            event = str(row.get("event", "metric"))
            unit = "epoch" if event == "epoch" else "step" if event == "step" else "run"
            completed = row.get("epoch", row.get("global_step"))
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                excluded = {"event", "seq", "phase", "unit", "completed", "total", "metrics", "artifacts"}
                metrics = {key: value for key, value in row.items() if key not in excluded}
            normalized.append({
                **row,
                "unit": row.get("unit", unit),
                "completed": row.get("completed", completed),
                "metrics": dict(metrics),
                "artifacts": list(row.get("artifacts", [])) if isinstance(row.get("artifacts", []), list) else [],
            })
        with target.open("w", encoding="utf-8") as handle:
            for row in normalized:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if normalized[-1].get("event") != "final":
                handle.write(
                    json.dumps(
                        {
                            **normalized[-1],
                            "event": "final",
                            "seq": len(normalized),
                            "phase": "evaluation",
                            "unit": "run",
                            "completed": normalized[-1].get("completed"),
                            "metrics": dict(normalized[-1].get("metrics", {})),
                            "artifacts": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        final_path = run_dir / "final_metrics.json"
        if not final_path.is_file():
            final_path.write_text(
                json.dumps(events[-1], indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def evaluate(self, result: RunResult) -> EvaluationResult:
        return EvaluationResult(metrics=dict(result.final_metrics), split="test")

    def save_checkpoint(self, result: RunResult, boundary: str = "last") -> Path:
        if boundary not in {"last", "best"}:
            raise ValueError("checkpoint boundary must be 'last' or 'best'")
        path = result.best_checkpoint if boundary == "best" else result.last_checkpoint
        if path is None:
            raise FileNotFoundError(f"run has no {boundary}.pt checkpoint: {result.run_dir}")
        return path

    def load_checkpoint(self, context: RunContext, path: str | Path) -> None:
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        context.state["resume_checkpoint"] = checkpoint
        self._loaded_resume = checkpoint


class NativeSingleStageRunner(LegacyRunnerAdapter):
    """Own the lifecycle for single-stage methods using the shared context."""

    @staticmethod
    def _will_continue(
        config: Mapping[str, Any],
        resume: Path | None,
    ) -> bool:
        if resume is None:
            return True
        trainer = config.get("trainer", {})
        target = trainer.get("epochs") if isinstance(trainer, Mapping) else None
        if target is None:
            target = config.get("epochs")
        if target is None:
            return True
        try:
            import torch

            payload = torch.load(resume, map_location="cpu", weights_only=False)
            return int(payload.get("completed_epoch", -1)) + 1 < int(target)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError):
            return True

    def fit(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult:
        ctx = self.prepare(context, config=config, output_dir=output_dir, resume=resume)
        effective_resume = (
            Path(resume).expanduser().resolve()
            if resume is not None
            else self._loaded_resume
        )
        ctx.state["lifecycle_active"] = self._will_continue(
            ctx.resolved_config, effective_resume
        )
        if effective_resume is not None and not ctx.state["lifecycle_active"]:
            return RunResult.from_run_dir(ctx.run_dir, resolve=False)
        if effective_resume is not None and self.spec.supports_resume and ctx.state["lifecycle_active"]:
            self._prepare_native_resume_log(ctx.run_dir)
            ctx.session._sequence = len(load_metric_events(ctx.run_dir / "metrics.jsonl"))
        ctx.state["resume_lifecycle"] = effective_resume is not None
        runner = self.spec.load()
        try:
            result = runner(
                dict(ctx.resolved_config),
                ctx.run_dir,
                effective_resume,
                context=ctx,
            )
            run_dir = Path(result).expanduser()
            write_run_report(
                run_dir,
                config=ctx.resolved_config,
                runner=self.spec.name,
                method=self.method,
                status="completed",
            )
            return RunResult.from_run_dir(run_dir, resolve=False)
        except BaseException as exc:
            try:
                ctx.session.fail_run(exc)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            raise


class GceRunner(NativeSingleStageRunner):
    pass


class AplRunner(NativeSingleStageRunner):
    pass


class CdrRunner(NativeSingleStageRunner):
    pass


class LossCorrectionRunner(NativeSingleStageRunner):
    pass


class NativeMultiModelRunner(LegacyRunnerAdapter):
    """Own the common lifecycle for jointly trained model groups."""

    def fit(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult:
        ctx = self.prepare(context, config=config, output_dir=output_dir, resume=resume)
        effective_resume = (
            Path(resume).expanduser().resolve()
            if resume is not None
            else self._loaded_resume
        )
        ctx.state["lifecycle_active"] = self._will_continue(
            ctx.resolved_config, effective_resume
        )
        if effective_resume is not None and not ctx.state["lifecycle_active"]:
            return RunResult.from_run_dir(ctx.run_dir, resolve=False)
        if effective_resume is not None and self.spec.supports_resume and ctx.state["lifecycle_active"]:
            self._prepare_native_resume_log(ctx.run_dir)
            ctx.session._sequence = len(load_metric_events(ctx.run_dir / "metrics.jsonl"))
        ctx.state["resume_lifecycle"] = effective_resume is not None
        runner = self.spec.load()
        try:
            result = runner(
                dict(ctx.resolved_config),
                ctx.run_dir,
                effective_resume,
                context=ctx,
            )
            run_dir = Path(result).expanduser()
            write_run_report(
                run_dir,
                config=ctx.resolved_config,
                runner=self.spec.name,
                method=self.method,
                status="completed",
            )
            return RunResult.from_run_dir(run_dir, resolve=False)
        except BaseException as exc:
            try:
                ctx.session.fail_run(exc)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            raise

    @staticmethod
    def _will_continue(
        config: Mapping[str, Any],
        resume: Path | None,
    ) -> bool:
        if resume is None:
            return True
        trainer = config.get("trainer", {})
        target = trainer.get("epochs") if isinstance(trainer, Mapping) else None
        if target is None:
            return True
        try:
            import torch

            payload = torch.load(resume, map_location="cpu", weights_only=False)
            return int(payload.get("completed_epoch", -1)) + 1 < int(target)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError):
            return True


class NativeStagedRunner(LegacyRunnerAdapter):
    """Expose existing staged algorithms through the shared phase context."""

    _RUNNER_OWNED_LIFECYCLE = frozenset({"dld", "upm", "lend"})

    @staticmethod
    def _is_completed_noop(
        spec: Any,
        config: Mapping[str, Any],
        resume: Path | None,
    ) -> bool:
        if spec.name not in {
            "dividemix", "volmin", "volminnet", "dld", "upm", "lend",
            "instance_transition", "cwd", "fine",
        } or resume is None:
            return False
        try:
            import torch

            payload = torch.load(resume, map_location="cpu", weights_only=False)
            if spec.name == "dividemix":
                state = payload["algorithm"]["dividemix_state"]
                target = int(config["dividemix"]["training"]["epochs"])
                return str(state.get("phase", "")) == "completed" and int(
                    state.get("main_completed_epochs", -1)
                ) >= target
            if spec.name in {"volmin", "volminnet"}:
                state = payload["algorithm"]["volminnet_state"]
                target = int(config["trainer"]["epochs"])
                return bool(state.get("completed")) and int(
                    state.get("completed_epochs", -1)
                ) >= target
            if spec.name == "dld":
                dld_settings = config.get("dld", {})
                if isinstance(dld_settings, Mapping) and isinstance(dld_settings.get("diffusion"), Mapping):
                    target = int(dld_settings["diffusion"]["epochs"])
                    state = payload.get("dld_state", {})
                    return int(state.get("completed_epochs", -1)) >= target
                target = int(config["trainer"]["epochs"])
                return int(payload.get("completed_epoch", -1)) + 1 >= target
            if spec.name == "upm":
                upm_settings = config.get("upm", {})
                if isinstance(upm_settings, Mapping) and isinstance(upm_settings.get("main"), Mapping):
                    target = int(upm_settings["main"]["epochs"])
                    state = payload.get("upm_state", {})
                    return int(state.get("main_completed_epochs", -1)) >= target
                target = int(config["trainer"]["epochs"])
                return int(payload.get("epoch", 0)) >= target
            if spec.name == "instance_transition":
                pipeline = payload.get("pipeline", {})
                if not isinstance(pipeline, Mapping):
                    return False
                phase = str(pipeline.get("phase", ""))
                phases = config.get("phases", {})
                if isinstance(phases, Mapping) and phase in {"correction", "revision"}:
                    correction_epochs = int(phases.get("correction_epochs", 0))
                    revision_epochs = int(phases.get("revision_epochs", 0))
                    if revision_epochs > 0:
                        return phase == "revision" and int(
                            payload.get("completed_epoch", -1)
                        ) + 1 >= revision_epochs
                    return phase == "correction" and int(
                        payload.get("completed_epoch", -1)
                    ) + 1 >= correction_epochs
                target = int(config["trainer"]["epochs"])
                return int(payload.get("completed_epoch", -1)) + 1 >= target
            if spec.name == "cwd":
                cwd = config.get("cwd", {})
                if isinstance(cwd, Mapping) and str(
                    cwd.get("protocol", "single_fold")
                ).strip().lower() == "five_fold":
                    state = payload.get("protocol_state", {})
                    return (
                        isinstance(state, Mapping)
                        and state.get("protocol") == "five_fold"
                        and bool(state.get("completed"))
                        and len(state.get("completed_folds", ())) == 5
                    )
                target = int(config["trainer"]["epochs"])
                return int(payload.get("completed_epoch", -1)) + 1 >= target
            if spec.name == "fine":
                target = int(config["trainer"]["epochs"])
                return int(payload.get("completed_epoch", -1)) + 1 >= target
            lend_settings = config.get("lend", {})
            if isinstance(lend_settings, Mapping) and isinstance(lend_settings.get("training"), Mapping):
                target = int(lend_settings["training"]["epochs"])
                return int(payload.get("completed_epoch", -1)) + 1 >= target
            target = int(config["trainer"]["epochs"])
            return int(payload.get("epoch", 0)) >= target
        except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError):
            return False

    def fit(
        self,
        context: RunContext | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> RunResult:
        ctx = self.prepare(context, config=config, output_dir=output_dir, resume=resume)
        effective_resume = (
            Path(resume).expanduser().resolve()
            if resume is not None
            else self._loaded_resume
        )
        ctx.state["lifecycle_active"] = not self._is_completed_noop(
            self.spec, ctx.resolved_config, effective_resume
        )
        if not ctx.state["lifecycle_active"]:
            return RunResult.from_run_dir(ctx.run_dir, resolve=False)
        if effective_resume is not None and self.spec.name in self._RUNNER_OWNED_LIFECYCLE:
            try:
                ctx.session.recover_metrics_from_checkpoint(effective_resume)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                self._prepare_native_resume_log(ctx.run_dir)
            else:
                # Native paper workflows write their own final row, while
                # their legacy checkpoint step is not necessarily a JSONL
                # sequence.  Re-enter after the committed epoch/phase tail.
                self._prepare_native_resume_log(ctx.run_dir)
            ctx.session._sequence = len(load_metric_events(ctx.run_dir / "metrics.jsonl"))
        ctx.state["resume_lifecycle"] = effective_resume is not None
        runner = self.spec.load()
        try:
            if ctx.state["lifecycle_active"] and not ctx.state.get("resume_lifecycle"):
                ctx.session.start_run()
            if ctx.state["lifecycle_active"] and self.spec.name not in self._RUNNER_OWNED_LIFECYCLE:
                ctx.session.start_phase(self.method, total_units=None)
            result = runner(
                dict(ctx.resolved_config),
                ctx.run_dir,
                effective_resume,
                context=ctx,
            )
            if ctx.state["lifecycle_active"] and self.spec.name not in self._RUNNER_OWNED_LIFECYCLE:
                ctx.session.end_phase(self.method)
            run_dir = Path(result).expanduser()
            # Staged algorithms retain their own metric-producing code.  Bring
            # those existing rows through the common event normalizer after the
            # run so sequence numbers and nested metric fields are canonical,
            # without touching any objective or optimizer code.
            if ctx.state["lifecycle_active"]:
                self._normalize_legacy_metrics(run_dir)
            write_run_report(
                run_dir,
                config=ctx.resolved_config,
                runner=self.spec.name,
                method=self.method,
                status="completed",
            )
            return RunResult.from_run_dir(run_dir, resolve=False)
        except BaseException as exc:
            try:
                ctx.session.fail_run(exc)
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            raise


class SupervisedRunner(LegacyRunnerAdapter):
    pass


class StatefulSupervisedRunner(SupervisedRunner):
    pass


class CorrectedRiskRunner(SupervisedRunner):
    pass


class ConditionalTeacherStudentRunner(SupervisedRunner):
    pass


class MultiModelRunner(LegacyRunnerAdapter):
    pass


class StagedRunner(LegacyRunnerAdapter):
    pass


class TwoStageRunner(StagedRunner):
    pass


class FoldRunner(StagedRunner):
    pass


class StatisticRiskRunner(StagedRunner):
    pass


class DualOptimizerRunner(StagedRunner):
    pass


class AlternatingRunner(NativeStagedRunner):
    pass


class ArtifactPipelineRunner(StagedRunner):
    pass


class BinaryRunner(SupervisedRunner):
    pass


class StepRunner(StagedRunner):
    pass


class DiffusionRunner(NativeStagedRunner):
    pass


class GraphStateRunner(NativeStagedRunner):
    pass


def adapter_class_for(spec: Any, method: str | None = None):
    if spec.name == "supervised":
        native_methods = {
            "gce": GceRunner,
            "apl": AplRunner,
            "cdr": CdrRunner,
            "loss_correction": LossCorrectionRunner,
        }
        if method in native_methods:
            return native_methods[method]
    if method == "mentornet":
        return NativeSingleStageRunner
    if method in {"dss"}:
        return NativeSingleStageRunner
    if method in {"cdr"}:
        return StatefulSupervisedRunner
    if method in {"loss_correction"}:
        return CorrectedRiskRunner
    if spec.name == "binary":
        return NativeSingleStageRunner
    if spec.name == "multi_model" and method == "jocor":
        return NativeMultiModelRunner
    if spec.name in {"coteaching", "cnlcu"}:
        return NativeMultiModelRunner
    if spec.name == "dld":
        return DiffusionRunner
    if spec.name == "upm":
        return AlternatingRunner
    if spec.name == "lend":
        return GraphStateRunner
    if spec.name in {"dividemix", "volminnet"}:
        return NativeStagedRunner
    if spec.name in {
        "instance_transition", "dual_t", "t_revision",
        "pcse", "mc_ldce", "volmin",
    }:
        return NativeStagedRunner
    if spec.name == "multi_model" or spec.name in {"coteaching", "cnlcu"}:
        return MultiModelRunner
    if spec.name == "fine":
        return NativeStagedRunner
    if spec.name in {"cal", "cwd", "l2rw"}:
        return NativeStagedRunner
    if spec.name in {"cal", "mc_ldce", "pcse"}:
        return StatisticRiskRunner
    if spec.name == "volmin":
        return DualOptimizerRunner
    if spec.name in {"ca2c", "importance_reweighting"}:
        return NativeStagedRunner
    if spec.name == "importance_reweighting":
        return ArtifactPipelineRunner
    if spec.name == "l2rw":
        return StepRunner
    if spec.name in {"instance_transition", "dual_t", "t_revision"}:
        return StagedRunner
    if spec.name == "supervised" and spec.lifecycle == "single_stage":
        return SupervisedRunner
    return StatefulSupervisedRunner


__all__ = [
    "AplRunner", "ArtifactPipelineRunner", "AlternatingRunner", "BinaryRunner", "CdrRunner",
    "ConditionalTeacherStudentRunner", "GceRunner",
    "CorrectedRiskRunner", "DiffusionRunner", "DualOptimizerRunner", "FoldRunner", "GraphStateRunner",
    "LegacyRunnerAdapter", "LossCorrectionRunner", "MultiModelRunner", "NativeMultiModelRunner",
    "NativeSingleStageRunner", "NativeStagedRunner",
    "StatefulSupervisedRunner", "StatisticRiskRunner",
    "StepRunner", "StagedRunner", "SupervisedRunner", "TwoStageRunner", "adapter_class_for",
]
