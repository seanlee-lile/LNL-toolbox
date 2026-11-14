from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
import uuid

from lnl_toolbox.catalog import discover_recipes, load_papers, recipe_by_id, load_recipe_config
from lnl_toolbox.data.probe import DatasetProbeResult, probe_dataset_path, suggest_dataset_alias
from lnl_toolbox.data.archive import extract_dataset_archive
from lnl_toolbox.data.registry import training_pool_size
from lnl_toolbox.data.profile import KnowledgeState, NoiseOrigin, resolve_dataset_capabilities
from lnl_toolbox.noise.quickstart_catalog import quick_start_noise_specs, visible_synthetic_noise_specs
from lnl_toolbox.training.compatibility import CompatibilityStatus
from lnl_toolbox.training.data_service import DEFAULT_DATA_SERVICE, DataService
from lnl_toolbox.training.runners import resolve_runner
from lnl_toolbox.training.service import ExperimentService

from .models import (
    QuickStartDatasetSummary,
    QuickStartMethodOption,
    QuickStartNoiseSelection,
    QuickStartPlan,
)
from .templates import DatasetConfigurationUnavailable, MethodTemplate, method_template_for_paper, quickstart_runtime_overrides


def _split_size(profile, name: str) -> int | None:
    if profile is None:
        return None
    return dict(profile.sample_counts_by_split).get(name)


def _epoch_details(config: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []

    def visit(value: object, path: tuple[str, ...] = ()) -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            child_path = path + (str(key),)
            if str(key) in {"epochs", "max_epochs", "num_epochs"} and not isinstance(child, Mapping):
                values.append(f"{'.'.join(child_path)}={child}")
            else:
                visit(child, child_path)

    visit(config)
    return tuple(values)


class QuickStartService:
    def __init__(
        self,
        data_service: DataService | None = None,
        *,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.data_service = data_service or DEFAULT_DATA_SERVICE
        self.experiment_service = ExperimentService(data_service=self.data_service)
        self.artifact_root = Path(artifact_root or "artifacts/web-quick-start").expanduser().resolve()
        self._method_cache: dict[str, tuple[QuickStartMethodOption, ...]] = {}
        self._selection_cache: dict[str, tuple[QuickStartMethodOption, ...]] = {}
        self._method_cache_lock = RLock()
        self._template_cache: dict[str, object] = {}
        self._recipe_cache: dict[str, object] | None = None

    def _recipes(self) -> dict[str, object]:
        with self._method_cache_lock:
            if self._recipe_cache is None:
                self._recipe_cache = {
                    item.id: item
                    for item in discover_recipes(include_conditional=True)
                }
            return self._recipe_cache

    def _template(self, paper, recipes: Mapping[str, object], dataset_adapter: str = "") -> MethodTemplate:
        key = f"{paper.id}:{dataset_adapter}"
        with self._method_cache_lock:
            cached = self._template_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        template = method_template_for_paper(
            paper, dataset_adapter=dataset_adapter, recipes=recipes,
        )
        with self._method_cache_lock:
            self._template_cache[key] = template
        return template

    def probe(self, path: str) -> DatasetProbeResult:
        return probe_dataset_path(path, data_service=self.data_service)

    @staticmethod
    def _require_ready_report(report, alias: str):
        """Stop the Quick Start flow before it consumes an incomplete inspect."""

        if report.status != "ready":
            raise ValueError(report.error or f"dataset inspect failed: {alias}")
        if report.profile is None:
            raise ValueError(f"dataset profile is unavailable after inspect: {alias}")
        return report

    def _accepted_report(self, alias: str):
        """Use the profile established at registration without reopening sample files."""
        return self._require_ready_report(self.data_service.status(alias), alias)

    def _accepted_capabilities(self, alias: str, report):
        return resolve_dataset_capabilities(
            report.profile, self.data_service.declarations(alias)
        )

    @staticmethod
    def _selection_key(dataset_alias: str, noise_selection: QuickStartNoiseSelection) -> str:
        return dataset_alias + ":" + json.dumps(noise_selection.to_dict(), sort_keys=True)

    @staticmethod
    def _apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, object]) -> dict[str, Any]:
        candidate = deepcopy(dict(config))
        for path, value in overrides.items():
            if path == "noise":
                candidate["noise"] = deepcopy(value)
                continue
            parts = tuple(str(path).split("."))
            current = candidate
            for part in parts[:-1]:
                child = current.get(part)
                if not isinstance(child, dict):
                    child = {}
                    current[part] = child
                current = child
            current[parts[-1]] = value
        return candidate

    def runtime_overrides(
        self,
        recipe_id: str,
        noise_selection: QuickStartNoiseSelection,
        method_inputs: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return the final, validated-stage choices to apply to a paper YAML."""

        from lnl_toolbox.catalog import recipe_by_id

        recipe = recipe_by_id(recipe_id)
        return quickstart_runtime_overrides(
            load_recipe_config(recipe), noise_selection, method_inputs=method_inputs,
        )

    @staticmethod
    def _summary(alias: str, report) -> QuickStartDatasetSummary:
        profile = report.profile
        noise = profile.noise if profile is not None else None
        return QuickStartDatasetSummary(
            alias=alias,
            adapter=report.adapter,
            path=report.location,
            display_name=(profile.dataset if profile is not None else report.adapter),
            num_classes=report.classes,
            train_size=report.train_samples,
            training_pool_size=training_pool_size(report.adapter, dict(profile.sample_counts_by_split)) if profile is not None else report.train_samples,
            validation_size=_split_size(profile, "validation"),
            test_size=report.test_samples,
            noise_status="unknown" if noise is None else noise.status.value,
            noise_origin="unknown" if noise is None else noise.origin.value,
            clean_train_labels=(
                "unknown" if profile is None else profile.clean_train_labels.value
            ),
        )

    def register_and_inspect(
        self,
        path: str,
        *,
        selected_adapter: str | None = None,
    ) -> QuickStartDatasetSummary | DatasetProbeResult:
        result = self.probe(path)
        if result.status == "already_registered":
            assert result.existing_alias is not None
            report = self.data_service.status(result.existing_alias)
            if report.status != "ready" or report.profile is None:
                report = self.data_service.inspect(result.existing_alias)
            report = self._require_ready_report(report, result.existing_alias)
            return self._summary(result.existing_alias, report)
        candidates = list(result.candidates)
        if selected_adapter is not None:
            candidates = [item for item in candidates if item.adapter == selected_adapter]
        if result.status == "ambiguous" and len(candidates) != 1:
            return result
        if len(candidates) != 1:
            return DatasetProbeResult(result.path, "unsupported", tuple(candidates))
        candidate = candidates[0]
        candidate_data = dict(candidate.data)
        if candidate_data.get("source_archive"):
            unpacked = extract_dataset_archive(Path(candidate_data["source_archive"]), self.artifact_root / "datasets" / "unpacked")
            inner_root = unpacked / candidate_data.pop("archive_subdir")
            recognized = probe_dataset_path(inner_root, data_service=self.data_service)
            if recognized.status == "already_registered":
                return self._summary(recognized.existing_alias, self._accepted_report(recognized.existing_alias))
            matches = [c for c in recognized.candidates if c.adapter == candidate.adapter]
            if len(matches) != 1:
                raise ValueError("unpacked dataset does not match the selected format")
            candidate_data = {**matches[0].data, "source_archive": str(Path(path).resolve())}
        aliases = [record.alias for record in self.data_service.catalog.records()]
        alias = suggest_dataset_alias(candidate.adapter, result.path, aliases)
        self.data_service.register(alias, candidate.adapter, candidate_data)
        with self._method_cache_lock:
            self._method_cache.clear()
            self._selection_cache.clear()
            self._template_cache.clear()
        report = self._require_ready_report(self.data_service.inspect(alias), alias)
        return self._summary(alias, report)

    def noise_options(self, dataset_alias: str) -> dict[str, object]:
        report = self._accepted_report(dataset_alias)
        capabilities = self._accepted_capabilities(dataset_alias, report)
        facts = {
            "noise_status": capabilities.noise_status.value,
            "noise_origin": capabilities.noise_origin.value,
            "clean_train_labels": capabilities.clean_train_labels.value,
            "noise_rate": capabilities.noise_rate.to_dict(),
            "status_source": (
                "inspected" if not self.data_service.declarations(dataset_alias).noise_override
                and report.profile.noise.status == capabilities.noise_status
                and capabilities.noise_status.value != "unknown" else "declared"
            ),
        }
        if capabilities.noise_status.value == "noisy":
            return {
                **facts,
                "dataset_state": "native" if capabilities.noise_origin is NoiseOrigin.NATIVE else "noisy",
                "requires_confirmation": False,
                "options": [{"key": "native", "label": "使用数据集原始 noisy labels"}],
            }
        if capabilities.noise_status.value == "unknown":
            return {
                **facts,
                "dataset_state": "unknown",
                "requires_confirmation": True,
                "options": [{"key": "clean", "label": "保持当前标签"}],
            }
        return {
            **facts,
            "dataset_state": "clean",
            "requires_confirmation": False,
            "options": [
                {"key": item.key, "label": item.label, "description": item.description, "requires_rate": item.requires_rate}
                for item in quick_start_noise_specs()
                if item.key == "clean" or item in visible_synthetic_noise_specs()
                or item.visible and item.category == "external"
            ] + ([{"key": "binary_asymmetric_rcn", "label": "二分类非对称噪声", "description": "分别设置正类与负类标签翻转概率。", "requires_rate": False}]
                 if report.classes == 2 else []),
        }

    @staticmethod
    def _status(result) -> str:
        if result.status is CompatibilityStatus.COMPATIBLE:
            return "ready"
        if result.status is CompatibilityStatus.COMPATIBLE_WITH_REQUIREMENTS:
            return "needs_input"
        return "unsupported"

    def method_options(
        self,
        dataset_alias: str,
        noise_selection: QuickStartNoiseSelection,
    ) -> tuple[QuickStartMethodOption, ...]:
        report = self._accepted_report(dataset_alias)
        capabilities = self._accepted_capabilities(dataset_alias, report)
        cache_key = json.dumps({
            "dataset": dataset_alias,
            "profile_fingerprint": report.profile.fingerprint,
            "capabilities": capabilities.to_dict(),
            "noise": noise_selection.to_dict(),
        }, sort_keys=True)
        with self._method_cache_lock:
            cached = self._method_cache.get(cache_key)
        if cached is not None:
            return cached

        profile_data = report.profile.to_dict()
        papers = load_papers()
        recipes = self._recipes()
        prepared: list[
            tuple[
                object,
                object | None,
                Mapping[str, Any] | None,
                tuple[str, ...] | None,
                tuple[str, ...] | None,
            ]
        ] = []
        compatibility_configs: dict[str, Mapping[str, Any]] = {}
        for paper in papers:
            try:
                template = self._template(paper, recipes, str(profile_data.get("adapter", "")))
                runtime_overrides = quickstart_runtime_overrides(
                    template.config,
                    noise_selection,
                    method_inputs=(
                        {"noise_rate_prior": capabilities.noise_rate.value}
                        if noise_selection.kind == "native" and capabilities.noise_rate.value is not None
                        else None
                    ),
                )
                candidate = self._apply_overrides(template.config, runtime_overrides)
                prepared.append((paper, template, candidate, None, None))
                compatibility_configs[paper.id] = candidate
            except DatasetConfigurationUnavailable as exc:
                prepared.append((paper, None, None, None, (str(exc),)))
            except Exception as exc:
                prepared.append((paper, None, None, (str(exc),), None))

        results: dict[str, object] = {}
        if compatibility_configs:
            try:
                batch = self.experiment_service.list_config_compatibility(
                    dataset_alias, compatibility_configs,
                    dataset_capabilities=capabilities,
                    check_prerequisites=False,
                )
                results = dict(batch)
            except Exception:
                # Keep failures isolated so one malformed paper does not hide all methods.
                for paper_id, candidate in compatibility_configs.items():
                    try:
                        results[paper_id] = self.experiment_service.list_config_compatibility(
                            dataset_alias, {paper_id: candidate},
                            dataset_capabilities=capabilities,
                            check_prerequisites=False,
                        )[0][1]
                    except Exception as exc:
                        results[paper_id] = exc

        options: list[QuickStartMethodOption] = []
        for paper, template, candidate, error, selection_problems in prepared:
            if error is not None:
                options.append(QuickStartMethodOption(
                    paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                    "metadata_error", error, (), None, None, None, (),
                ))
                continue
            if selection_problems is not None:
                options.append(QuickStartMethodOption(
                    paper.id, paper.acronym, paper.title, paper.summary,
                    paper.venue, paper.year, "unsupported",
                    selection_problems, (), None,
                    None, candidate, (),
                ))
                continue
            assert template is not None and candidate is not None
            result = results.get(paper.id)
            if isinstance(result, Exception) or result is None:
                reason = str(result) if isinstance(result, Exception) else "兼容性检查未返回结果"
                options.append(QuickStartMethodOption(
                    paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                    "metadata_error", (reason,), (), template.recipe_id,
                    "paper_reproduction", candidate, (),
                ))
                continue
            status = self._status(result)
            reasons = tuple(item.message for item in result.reasons)
            options.append(QuickStartMethodOption(
                paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                status,
                reasons,
                tuple(result.required_user_inputs),
                template.recipe_id,
                "paper_reproduction",
                candidate,
                tuple(result.required_input_paths),
            ))
        final = tuple(options)
        with self._method_cache_lock:
            self._method_cache[cache_key] = final
            self._selection_cache[self._selection_key(dataset_alias, noise_selection)] = final
        return final

    @staticmethod
    def _plan_id(dataset_alias: str, paper_id: str) -> str:
        return f"{dataset_alias}-{paper_id}-{uuid.uuid4().hex[:10]}"

    def _external_label_facts(self, noise_selection, report) -> dict[str, object] | None:
        """Validate a chosen external label file before any plan is produced."""

        if str(getattr(noise_selection, "key", "")) != "external_torch":
            return None
        from lnl_toolbox.noise.quickstart_catalog import inspect_external_label_file

        problems: list[str] = []
        try:
            facts = inspect_external_label_file(noise_selection.path or "")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return {"compatible": False, "problems": (str(exc),)}
        counts = dict(getattr(report.profile, "sample_counts_by_split", {}) or {})
        expected = training_pool_size(str(report.adapter), counts)
        if expected is not None and int(facts["rows"]) != int(expected):
            problems.append(
                f"外部标签文件有 {facts['rows']} 行，而当前训练池是 {expected} 个样本；"
                "请确认文件与所选数据集/子集一致"
            )
        facts["expected_rows"] = expected
        facts["compatible"] = not problems
        facts["problems"] = tuple(problems)
        facts["summary"] = (
            f"外部标签：{facts['rows']} 行，实测噪声率 {float(facts['measured_noise_rate']):.3f}，"
            f"路径 {facts['path']}"
        )
        return facts

    def build_plan(
        self,
        *,
        dataset_alias: str,
        noise_selection: QuickStartNoiseSelection,
        paper_id: str,
        user_inputs: Mapping[str, object] | None = None,
    ) -> QuickStartPlan:
        from lnl_toolbox.catalog import recipe_by_id

        with self._method_cache_lock:
            options = self._selection_cache.get(self._selection_key(dataset_alias, noise_selection))
        if options is None:
            raise ValueError("请先完成第 2 步到第 3 步的方法适用性检查。")
        selected = next((item for item in options if item.paper_id == paper_id), None)
        if selected is None or selected.recipe_id is None:
            raise ValueError("所选论文没有对应的已登记 YAML 配置。")
        recipe = recipe_by_id(selected.recipe_id)
        config = load_recipe_config(recipe)
        label_file = self._external_label_facts(noise_selection, self._accepted_report(dataset_alias))
        if label_file is not None and not label_file["compatible"]:
            return QuickStartPlan(
                self._plan_id(dataset_alias, paper_id), dataset_alias, paper_id,
                selected.acronym, noise_selection, "paper_reproduction",
                selected.recipe_id, None, "unsupported", (), None, None,
                summary="所选外部标签文件与当前数据集不匹配。",
                details=tuple(label_file["problems"]),
            )
        inputs = dict(user_inputs or {})
        remaining = list(selected.required_user_inputs)
        for required in tuple(remaining):
            if required == "noise_rate_prior" and inputs.get(required) not in (None, ""):
                remaining.remove(required)
                continue
            if required.startswith("config:"):
                mapped = next((paths for name, paths in selected.required_input_paths if name == required), ())
                if mapped and all(".".join(path) in inputs and inputs[".".join(path)] not in (None, "") for path in mapped):
                    remaining.remove(required)
        status = selected.status if selected.status != "needs_input" or remaining else "ready"
        plan_id = self._plan_id(dataset_alias, paper_id)
        output_dir = self.artifact_root / "runs" / plan_id
        # Store the selected noise and method-specific inputs in the review YAML,
        # not in the paper's built-in source file.
        command = f"lnl run --recipe {selected.recipe_id} --data {dataset_alias} --output-dir {output_dir} --check-data"
        dry_run = f"lnl run --recipe {selected.recipe_id} --data {dataset_alias} --output-dir {output_dir} --dry-run --check-data"
        return QuickStartPlan(
            plan_id, dataset_alias, paper_id, selected.acronym, noise_selection,
            "paper_reproduction", selected.recipe_id, None, status,
            tuple(remaining), command, dry_run,
            summary="已读取所选论文的 YAML；完整数据与依赖检查在最后一步执行。",
            details=tuple(f"训练轮次：{item}" for item in _epoch_details(config))
            + (f"配置文件：{recipe.config_path}", f"输出目录：{output_dir}")
            + (() if label_file is None else (str(label_file["summary"]),)),
        )


__all__ = ["QuickStartService"]
