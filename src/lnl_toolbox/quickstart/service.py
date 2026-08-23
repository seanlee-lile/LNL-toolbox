from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
import uuid

from lnl_toolbox.catalog import discover_recipes, load_papers, recipe_by_id, load_recipe_config
from lnl_toolbox.data.probe import DatasetProbeResult, probe_dataset_path, suggest_dataset_alias
from lnl_toolbox.data.profile import KnowledgeState, NoiseOrigin
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
from .templates import MethodTemplate, adapt_method_template, find_exact_reproduction, method_template_for_paper


class _CachedDatasetService:
    """Provide one local-dataset overlay without rereading the catalog per paper."""

    _SOURCE_KEYS = {"root", "path", "noise_path", "labels_path", "annotation_root"}

    def __init__(self, service: DataService, alias: str) -> None:
        self._service = service
        self._alias = alias
        applied = service.apply({}, alias)
        self._data = dict(applied.get("data", {}) or {})
        self._local_dataset = dict(applied.get("local_dataset", {}) or {})

    def apply(self, config: Mapping[str, Any], alias: object) -> dict[str, Any]:
        if str(alias) != self._alias:
            return self._service.apply(config, alias)
        result = deepcopy(dict(config))
        data = dict(result.get("data", {}) or {})
        for key in self._SOURCE_KEYS:
            data.pop(key, None)
        data.update(self._data)
        result["data"] = data
        result["local_dataset"] = dict(self._local_dataset)
        return result


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

    def _template(self, paper, recipes: Mapping[str, object]) -> MethodTemplate:
        with self._method_cache_lock:
            cached = self._template_cache.get(paper.id)
        if cached is not None:
            return cached  # type: ignore[return-value]
        selected = next(
            (item for item in paper.configs if item.profile == "reproduction"),
            None,
        )
        if selected is None:
            raise ValueError(f"paper {paper.id!r} has no formal reproduction recipe")
        recipe = recipes.get(selected.recipe_id)
        if recipe is None:
            raise ValueError(f"unknown recipe {selected.recipe_id!r}")
        template = MethodTemplate(
            paper.id, selected.recipe_id, recipe, load_recipe_config(recipe)
        )
        with self._method_cache_lock:
            self._template_cache[paper.id] = template
        return template

    @staticmethod
    def _exact_reproduction(paper, recipes, dataset_adapter: str, noise_selection) -> str | None:
        wanted_noise = "clean" if noise_selection.kind in {"clean", "native"} else noise_selection.key
        normalized_adapter = str(dataset_adapter).lower().replace("-", "_")
        for item in paper.configs:
            if item.profile != "reproduction":
                continue
            recipe = recipes.get(item.recipe_id)
            if recipe is None:
                continue
            config = load_recipe_config(recipe)
            data = config.get("data", {}) or {}
            noise = config.get("noise", {}) or {}
            data_name = str(data.get("name", "")).lower().replace("-", "_")
            if data_name != normalized_adapter:
                continue
            if str(noise.get("name", "clean")) != wanted_noise:
                continue
            configured_rate = noise.get("rate")
            if noise_selection.rate is not None and configured_rate is not None:
                if float(configured_rate) != float(noise_selection.rate):
                    continue
            return recipe.id
        return None

    def probe(self, path: str) -> DatasetProbeResult:
        return probe_dataset_path(path, data_service=self.data_service)

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
            report = self.data_service.inspect(result.existing_alias)
            return self._summary(result.existing_alias, report)
        candidates = list(result.candidates)
        if selected_adapter is not None:
            candidates = [item for item in candidates if item.adapter == selected_adapter]
        if result.status == "ambiguous" and len(candidates) != 1:
            return result
        if len(candidates) != 1:
            return DatasetProbeResult(result.path, "unsupported", tuple(candidates))
        candidate = candidates[0]
        aliases = [record.alias for record in self.data_service.catalog.records()]
        alias = suggest_dataset_alias(candidate.adapter, result.path, aliases)
        self.data_service.register(alias, candidate.adapter, candidate.data)
        with self._method_cache_lock:
            self._method_cache.clear()
            self._template_cache.clear()
        report = self.data_service.inspect(alias)
        return self._summary(alias, report)

    def noise_options(self, dataset_alias: str) -> dict[str, object]:
        capabilities = self.data_service.capabilities(dataset_alias, persist=False)
        if capabilities.noise_origin is NoiseOrigin.NATIVE:
            return {
                "dataset_state": "native",
                "requires_confirmation": False,
                "options": [{"key": "native", "label": "使用数据集原始 noisy labels"}],
            }
        if capabilities.noise_status.value == "unknown":
            return {
                "dataset_state": "unknown",
                "requires_confirmation": True,
                "options": [{"key": "clean", "label": "保持当前标签"}],
            }
        return {
            "dataset_state": "clean",
            "requires_confirmation": False,
            "options": [
                {"key": item.key, "label": item.label, "description": item.description, "requires_rate": item.requires_rate}
                for item in quick_start_noise_specs()
                if item.key == "clean" or item in visible_synthetic_noise_specs()
            ],
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
        cache_key = json.dumps({
            "dataset": dataset_alias,
            "noise": noise_selection.to_dict(),
        }, sort_keys=True)
        with self._method_cache_lock:
            cached = self._method_cache.get(cache_key)
        if cached is not None:
            return cached

        profile = self.data_service.inspect(dataset_alias).profile
        profile_data = {} if profile is None else profile.to_dict()
        dataset_service = _CachedDatasetService(self.data_service, dataset_alias)
        options: list[QuickStartMethodOption] = []
        metadata_errors: dict[str, QuickStartMethodOption] = {}
        candidates: dict[str, Mapping[str, Any]] = {}
        templates: dict[str, object] = {}
        papers = load_papers()
        recipes = self._recipes()
        for paper in papers:
            try:
                template = self._template(paper, recipes)
                templates[paper.id] = template
                candidates[paper.id] = adapt_method_template(
                    template.config,
                    dataset_alias=dataset_alias,
                    dataset_profile=profile_data,
                    noise_selection=noise_selection,
                    data_service=dataset_service,
                )
            except Exception as exc:
                metadata_errors[paper.id] = QuickStartMethodOption(
                    paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                    "metadata_error", (str(exc),), (), None, None, None, (),
                )
        try:
            compatibility = dict(self.experiment_service.list_config_compatibility(
                dataset_alias, candidates
            ))
        except Exception:
            # Preserve the old per-paper failure isolation if one candidate is malformed.
            compatibility = {}
            for paper_id, candidate in candidates.items():
                try:
                    compatibility[paper_id] = self.experiment_service.list_config_compatibility(
                        dataset_alias, {paper_id: candidate}
                    )[0][1]
                except Exception as exc:
                    compatibility[paper_id] = exc

        for paper in papers:
            if paper.id in metadata_errors:
                options.append(metadata_errors[paper.id])
                continue
            candidate = candidates[paper.id]
            result = compatibility.get(paper.id)
            if isinstance(result, Exception) or result is None:
                reason = str(result) if isinstance(result, Exception) else "兼容性检查未返回结果"
                options.append(QuickStartMethodOption(
                    paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                    "metadata_error", (reason,), (), None, None, None, (),
                ))
                continue
            template = templates[paper.id]
            options.append(QuickStartMethodOption(
                paper.id, paper.acronym, paper.title, paper.summary, paper.venue, paper.year,
                self._status(result),
                tuple(item.message for item in result.reasons),
                tuple(result.required_user_inputs),
                template.recipe_id,
                "paper_reproduction" if self._exact_reproduction(
                    paper, recipes, dataset_adapter=str(profile_data.get("adapter", "")), noise_selection=noise_selection
                ) else "toolbox_adapted",
                candidate,
                tuple(result.required_input_paths),
            ))
        final = tuple(options)
        with self._method_cache_lock:
            self._method_cache[cache_key] = final
        return final

    @staticmethod
    def _plan_id(dataset_alias: str, paper_id: str) -> str:
        return f"{dataset_alias}-{paper_id}-{uuid.uuid4().hex[:10]}"

    def build_plan(
        self,
        *,
        dataset_alias: str,
        noise_selection: QuickStartNoiseSelection,
        paper_id: str,
        user_inputs: Mapping[str, object] | None = None,
    ) -> QuickStartPlan:
        from lnl_toolbox.catalog import paper_by_id

        paper = paper_by_id(paper_id)
        profile = self.data_service.inspect(dataset_alias).profile
        if profile is None:
            raise ValueError(f"dataset profile is unavailable: {dataset_alias}")
        exact = find_exact_reproduction(
            paper, dataset_adapter=profile.adapter, noise_selection=noise_selection
        )
        if exact is not None:
            plan_id = self._plan_id(dataset_alias, paper.id)
            config = load_recipe_config(recipe_by_id(exact))
            output_dir = self.artifact_root / "runs" / plan_id
            details = tuple(f"训练轮次：{item}" for item in _epoch_details(config)) + (
                f"输出目录：{output_dir}",
            )
            return QuickStartPlan(
                plan_id, dataset_alias, paper.id, paper.acronym,
                noise_selection, "paper_reproduction", exact, None, "ready", (),
                f"lnl run --recipe {exact} --data {dataset_alias} --output-dir {output_dir} --check-data",
                f"lnl run --recipe {exact} --data {dataset_alias} --output-dir {output_dir} --dry-run --check-data",
                "条件完全匹配正式论文复现配置。",
                details=details,
            )
        template = method_template_for_paper(paper)
        candidate = adapt_method_template(
            template.config,
            dataset_alias=dataset_alias,
            dataset_profile=profile.to_dict(),
            noise_selection=noise_selection,
            data_service=self.data_service,
            method_inputs=user_inputs,
        )
        result = self.experiment_service.list_config_compatibility(
            dataset_alias, {paper.id: candidate}
        )[0][1]
        status = self._status(result)
        details = tuple(item.message for item in result.reasons)
        plan_id = self._plan_id(dataset_alias, paper.id)
        if status != "ready":
            return QuickStartPlan(
                plan_id, dataset_alias, paper.id, paper.acronym, noise_selection,
                "toolbox_adapted", None, None, status, tuple(result.required_user_inputs),
                summary="当前方法需要补充输入或不适用于该数据集。", details=details,
            )

        try:
            self.experiment_service.preflight(candidate, check_data=True)
        except Exception as exc:
            return QuickStartPlan(
                plan_id, dataset_alias, paper.id, paper.acronym, noise_selection,
                "toolbox_adapted", None, None, "unsupported", (),
                summary="适配配置未通过训练前检查。", details=(str(exc),),
            )
        self.artifact_root.joinpath("configs").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        stem = f"{stamp}-{dataset_alias}-{paper.id}-{noise_selection.key}"
        config_path = self.artifact_root / "configs" / f"{stem}.yaml"
        provenance_path = config_path.with_suffix(".json")
        output_dir = self.artifact_root / "runs" / plan_id
        import yaml

        config_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        provenance_path.write_text(json.dumps({
            "kind": "toolbox_adapted",
            "dataset_alias": dataset_alias,
            "paper_id": paper.id,
            "base_recipe": template.recipe_id,
            "noise": noise_selection.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return QuickStartPlan(
            plan_id, dataset_alias, paper.id, paper.acronym, noise_selection,
            "toolbox_adapted", None, str(config_path), "ready", (),
            f"lnl run --config {config_path} --output-dir {output_dir} --check-data",
            f"lnl run --config {config_path} --output-dir {output_dir} --dry-run --check-data",
            "基于现有方法模板生成的 Toolbox 适配配置，不等同于论文原始复现。",
            str(provenance_path), details + tuple(
                f"训练轮次：{item}" for item in _epoch_details(candidate)
            ) + (f"输出目录：{output_dir}",),
        )


__all__ = ["QuickStartService"]
