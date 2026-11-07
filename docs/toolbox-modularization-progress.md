# Toolbox modularization progress

## code 分支三方法预合并（2026-10-31）

- 任务：把远端 `code@413d858f3d72c3ff74fec511e3f881e922968b2a` 的 CWD、PCSE、LEND 改动局部合入当前工作区；用户随后明确批准提交并推送到远端 scratch，不切换分支。
- 当前分支 / 基线：`scratch@f48715faae5e4712495c1e1df2fd68287ce910e6`；共同基线 `c59a9b5d78e18b84232775a6cf3aa611016e2931`。
- 来源提交：`637e7462`（CWD Adam beta1 调度）、`0f2f33ff`（PCSE 按 clean priors 加权共享协方差）、`413d858f`（LEND author_release 可选实现）。
- 进度：来源核实、范围核对、局部应用、冲突处理、针对性验证、最终范围记录，共 6 / 6。
- 用户批准的修改清单（13 个已有文件，无新增源码或依赖）：
  - `src/lnl_toolbox/algorithms/lend/algorithm.py`、`config.py`、`dilution.py`、`graph.py`、`history.py`；
  - `src/lnl_toolbox/algorithms/pcse/gda.py`；
  - `src/lnl_toolbox/training/cwd_experiment.py`、`lend_experiment.py`、`planning.py`；
  - `tests/test_cwd.py`、`tests/test_pcse.py`、`tests/test_lend.py`；
  - 本文档。
- 逐文件变化：
  - CWD runner：可选 `optimizer.beta1_schedule`，zero-based epoch 80 将 Adam beta1 从 0.9 切换至 0.1；默认关闭，记录有效 beta1。
  - PCSE GDA：以恢复的干净类别先验加权各类协方差，替换等权平均；保留原对称化与 ridge 处理。
  - LEND algorithm：按显式 implementation_variant 分流；author_release 使用连续 noisy-class probability 权重；保留 paper_equation 路径和旧 checkpoint 身份兼容。
  - LEND config：校验 paper_equation / author_release，记录各实现的有效控制项及 inactive inherited controls。
  - LEND graph：新增作者实现的平方欧氏 top-k、AAT 图及其零度数行为。
  - LEND dilution：新增 lambda=0.99 resolvent solve、整批 NaN 回退与 softmax。
  - LEND history：新增传播前历史混合与直接存储，zero-based epoch 10 开始使用上一轮历史。
  - LEND runner：保存实现身份和权重统计，不改变数据、模型、优化器、调度器或结果选择协议。
  - planning.py：仅在 author_release 分支展示实际实现语义；原计划路径保留。
  - 三个测试文件：合入远端新增覆盖；CWD 与 LEND 保留当前 scratch 已有测试调整，LEND 不恢复本地已删除的旧测试，并补查 loss sum 与 objective 的区别。
- 一致性：通过 GitHub tree blob SHA 与本地 `git hash-object` 比较，9 个生产源码文件全部与 code tip 一致；测试文件的本地差异为上述有意保留项。直接 git fetch 遇到网络连接失败，改用 GitHub 接口获取同一固定 SHA 的提交补丁及 tree，不修改远端地址。
- 实际验证（既有 `F:/Miniconda/envs/pytorch/python.exe`，`PYTHONPATH=src;tests;.`，`PYTHONDONTWRITEBYTECODE=1`，torch threads=2）：
  - `unittest.defaultTestLoader.loadTestsFromNames(['test_cwd','test_pcse','test_lend'])`：130 项通过，89.183 秒。
  - `loadTestsFromNames(['test_registry','web.test_quick_start_api','test_lend._lend_algorithm_LENDAlgorithmTest.test_exact_batch_mean_formula_uses_four_sample_denominator','test_lend._lend_algorithm_LENDAlgorithmTest.test_one_forward_batch_mean_and_noisy_targets'])`：61 项通过，19.554 秒；包含最终补充的两条 loss-sum 断言。
  - `python -B -m ruff check --no-cache src/lnl_toolbox/algorithms/lend src/lnl_toolbox/algorithms/pcse/gda.py src/lnl_toolbox/training/cwd_experiment.py src/lnl_toolbox/training/lend_experiment.py src/lnl_toolbox/training/planning.py tests/test_cwd.py tests/test_pcse.py tests/test_lend.py`：通过。
  - `git diff --check`：通过，仅有现有 Windows 行尾提示。
  - 发布前补查：以 `unittest.defaultTestLoader.discover(DIRECTORY, pattern='test_*.py')` 和 `TextTestRunner(verbosity=1, buffer=True)` 运行完整 `tests`：1191 项通过（390.840 秒）；完整 `web`：142 项通过（305.124 秒）；完整 `src/lnl_toolbox/scratch/tests`：224 项通过（39.429 秒）。环境变量及 torch 线程数同上。三组均无失败、错误或跳过。
- 范围保护：未修改 Web、正式 YAML、数据或 checkpoint；原有 `data/cifar_n.py`、`training/dual_t_evidence_experiment.py`、SSH 密钥及未跟踪训练产物保持不动。实际源码修改与批准清单一致。
- 限制：没有执行浏览器点击短训练或全量训练；不是正式论文准确率验收。CWD 新调度仍需显式开启，LEND 原配置默认仍为 paper_equation；没有擅改默认实现。
- 发布安排：用户已批准提交并推送 `scratch`；只暂存本次 13 个文件，单个提交 `Integrate CWD PCSE and LEND updates from code`，不创建分支、不重写历史、不强制推送。实际提交号及远端结果以 Git 记录为准。共享文件协作关注点是 planning.py 的 LEND 计划分支及本文档的新记录。
- 下一步：提交后推送至 origin 的 scratch，并核对远端提交号；若需默认启用 author_release，另行明确批准配置修改范围。

## 提取旧 life 的独立改动（2026-10-29）

- 当前任务：从 `life@ca855a7` 提取开发工具、错误诊断和论文资料页容错，不覆盖当前训练实现。
- 分支 / 基线：提取分支 `codex/extract-life-1019`，基线 `scratch@84959b1`；合并目标为本地 scratch。
  没有引入最新 life；未推送。
- Checklist：历史与来源核对、局部提取、针对性检查、26 篇前后对照、完整检查、本地合并与范围复核。
- 进度：6 / 6（100%）；本节随本地合并提交生效。提取提交：`9917951`。
- 修改范围：`src/lnl_toolbox/cli/main.py`、`src/lnl_toolbox/training/service.py`、
  `web/command_console.py`、`web/index.html`、`web/assets/run_output.js`、`tests/test_cli.py`、
  `tests/test_experiment_service_data_contract.py`、`web/test_command_console.py`、
  `web/test_training_status.py`、`docs/development-guide.md` 和本文档。
- 新增：`scripts/new_paper.py`、`scripts/check_paper_registration.py`、
  `tests/test_new_paper.py`、`tests/test_paper_registration.py`。工具与其检查取自旧 life；
  去掉固定论文 / 配置数量断言，不加入 CI，不新增依赖。
- 保留 scratch：正式配方选择、原始参数校验顺序、数据适配、配置保存、启动分发、恢复、
  算法 / 模型 / 预处理、正式 YAML、Quick Start 核心、现有 CI；不引入 life 的强制重复准入或 smoke 默认回退。
- 验证：针对性 53 项 + 论文页 Node 行为 1 项通过；登记工具检查 PASS（0 issues）；Ruff 通过。
  26 篇现有页面 payload 与基线完全一致，26 篇完整训练调用参数一致；调用被拦截，没有执行 26 篇真实训练。
  AST 对照确认 CLI 参数解析（除 help）、保存、启动、分发、恢复和成功校验路径不变；受保护路径 git diff 为空。
  完整单元 1155 项、Web 142 项、Scratch 224 项全部通过，无失败或跳过。
- 实际运行的检查命令：在既有 `F:/Miniconda/envs/pytorch/python.exe` 环境设置
  `PYTHONPATH=src;tests;.`、`PYTHONDONTWRITEBYTECODE=1`；分别以
  `python -B -c "import torch,unittest; torch.set_num_threads(2); r=unittest.TextTestRunner(verbosity=1,buffer=True).run(unittest.defaultTestLoader.discover(DIRECTORY)); raise SystemExit(not r.wasSuccessful())"`
  运行 `tests`、`web`（pattern=`test_*.py`）和 `src/lnl_toolbox/scratch/tests`。
  `python -B scripts/check_paper_registration.py`、
  `python -B -m ruff check --no-cache src tests web scripts/new_paper.py scripts/check_paper_registration.py`、
  `git diff --check` 均通过。
- 真实 Web 短训练：临时 HTTP 服务，通过 methods → plan → `/api/run` → jobs
  选择 GCE / `gce-cifar10-noise02-reproduction` / 已登记 `cifar10-local`，仅用 `--epochs 1` 缩短预算。
  实际完成 1 轮、19 次参数更新，退出码 0，Web 状态 completed；临时输出自动清理。
  这是 Web API 到真实训练的验证，不是浏览器逐次点击，也不是 26 篇论文全部真实训练。
- 冲突与边界：CLI / service / Web / 共享文档为协作敏感文件，均局部合入；
  原有 Dual-T 文件状态、SSH 密钥和训练产物保持不动，不加入提交。
- 范围复核：11 个已有文件、4 个新增文件与批准白名单完全一致；无计划外源码文件。
  不需要整理临时提交历史；完整检查和真实短训练通过后，局部提取合并回本地 scratch。
- 精确下一步：查看本地合并结果与差异；若需要发布，另行批准推送 scratch。

## CI input-validation and resume repair (2026-10-15)

- Task: resolve the failures in quality run `38672394362`, without freezing configurable experiment values to historical defaults. Branch/base: `scratch` / `8f673918f2246864022a6db093ae2ac27216665e`. No branch change, commit or push is authorized.
- Approved modification allowlist: `src/lnl_toolbox/catalog.py`, `src/lnl_toolbox/training/cwd_experiment.py`, `configs/smoke/mc_ldce_cifar10_smoke.yaml`, `tests/test_cli.py`, `tests/test_registry.py`, `tests/test_cwd.py`, `tests/test_training.py`, and this progress document. No new files or dependencies.
- Checklist: (1) repair model-family/nested input validation; (2) replace fixed-default assertions with supported-custom-configuration checks and correct CE fixture facts; (3) migrate the existing MC-LDCE short fixture to the implemented lifecycle; (4) preserve CWD loader RNG state across epoch-boundary resume; (5) run and close focused, full-unit, Web, Scratch and lint regressions. Completed: 5 / 5 (100%); the approved second-stage repairs and final local regressions are complete. Earlier failed runs below are diagnostic history, not the final status.
- Input validation: model variants are checked against the selected model family at any stage nesting depth; model initialization options, risk-correction names and LEND dilution policy/tolerance/iteration limits are checked before normalization. Supported choices are implementation domains, not prescribed experiment defaults.
- Configurable controls: LEND, UPM and CNLCU checks accept modified model/budget/optimizer/scheduler settings and verify that validation does not rewrite source configurations. CE compatibility uses explicitly declared clean validation data; its runtime cleanliness requirement was not disabled.
- CWD diagnosis: `PreparedData.loader()` uses an independent Torch generator. Global checkpoint RNG restoration alone resets each loader's sample-order state after restart. Checkpoints now retain and restore the train, snapshot and test loader generators. A strengthened multi-batch test failed before this repair, then passed with zero tolerance on model tensors, exact metric equality and exact loader-state equality. Older incomplete checkpoints without loader RNG states are rejected when more epochs remain, rather than silently promising identical continuation.
- MC-LDCE: only the old synthetic smoke YAML is moved from lifecycle 2/fixed features to the existing lifecycle 3/joint-feature training. Formal configurations and algorithm equations are unchanged. The fixture completes two real optimization epochs and checkpoint loading.
- Focused command (passed, 31 tests): `python -m unittest test_cli.ConfigParameterGateTest test_cli._lend_cli_LENDCliTest test_cli._upm_cli_UPMCliTest test_cli._unified_cli_UnifiedCliTest.test_plain_ce_validate_check_data_reports_compatible test_registry._cnlcu_readiness_CNLCUReadinessTest test_registry._compatibility_RunnerRequirementsTest.test_cal_mc_ldce_and_ca2c_preserve_their_synthetic_smokes test_cwd._cwd_training_CWDTrainingTest.test_interrupted_fold_resumes_same_plan_without_validation_metrics test_training._new_paper_training_NewPaperTrainingTest.test_all_new_workflows_smoke_and_resume -v` (`PYTHONPATH=src;.;tests`).
- Additional focused command (passed after the legacy-state guard): `python -m unittest test_cwd._cwd_training_CWDTrainingTest.test_interrupted_fold_resumes_same_plan_without_validation_metrics -v`. Scoped Ruff and `git diff --check` passed.
- Files inspected: the allowlist plus model factories, LEND/UPM/CNLCU parsers, shared data-loader/checkpoint/snapshot contracts, CLI planning and the CI workflow. Collaborator warning: the worktree already contained 28 modified files, including `catalog.py`, `test_registry.py` and `test_training.py`; these existing changes are preserved. Other pre-existing Web/model/algorithm/recipe work is outside this task.
- Limitations: local verification is Windows with its existing Python/Torch installation, not the hosted Linux/Python 3.12 runner. CI has not been rerun; no Web training acceptance is claimed from unit tests.
- Regression commands (`PYTHONPATH=src;.;tests`, existing local Python runtime):
  - `python -m unittest discover -s tests -q`: 1179 tests in 478.884s; one error remains. The all-built-in-config check previously stopped on Loss Correction; it now reaches `pcse-multiclass-volmin-smoke`, whose old `transition_stage.parameterization` contains unsupported `initial_flip_mass`, `max_flip_mass` and `temperature`. A separate read-only validation of all 63 non-conditional built-in recipes identified exactly this one invalid recipe. Its YAML is outside the approved allowlist.
  - `python -m unittest discover -s web -p 'test_*.py' -q`: 163 tests in 422.226s; six failure records: parameter-order subtests for Co-teaching/CNLCU/PDL, Loss Correction presentation expectation, noise-prior derivation expectation, and CDR permission expectation. No Web file was modified by this task.
  - `python -m unittest discover -s src/lnl_toolbox/scratch/tests -q`: 249 tests in 15.470s; three failures and one error: synthetic fixture incompatible with classwise validation splitting, cross-config Co-teaching model assertion, pure-helper import isolation assertions for LEND/CDR, and Scratch CNN8 missing the classifier BatchNorm1d layer present in the shared CNN8. No Scratch file was modified by this task.
  - `python -m ruff check src tests web`: passed. `git diff --check`: passed (line-ending warnings only).
- Final scope review for this stage: all eight approved files were changed, no files were created, and the 28 pre-existing dirty paths were preserved. High-conflict shared-file changes are restricted to the input-gate section of `catalog.py`, the CNLCU test in `test_registry.py`, the new-workflow test in `test_training.py`, and this new progress section.
- The first-stage failures above are retained as diagnostic history. The user subsequently approved a second modification allowlist: `configs/smoke/pcse_multiclass_volmin_smoke.yaml`, `web/command_console.py`, `web/lnl_parameter_metadata_registry_revised.yaml`, `web/test_command_console.py`, `src/lnl_toolbox/scratch/data_runtime.py`, `src/lnl_toolbox/scratch/blocks/models.py`, `src/lnl_toolbox/scratch/tests/test_formula_templates.py`, and `src/lnl_toolbox/scratch/tests/test_runtime_isolation.py`. No new files or dependencies were approved or added.
- Second-stage file changes:
  - `configs/smoke/pcse_multiclass_volmin_smoke.yaml`: migrate obsolete transition parameterization to the existing sigmoid-off-diagonal implementation, preserving the initial transition matrix (diagonal 0.95, off-diagonal 0.025). Formal PCSE defaults are unchanged.
  - `tests/test_training.py`: add direct validation, real pipeline training and completed-checkpoint resume of the unmodified registered PCSE short fixture; the test does not replace its parameterization.
  - `web/lnl_parameter_metadata_registry_revised.yaml`: remove the stale hidden-path entry for the editable Loss Correction risk corrector, matching the existing method-aware presentation policy.
  - `web/test_command_console.py`: check ordering of explicitly ordered controls without forbidding additional supported controls; resolve method-aware presentation precedence; test known-matrix warmup visibility, mode-dependent CDR permissions, and independence of dataset noise and the editable method noise prior. Wrong derived values remain rejected.
  - `src/lnl_toolbox/scratch/blocks/models.py`: align Scratch CNN8/CNN9 topology, BatchNorm, dropout and initialization with the existing shared implementations; preserve the pre-existing model-name migration.
  - `src/lnl_toolbox/scratch/data_runtime.py`: construct bounded balanced synthetic fixtures and complete class groups for classwise splitting, including splits configured after source loading. Real-data splitting remains fail-closed and unchanged.
  - `src/lnl_toolbox/scratch/tests/test_formula_templates.py`: test cross-peer recipe wiring with customizable models/budgets, exact shared/Scratch model tensors and outputs, and early/late classwise fixture splitting with original plans unchanged.
  - `src/lnl_toolbox/scratch/tests/test_runtime_isolation.py`: explicitly allow only the existing LEND/CDR pure-math helper imports; retain rejection of other algorithm/trainer imports.
  - `web/command_console.py`: approved for inspection/possible repair but no edits made by this task; existing presentation/permission behavior was correct for the failed assertions.
- Second-stage regression: `python -m unittest discover -s tests -q` passed all 1180 tests in 852.422s; `python -m unittest discover -s src/lnl_toolbox/scratch/tests -q` passed all 252 tests in 58.189s; the final fresh `python -m unittest discover -s web -p 'test_*.py' -q` passed all 164 tests in 797.529s. `python -m ruff check src tests web` and `git diff --check` passed (line-ending warnings only). The 24-test focused batch and a previously started full Web process initially had only the stale Loss Correction warmup-group assertion remaining; after changing it to check actual known-matrix visibility, its focused recheck passed (one comprehensive all-paper test, 55.436s), followed by the fully green fresh Web run above.
- Execution environment: Windows, existing `F:/Miniconda/envs/pytorch/python.exe` (Python 3.11.15); `PYTHONIOENCODING=utf-8`, `PYTHONPATH=src;.;tests` for unit tests, `src;.` for discovered Web/Scratch tests and Ruff, and `src;.;web` for the focused Web recheck. Full-suite stdout was redirected to `$null`; unittest failure/summary stderr remained visible. No new runtime or dependencies were installed.
- Scope: this task has modified 15 of the 16 approved paths, with no added files. The other 28 initially dirty paths/overlapping prior edits remain preserved. Shared catalog, Web metadata/tests, Scratch model factory and progress-document changes need joint diff review by collaborators.
- Final scope/diff: 15 approved paths modified by this task; the sixteenth approved path (`web/command_console.py`) retains only pre-existing changes. No new or unplanned files were produced, no staging/checkpoint commits/Git operations altering state were performed, and the original 28 dirty paths were preserved. Final `git diff --check` passed. Formal reproduction YAMLs, shared training algorithms/data adapters, CI workflow and dependencies were intentionally left unchanged by this task.
- Known limitations: hosted Linux/Python 3.12 CI was not rerun, and local regression success does not claim 26-paper browser training acceptance. Legacy incomplete CWD checkpoints without loader RNG state cannot provide exact continuation and now report that limitation explicitly.
- Exact next step / integration: review the approved 15-file incremental changes alongside the pre-existing model-name migration; use the commands above for local reproduction; request separate approval before staging, committing or publishing. History cleanup is not needed (no task commits); local implementation is ready for review, but publication remains unapproved and hosted CI is unverified.

## Quick Start method–dataset portability closeout (2026-08-28)

- Current task: make Quick Start select datasets from the same `MethodRequirements × DatasetCapabilities` contract used by real preflight, while preserving formal reproduction recipes and paper training stages.
- Branch/base: `new-cli` at `62892f5`; no branch creation, commit, or push was authorized. The pre-existing `training/runners.py`/`tests/test_jocor.py` working-tree changes and five unrelated untracked files were preserved.
- Checklist: remove the private class-space gate; bind generic class-dependent config; publish noisy/artifact requirements; prove READY→plan/preflight; run real-data/focused/full acceptance. Completed: 5 / 5 (100%).
- Public behavior: formal parity still uses the exact recipe. Generic Quick Start binds the selected dataset's class count, discards stale transition matrices, and delegates all READY/INCOMPATIBLE/NEEDS_INPUT decisions to `ExperimentService` and the runner requirement provider. Importance Reweighting no longer rejects a binary dataset by name; missing binary rates remain explicit user inputs. Noisy-only methods require noisy labels, T-Revision rejects incompatible non-class-dependent noise, and external PCSE retains its exact implemented-variant boundary.
- Input binding: existing CIFAR ResNet/PreActResNet families now consume `input_channels` from `PreparedData.input_spec`, defaulting to three channels. This removes the hidden Fashion-MNIST failure without changing CIFAR formal behavior or any paper objective.
- Real compatibility matrix: CIFAR-10, CIFAR-100 and Fashion-MNIST were checked with clean and symmetric-0.2 selections across all 26 papers. Every menu item marked READY produced a READY plan; other entries remained typed INCOMPATIBLE or NEEDS_INPUT. Real `supervised → GCE → ResNet-34` runs completed one optimization epoch on all three datasets (global steps 1, 2 and 1 respectively) using temporary artifacts.
- Files inspected: Quick Start templates/service/models, runner registry, ExperimentService, ResNet builders, Importance Reweighting config, local dataset catalog, affected tests, and the shared data-flow/progress documents.
- Files modified: `src/lnl_toolbox/quickstart/templates.py`, `src/lnl_toolbox/quickstart/service.py`, `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/service.py`, `src/lnl_toolbox/training/experiment.py`, `src/lnl_toolbox/models/cifar_resnet.py`, `src/lnl_toolbox/algorithms/importance_reweighting/config.py`, `tests/test_quickstart_service.py`, `tests/test_registry.py`, `tests/test_training.py`, `tests/test_cli.py`, `tests/test_importance_reweighting.py`, `web/test_quick_start_integration.py`, `docs/data-flow-guide.md`, and this progress record. Files added: none.
- Tests executed: Quick Start `13/13`; registry `53/53`; Web integration `2/2`; training `75/75`; CLI `89/89`; Importance Reweighting `50/50`; nine affected method suites `257/257`; full unittest `969/969`; Python compile and `git diff --check` passed.
- Blockers: none. Assumption: shortened real runs validate data portability and first-stage execution, not paper numerical reproduction.
- Local checkpoint commits: none. History cleanup: not needed. Push readiness: implementation is reviewable, but commit and publication remain unapproved.
- Collaborator integration: `training/runners.py`, `training/service.py`, `training/experiment.py`, Quick Start files and both shared documents are high-conflict. `tests/test_jocor.py` is pre-existing collaborator work and was not modified by this task.
- Exact next step: human review of the scoped diff; request separate authorization before staging or committing.

## Five startup-boundary fixes (2026-08-25)

- Current task: repair only the five confirmed startup/data-role defects reported by the Web formal-run path.
- Branch/base: `new-cli` at `3a7582d`; no commit or push performed.
- Completed: 5 / 5 code fixes and focused regression checks.
- JoCoR now derives `validation_size` from its data configuration before choosing the validation loader. PDL now derives `official_pdl` from the configured correction/phase variant before loading revision artifacts. The paper algorithms are unchanged.
- CWD's folded CIFAR adapter is correctly treated as a train/test-only source during inspection; the absence of a native validation split no longer aborts preflight.
- MC-LDCE and CA2C requirements and runners no longer request or materialize a validation loader when `validation_size` is zero. They retain validation metrics when an explicit validation role exists.
- MentorNet remains fail-closed: a missing or hash-mismatched MentorArtifact is an external resource readiness failure, not a reason to bypass validation.
- Validation: Ruff passed; data adapters `54/54`; registry `52/52`; training `74/74` with one CUDA skip; CWD data inspection against the existing F-drive CIFAR source returned `ready`; zero-validation role inspection showed no validation role for MC-LDCE/CA2C. Full unittest ran `957` tests: `954` passed, `1` skipped, and the same two baseline failures remain (95-vs-94 YAML count and the pre-existing NoiseManifest error-text mismatch).
- Exact next step: run the formal Web paths for JoCoR, PDL, CWD, MC-LDCE and CA2C with available data/resources, then separately prepare the pinned MentorArtifact/PCSE resources for the remaining external-resource cases.

## Unified data connector final semantic closeout (2026-08-25)

- Current task: close the last validation/TEST and compatibility-origin semantics without changing DataService or paper algorithms.
- Branch/base: `new-cli` at `3a7582d`; no branch creation, commit, or push is authorized.
- Checklist: FINE role declaration; CAL role declaration; train-only progress; final-only TEST; compatibility origin table; focused/full acceptance. Completed: 5 / 6 (83%); code regression is complete, but the all-paper formal-runtime acceptance is not.
- Result: FINE/CAL request validation only when `data.validation_size` or `data.num_val` is positive. Their runners never alias TEST as validation. Epoch rows are train-only or use a real independent validation role; TEST is evaluated once after training and appears only in one final event with leakage flags.
- Compatibility: class-count limits for CWD, Importance Reweighting, PCSE, VolMinNet and unresolved MC-LDCE are reported as `implemented_variant_limit`; FINE/DLD modality and MentorArtifact interface limits use the same origin. Binary Risk and L2RW trusted supervision remain algorithm requirements.
- Files added: none. DataService, algorithms, YAML, CLI/Web/Scratch, taxonomy/guideline, data sources and the four pre-existing untracked files remain unchanged.
- Focused tests: FINE `14/14`; CAL `13/13`; registry `52/52`; training `74/74` with one CUDA skip. Both validation modes, resume final-row replacement and train-only/train+validation SVGs are covered. Ruff and `git diff --check` pass.
- Full regression: `957` tests executed; `954` passed, `1` skipped, and only the two already-recorded baseline assertions failed (95 vs 94 YAML count; manifest namespace error text vs `length`).
- Formal preflight: `24/26` formal recipes are compatible after injecting the existing MentorArtifact. CWD is blocked before training because the folded CIFAR adapter's validation probe raises `folded CIFAR binary view exposes train and test`; PCSE is blocked because no discovered UPM run matches its pinned checkpoint SHA-256. These are pre-existing acceptance-resource/data-adapter blockers outside this task's allowlist. Formal FINE/CAL also cannot be reduced to one stage solely through their current public `trainer.epochs` budget because their fixed warm-up budgets would then exceed the total budget; their unchanged formal protocols were therefore not rewritten for this check.
- Exact next step: separately approve the CWD validation-probe fix and provide the pinned PCSE source run (or approve a formal acceptance protocol that records stage-specific budget overrides). Then run all 26 first real stages and compatible alternate-data cases before marking 6/6.
- Ready for history cleanup/push: no; neither commit nor publication is authorized.

## 26-paper unified data boundary (2026-08-25)

- Current task: separate method data requirements from reproduction protocols, then migrate 26 paper methods without changing their training engines.
- Branch/base: `new-cli` at `3b1675b`; four pre-existing untracked audit/WebUI files are intentionally preserved.
- Checklist: repository/audit baseline; neutral PreparedData contracts; shared-runner migration; specialized role/view migration; remaining dedicated-runner migration; 26-paper formal/alternate-data acceptance.
- Completed: 4 / 6 (67%); baseline audit、backward-compatible contract foundation、shared-runner migration 和 specialized role/view migration 已完成；remaining dedicated-runner migration 已完成 Phase 5A，但该 checklist 项尚未整体完成。
- Contract result: `InputSpec`, `NoiseDescriptor`, `DataProtocol`, `UNLABELED` and `CURRICULUM` are public vocabulary. `PreparedData` exposes input/noise/available-role contracts and fail-closed validation selection. Explicit protocols participate in the data fingerprint; legacy callers retain their prior manifest fingerprint shape.
- Files inspected: data contracts/profile/service, reproduction compatibility layer, all dedicated experiment entry points, runner compatibility requirements, data tests, shared data-flow/file-map documents, taxonomy coverage, and the 26-method data-requirements audit.
- Files modified: phase 1 changed `src/lnl_toolbox/data/contracts.py`, `src/lnl_toolbox/data/__init__.py`, `src/lnl_toolbox/training/data_service.py`, `tests/test_data_adapters.py`; phase 2 changed `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/experiment.py`, `tests/test_registry.py`, `tests/test_training.py`; phase 3 changed `src/lnl_toolbox/data/views.py`, the data export/service, and the data-entry sections of `dld_experiment.py`, `dividemix_experiment.py`, `l2rw_experiment.py`, with contract coverage in `tests/test_data_adapters.py`; phase 5A changed capability declarations, PCSE train-mode data construction, VolMinNet class-count parsing, and their focused tests. All phases update `docs/data-flow-guide.md`, `docs/file-map.md`, and this progress record. Files added: none.
- Shared-runner result: CE/GCE/APL consume only TRAIN, the selected validation role and TEST. GCE/APL accept compatible IMAGE/TABULAR data, native observed-only noise does not require clean labels or a private manifest, while formal generated-noise recipes retain manifest identity. `feature_mlp` consumes tabular `input_spec.feature_dim`; loss/selector/optimizer/epoch loop remain unchanged.
- Specialized role/view result: `PreparedData.view_loader()` projects one named view while preserving target/index, and `subset_loader()` constructs validated dynamic training partitions. DLD now obtains aligned weak/strong loaders from the service; DivideMix obtains per-epoch labeled/unlabeled loaders from the service; L2RW selects its validation loader through the declared requirements. Their feature extraction, diffusion, GMM, MixMatch, meta gradient and checkpoint state are unchanged. Dynamic DivideMix membership deliberately remains a subset operation rather than a static UNLABELED role.
- Tests: focused data contracts pass 57/57; registry passes 46/46; training passes 71/71 with 1 CUDA skip. The added generic tabular GCE and APL cases each completed a real CPU epoch with finite test loss. The phase 2 full unittest ran 942 tests with 2 failures, 2 errors and 1 skip; all four failures are the same pre-existing failures observed before phase 2: stale YAML count, stale noise error text, and two PCSE validation-index/manifest mismatches. Official-format image and tabular fixtures each completed a real one-epoch path in phase 1. Ruff and `git diff --check` pass.
- Blockers: none for the contract foundation. The four pre-existing full-suite failures must remain visible and are outside this phase's allowlist.
- Phase 3 tests: data adapters/service pass 53/53; DLD passes 27/27; DivideMix passes 30/30; L2RW passes 16/16. Ruff and `git diff --check` pass. Full unittest reran 942 tests with the unchanged 2 failures, 2 errors and 1 skip: stale YAML count, stale noise error text, and two PCSE validation-index/manifest mismatches. No phase 3 regression was found.
- Phase 5A result: CDR、DSS、CA2C 和 VolMinNet 的能力声明不再包含无论文依据的 IMAGE/CIFAR/10-or-100-class 限制；PCSE train mode 从 `PreparedData` 获取 dataset、类别数和 feature dimension，external UPM reproduction 路径保持不变。CDR/DSS 在非内置名称的四分类 tabular adapter 上完成真实 CPU epoch；CA2C generic tabular smoke 继续通过；VolMinNet 在四分类 tabular 数据上完成完整 epoch；PCSE generic adapter 完成第一个真实预训练 epoch。算法公式、正式 YAML、CLI、Web、CAL、FINE、MentorNet 和 MC-LDCE `C>=3` 未修改。
- Phase 5A tests: registry 46/46；training 72/72 with 1 CIFAR-data skip；VolMinNet/estimators 49/49；PCSE generic first-stage 1/1；Ruff 通过。完整 unittest 运行 945 项，结果仍为实施前相同的 2 failures、2 errors、1 skip：stale YAML count、stale noise error text 和两项 PCSE independent-validation index/manifest mismatch；没有新增失败。
- Exact next step: propose the next separately approved dedicated-runner batch. Do not mark item 5 complete until all remaining runners are migrated.
- Ready for history cleanup: no; the 26-paper migration is in progress. Ready to push: no.
- Collaborator note: contracts, DataService, runner registry and shared documents are high-conflict surfaces. Phase 3 changes shared `data_service.py` and only the data-entry sections of `dld_experiment.py`, `dividemix_experiment.py`, and `l2rw_experiment.py`. Phase 5A changes only CA2C's capability declaration, not its algorithm/runner; FINE/UPM and all loss, selector, recipe, CLI and Web files remain unchanged.

## Web configuration workflow and semantic Sweep (2026-08-22)

- Current task: carry one recipe/project-YAML context through paper guidance, YAML editing, execution, parameter Sweep, and result management.
- Branch: `codex/cli`; base commit: `7a8c9be`; the committed beginner-tutorial baseline was preserved.
- Checklist: workflow audit; project-YAML Sweep API; Registry-enforced matrix validation; cross-module UI links; focused/browser acceptance; full regression and documentation.
- Completed: 6 / 6 (100%); implementation, focused tests, real browser workflow, runtime Sweep evidence, documentation, full regression, and scope audit are complete.
- Behavior: paper recipes can be prepared for formal execution, copied into a safe project YAML, or passed directly to Sweep. Saved project YAMLs can validate, prepare a run, enter Sweep, and pass the Sweep output to run management without losing source context.
- Sweep semantics: both recipe and project-YAML sources use the revised four-level Registry. Locked/unknown/type-incompatible paths fail server-side; paper parameters show provenance and reproduction impact and are reported as deviations in every affected planned run.
- Safety correction: Web reproduction metadata is stored under `meta.web_parameter_record`. The training core's top-level `parameter_record` remains reserved for hyperparameter-sampling records, preventing Web-saved Binary Risk configurations from failing in `ParameterRecord.from_dict()`.
- Runtime evidence: a project-YAML batch/LR 2×2 Sweep completed 4/4 runs; after the metadata correction, a Binary Risk 1×2 paper-parameter Sweep completed 2/2 runs and reported `risk.rho_positive` as the paper-deviation path.
- Validation: Web 39/39 and full unittest 845/845 passed; Ruff, embedded JavaScript syntax, and `git diff --check` passed.
- Files changed: Web backend/page/test and four approved shared documents. No CLI, Sweep engine, recipe, Registry, algorithm, runner, model, dataset adapter, or training default changed.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; changes are scoped to configuration context, Sweep authorization, and documentation.

## Web beginner tutorial workflow (2026-08-22)

- Current task: implement `docs/LNL-Toolbox-简明操作教程.md` as an executable Web workflow rather than a single action dropdown.
- Branch: `codex/cli`; base commit: `f99ed98`; the worktree was clean at task start.
- Checklist: tutorial audit; versioned six-step contract; guided UI and quick-mode compatibility; focused tests; real browser execution; full regression and documentation.
- Completed: 7 / 7 (100%); implementation, focused tests, JavaScript/Ruff checks, real browser execution, documentation, full regression, and final scope review are complete.
- Behavior: the default page advances only after successful `doctor → list → validate → dry-run → run`; the final step inspects resolved config, metrics, files, phase, epoch target and `last.pt` before enabling resume. A completed target is marked `not-needed` instead of launching a no-op resume.
- Runtime evidence: the real Web page used registered `local-cifar10`, ran `cifar10-symmetric-ce-smoke` for its configured 2 epochs, and wrote `artifacts/runs/web-tutorial-acceptance`. Final evidence was epoch `2/2`, validation accuracy `0.16796875`, test accuracy `0.16015625`, complete config/metric/checkpoint files, and no browser errors.
- Files changed: Web backend/page/test, Web README, concise tutorial, file map, and this progress record. No CLI, runner, algorithm, data service, recipe, or parameter registry changed.
- Validation: Web 34/34 and full unittest 845/845 passed; Ruff, JavaScript syntax, and `git diff --check` passed. Browser acceptance completed all six tutorial steps and correctly treated resume as unnecessary after reaching epoch `2/2`.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; edits are limited to the beginner tutorial contract and presentation.

## Revised Web parameter permission policy (2026-08-21)

- Current task: make `lnl_parameter_metadata_registry_revised.yaml` the only active Web parameter-selection policy without changing recipes, training code, or the page layout.
- Branch: `codex/cli`; base commit: `840ace1`; the revised registry was the only untracked file at task start.
- Checklist: compare registries; switch authority; test editable/locked boundaries; focused regression; full regression and documentation.
- Completed: 6 / 6 (100%); registry comparison, authority switch, boundary tests, focused/full regression, browser acceptance, and documentation are complete.
- Registry scope: all 26 methods and all 1,231 existing paths are retained. The revised levels contain 120 basic, 137 paper, 809 advanced, and 165 locked parameters.
- Safety boundary: supported experimental choices are editable under the revised policy, while structural/runtime/integrity fields such as `execution.runner` remain server-enforced locks.
- Files changed: revised registry, Web backend/test, and three shared documents. No page layout, recipe, algorithm, runner, model, dataset adapter, or training default changed.
- Validation: Web 32/32 and full unittest 845/845 passed; Ruff passed. Browser acceptance confirmed Binary Risk exposes 6 basic, 6 paper, 6 advanced, and 4 locked fields; `model.name` and both noise rates are editable while `execution.runner` remains read-only.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: `web/command_console.py`, Web tests, and shared documents are high-conflict integration surfaces; edits are limited to the registry authority and permission assertions.

## Web parameter metadata registry (2026-08-21)

- Current task: make the 26-paper parameter registry the Web authorization, grouping, provenance, and reproduction-deviation source.
- Branch: `codex/cli`; base commit: `4ace09c`; the supplied registry was the only untracked file at task start.
- Checklist: registry audit; path/binding repair; backend schema and enforcement; four-level UI; focused tests; full regression and documentation.
- Completed: 7 / 7 (100%); implementation, focused Web tests, full regression, lint, JavaScript syntax, and browser acceptance are complete.
- Registry audit: 26 methods and 99 paper parameters were retained; stale T-Revision, DivideMix, split, scheduler, model-list, PCSE, DSS and DLD paths were repaired and each method is bound to its catalog formal recipe.
- Safety: locked values are visible but cannot be changed by parameter patches or complete-YAML API requests. Paper-level changes require acknowledgement and persist `modified_from_paper`, the formal recipe, and changed-from/value evidence in `meta.web_parameter_record`; the top-level `parameter_record` remains reserved for training hyperparameter sampling.
- Files changed: the supplied registry, Web backend/page/test, and three shared documents. No recipe, algorithm, runner, model, dataset adapter, catalog, or training default changed.
- Validation: Web 31/31 and full unittest 845/845 passed; Ruff, JavaScript syntax, and `git diff --check` passed. Browser acceptance confirmed four groups, read-only locked fields, paper provenance, and the paper-deviation warning.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: Web files and shared documents are high-conflict integration surfaces; changes are limited to parameter metadata presentation and authorization.

## Web matrix sweep and result visualization (2026-08-21)

- Current task: expose typed Cartesian parameter sweeps, native Windows path selection, structured CLI tables, and run metrics/curves in the local Web console.
- Branch: `codex/cli`; base: clean working tree at task start.
- Checklist: Sweep CLI; result contract; Web APIs; Web controls; focused tests; full regression and documentation.
- Completed: 6 / 6 (100%); implementation, browser acceptance, focused tests, documentation, and final regression are complete.
- Behavior: `--matrix PATH=JSON_ARRAY` preserves types and defaults to the recipe seed; Web preflights and lists every combination before execution. Run management reads complete and partial artifacts, uses Sweep manifests for authoritative status, and overlays up to five curves by default.
- Follow-up UI: long run lists can be filtered by name, method, status, or seed and collapsed without clearing selected curves. Native path dialogs fall back to the project root when the current value is empty, relative, or invalid.
- Paper guidance follow-up: result lists are compare-only; the paper page now explains the method lifecycle and its mapping to YAML fields and implementation modules, with a direct YAML editor shortcut instead of exposing profile/variant identifiers.
- Resume inspection follow-up: Web resume now requires a read-only inspection of the selected run/checkpoint and shows resolved configuration, file/checkpoint inventory, current epoch/phase, stage schedule, best metrics, and explicit readiness blockers before enabling the command.
- Browser evidence: the real page generated the two-dimensional batch/LR command, the backend planned exactly four runs, the run manager loaded 144 existing run directories, and structured Smoke recipes rendered as a table.
- Files changed: Sweep/results contracts, unified CLI, Web backend/page, three focused tests, and three shared documents. No algorithm, runner, model, adapter, YAML, or training default changed.
- Validation: Sweep 7/7, Web 26/26, unified CLI 43/43, and full unittest 845/845 passed. Ruff, JavaScript syntax, `git diff --check`, and a no-training four-combination Sweep dry-run passed.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; commit and push are not authorized.
- Collaborator note: `cli/main.py`, Web files, and shared documents are high-conflict integration surfaces; edits are scoped to CLI/Web presentation and result inspection.

## Curated public recipe surface (2026-08-21)

- Current task: keep all reproducibility YAMLs while exposing only useful templates to ordinary users.
- Branch: `codex/cli`; base commit: `6ed461f` with the approved schema-v1 working tree preserved.
- Checklist: public manifest; catalog/CLI filtering; Web filtering; tests; docs; final regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Public behavior: `lnl list experiments` and the Web beginner page expose four curated templates. `--all`, direct recipe lookup, the paper catalog, and `/recipe` retain access to the complete catalog.
- Configuration safety: no experiment YAML, runner, algorithm, dataset adapter, or training default was changed.
- Validation: unified CLI 38/38, Web 21/21, and full unittest 833/833 passed. Ruff, Web JavaScript syntax, and `git diff --check` passed.
- Runtime evidence: the public `cifar10-clean-smoke` template completed one Fashion-MNIST GPU epoch through the registered alias. Browser inspection showed exactly four beginner templates, while `/recipe` retained all 64 configuration sources.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; no commit or push is authorized.
- Collaborator note: the recipe manifest, `catalog.py`, CLI and Web files are shared integration surfaces.

## Versioned YAML contract migration (2026-08-21)

- Current task: audit every YAML parameter category, preserve the old files, and make active recipes portable and directly runnable.
- Branch: `codex/cli`; base commit: `6ed461f`.
- Checklist: archive; schema/normalization; 92-file migration; Binary Risk/MentorNet adaptation; tests/docs; final regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Configuration result: 92 / 92 active YAMLs use schema v1; 76 are complete experiment/auxiliary configurations and 16 are explicit fragments; all 64 built-in recipes remain internally discoverable and four curated templates are public by default.
- Data result: machine-local CIFAR/MNIST roots are removed from active recipes and resolved through a unique local registration; the UCI Heart engineering recipe retains its single-file fallback path required by its strict source contract.
- Legacy recovery: `archive/configs-legacy-2026-08-21/manifest.json` records 92 matching SHA-256 snapshots.
- Tests executed: config schema 4/4, Binary Risk 5/5, MentorNet 9/9, DataService 12/12, ExperimentService 3/3, runner planning 7/7, unified CLI 37/37, and the full unittest suite 832/832. Ruff and `git diff --check` passed.
- Runtime evidence: rootless `cifar10-clean-smoke` resolved the unique `local-cifar10` registration, loaded 50,000 train / 10,000 test samples, and completed one GPU epoch. Its persisted resolved configuration remains canonical and does not contain runtime-only aliases.
- Exact next step: human diff review; commit only after separate authorization.
- Local checkpoint commits: none; no commit or push is authorized.
- Collaborator note: `catalog.py`, `training/service.py`, `training/data_service.py`, all active YAMLs, and shared documents are high-conflict files.

## Dataset-independent training selection (2026-08-21)

- Current task: remove the Web/CLI assumption that every non-tabular dataset verifies through a CIFAR recipe.
- Branch: `codex/cli`; base commit: `2f3ee42` with pre-existing CI/fixture working-tree edits preserved.
- Checklist: automatic verification profile; CLI compatibility; separate Web training-template/data selectors; real Fashion-MNIST verification; switched-template training; tests and documentation.
- Completed: 6 / 6 (100%). Implementation, real-data validation, browser validation, and final regression are complete.
- Real evidence: registered Fashion-MNIST loaded 60,000 train / 10,000 test samples and completed one automatic GPU epoch without a recipe. `cifar10-clean-smoke --data fashion-mnist` also completed one epoch, proving that template and data selection are independent.
- Browser evidence: selecting Fashion-MNIST in the beginner page generated `lnl validate --recipe cifar10-clean-smoke --data fashion-mnist --check-data`; data verify generated `lnl data verify fashion-mnist` and hid the recipe field.
- Files added: none.
- Tests executed: DataService 10/10, unified CLI 37/37, Web 21/21, full unittest 826/826; Ruff and `git diff --check` passed.
- Exact next step: human diff review; no commit or push is authorized.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `cli/main.py`, `training/data_service.py`, Web UI, and shared documents are high-conflict; no paper runner, dataset adapter, model, or recipe was changed.

- Current task: unified experiment service, result contract, overrides, sweep, comparison, runner introspection, and CI.
- Branch: `codex/cli`
- Base commit: `5e9c19b`
- Checklist: 6 infrastructure items.
- Completed: 6 / 6 (100%).
- In progress: none; final handoff pending.
- Files inspected: CLI, runner registry, experiment entry, evaluation exports, project metadata, architecture/data-flow/file-map documents, and related tests.
- Files modified: `pyproject.toml`, CLI/training/evaluation integration modules, tests, and three existing documents.
- Files added: quality workflow; config override, planning, results, service, sweep, comparison modules; six focused test files.
- Local checkpoint commits: none.
- Tests executed: focused tests passed; unified CLI 30/30; full unittest 734/734; coverage full suite passed at 74%; Ruff passed; `git diff --check` passed; wheel built, installed in a temporary environment, imported from that environment, and exposed the unified `lnl --help` commands.
- Blockers: none.
- Assumptions: sweeps are sequential; completed legacy result files with a recognized test metric remain resume-compatible.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: yes; no checkpoint cleanup is needed.
- Ready to push: ready after an explicit commit and publication request.
- Collaborator note: `cli/main.py`, `training/runners.py`, `docs/file-map.md`, and `pyproject.toml` are shared/high-conflict files; changes are scoped to the approved infrastructure contracts.

## Unified data protocol migration (2026-08-18)

- Current task: one dataset registry, one preparation service, and one data entry for all paper runners.
- Branch: `codex/cli`.
- Base commit: `1ed54c4`.
- Checklist: protocol/registry; runner migration; real-noise adapters; manifest/checkpoint identity; tests; documentation.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Implemented datasets: CIFAR-10/100, CIFAR binary view, CIFAR-10N/100N, MNIST/Fashion-MNIST, Clothing1M, Animal-10N, UCI binary, synthetic binary/multiclass.
- Implemented safety: no automatic downloads, no clean target in train batches, stable global indices, deterministic epoch loaders, run-local data manifest, checkpoint fingerprint validation.
- Runner status: all listed experiment runners call `prepare_experiment_data()`; static gate rejects direct concrete loader imports and local `DataLoader` construction.
- Tests: unified data tests and affected workflow suites pass; full unittest passed 797 / 797; 63 directly available built-in recipes validate and construct their runner successfully. The conditional PCSE reproduction recipe correctly refuses construction without `LNL_PCSE_SOURCE_RUN`.
- Exact next step: human diff review, then commit only if requested.
- Collaborator note: `training/experiment.py`, `training/checkpoint.py`, `cli/main.py`, and the four documentation files are high-conflict shared files; edits are limited to generic data construction and identity validation.

## Official dataset-format correction (2026-08-19)

- Current task: verify installed and deferred datasets against primary dataset releases and official paper repositories.
- Branch: `codex/cli`; base commit: `1ed54c4`.
- Checklist: CIFAR-N safe loading; MNIST IDX compatibility; Clothing1M official metadata; ANIMAL-10N official layouts; focused tests; full regression.
- Completed: 6 / 6 (100%); implementation and validation are complete.
- Real-data evidence: local CIFAR-10N and CIFAR-100N each load 50,000 train / 10,000 test samples; local Fashion-MNIST IDX.GZ loads 60,000 train / 10,000 test samples.
- Focused evidence: official-format adapter and unified-service tests pass 13 / 13; affected data/CLI tests pass 47 / 47; full unittest passes 802 / 802.
- Exact next step: human diff review, then commit only if requested.
- Collaborator note: no runner, algorithm, model, configuration, checkpoint, or registry interface changed.

## CLI experiment workflow v2 (2026-08-20)

- Current task: align the documented CLI workflow, full dry-run preflight, matrix sweep/status, and grouping-aware comparison/reporting.
- Branch: `codex/cli`; base commit: `ccc9dc5`.
- Checklist: documentation; shared preflight; Sweep v2; Compare v2; final CLI/regression validation.
- Completed: 5 / 5 (100%); implementation and final validation are complete.
- Files inspected: README, development/data-flow guidance, CLI, experiment service, sweep, comparison, override engine, Result Contract, and focused tests.
- Files modified: README, development/file-map/progress documents, CLI, experiment service, sweep/comparison modules, and four existing test files.
- Files added: none.
- Local checkpoint commits: none.
- Tests executed: Task 1 full suite `802/802`; Task 2 focused tests and full suite `804/804`; Task 3 focused tests and full suite `808/808`; Task 4 focused tests and full suite `812/812`; final focused tests, Ruff, CLI help/dry-run checks, `git diff --check`, and final full suite `812/812` all passed.
- Blockers: none.
- Assumptions: sweeps remain sequential; duplicate seeds and duplicate resolved matrix configurations fail; strict compare excludes leakage-marked runs.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: no checkpoint cleanup is needed.
- Ready to push: ready after explicit commit and publication approval; neither operation has been performed.
- Collaborator note: `README.md`, `cli/main.py`, and shared progress documents are high-conflict files; changes are limited to CLI experiment infrastructure.

## Quality workflow dry-run preflight correction (2026-08-20)

- Current task: keep default dry-run data preflight while making planning-only tests and CI smoke independent of local CIFAR files.
- Branch: `codex/cli`; base commit: `ee88b26`.
- Checklist: enumerate dry-run calls; reproduce clean-runner failures; add explicit planning-only bypasses; run full validation.
- Completed: 4 / 4 (100%).
- Pre-change evidence: a clean Git archive without `data/cifar10` reproduced 10 failures across unified CLI, Co-teaching, DLD, LEND, and UPM planning tests.
- Implementation: planning-only CIFAR dry-runs now pass `--no-check-data`; the missing-data semantic test remains unchanged; the quality workflow uses the same explicit bypass for its final smoke.
- Tests executed: Ruff passed; clean-runner focused tests passed; coverage full suite passed `812/812` with 75% coverage; unified CLI passed `33/33`; `lnl --help` and the explicit no-data dry-run passed; sdist and wheel builds passed.
- Files added: none.
- Local checkpoint commits: none.
- Blockers: none.
- Exact next step: human diff review, then commit only the approved files if requested.
- Ready for history cleanup: no cleanup needed.
- Ready to push: ready after explicit commit and publication approval.
- Collaborator note: `.github/workflows/quality.yml` is shared; no source, algorithm, data-preflight, Result Contract, or checkpoint behavior changed.

## Local dataset catalog and executable format evidence (2026-08-20)

- Current task: machine-local dataset registration, Web controls, and training-backed format verification.
- Branch: `codex/cli`; base commit: `3942558`.
- Checklist: local catalog/CLI; grayscale and real-noise compatibility; Web controls; official-format fixtures and real-data runs; final regression/documentation.
- Completed: 5 / 5 (100%); implementation and validation are complete.
- Files added: `data/local_catalog.py`, `tests/test_local_data_catalog.py`, `tests/test_dataset_training_fixtures.py`.
- Training evidence: official-format temporary fixtures completed 1 epoch for MNIST, Fashion-MNIST, Clothing1M, Animal-10N, and UCI Heart. Real F-drive data completed 1 epoch for CIFAR-10, CIFAR-100, CIFAR-10N, CIFAR-100N, Fashion-MNIST, and the CIFAR airplane/automobile view. Synthetic binary and multiclass runners produced epoch metrics.
- State semantics: registration and layout inspection never imply trainability; only a completed epoch plus data manifest is recorded as `training_verified`; a changed catalog source signature produces `verification_stale`.
- Tests executed: local catalog `2/2`, Web `15/15`, dataset training fixtures `5/5`, unified CLI `35/35`, noisy CE `11/11`, clean baseline `5/5`, torch training `39/39`; Ruff passed; full unittest passed `821/821`; `git diff --check` passed.
- Blockers: none.
- Exact next step: human diff review; commit only after explicit authorization.
- Local checkpoint commits: none. No commit or push is authorized.

## Web dataset registration usability pass (2026-08-20)

- Current task: validate and improve the full Web lifecycle for local dataset registration.
- Branch: `codex/cli`; base commit: `77cd274` plus the existing DataService working-tree changes.
- Checklist: real register/inspect/verify/remove walkthrough; operation-specific fields; actionable status and summaries; backend error JSON; two-step removal; regression and documentation.
- Completed: 6 / 6 (100%). Implementation, real-browser lifecycle validation, cleanup, and final regression are complete.
- Real evidence: a temporary CIFAR-10 alias loaded 50,000 train and 10,000 test samples, completed a one-epoch verification, and was removed without deleting source data.
- Tests executed: Web tests passed 20/20; full unittest passed 824/824; Ruff on `src tests web`, JavaScript syntax validation, and `git diff --check` passed.
- Files added: none.
- Exact next step: human diff review; no commit or push is authorized.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `README.md`, `cli/main.py`, Web files, and shared progress/file-map documents are high-conflict; changes are scoped to local data source selection and evidence status.

## Unified data management facade and Web API (2026-08-20)

- Current task: Data Phase 1/5/6 — one management facade for CLI, Web and experiment preflight.
- Branch: `codex/cli`; base commit: `77cd274`.
- Checklist: status contract; CLI list/status/path/inspect/verify; shared preflight; direct Web data API; CI gate; documentation and regression.
- Completed: 6 / 6 (100%).
- Public behavior: `missing/incomplete/ready` describes adapter-backed readiness; one-epoch `training_verified` evidence remains separate. `inspect` loads train/test, while `verify` performs the same inspection before training.
- Integration: doctor, validate, dry-run, run and sweep reach data validation through `ExperimentService` and the injected `DataService`; existing paper runners retain the compatibility `prepare_experiment_data()` entry.
- Tests executed: affected suites passed 76/76; Web focused tests passed 17/17 including a real local HTTP API request; full unittest passed 823/823; Ruff on `src tests web`, embedded Web JavaScript syntax, and `git diff --check` passed.
- Blockers: none. Conda emits a pre-existing missing OpenCL vendor temp-file warning before commands, but the `pytorch` environment and all tests complete successfully.
- Exact next step: human diff review; commit only after explicit authorization.
- Local checkpoint commits: none. No commit or push is authorized.
- Collaborator note: `cli/main.py`, `training/service.py`, `.github/workflows/quality.yml`, Web files, and shared documents are high-conflict; edits are limited to the approved data-management integration.

## Original Web console and Recipe subpage (2026-08-20)

- Current task: retain the original single-page Web console and expose the existing Recipe/YAML editor at `/recipe`.
- Branch: `codex/cli`; base commit: `77cd274` plus the completed DataService working-tree changes above.
- Checklist: restore original layout; remove unapproved workspace tabs; `/recipe` entry; `lnl web`; HTTP/CLI tests; documentation.
- Completed: 6 / 6 checklist items (100%).
- Public behavior: `lnl web` starts `127.0.0.1:8765` and opens the original console at `/`; `lnl web --no-open` suppresses browser launch; `/recipe` opens the Recipe/YAML editor directly.
- Files added: none. One route-aware HTML asset is intentionally reused to avoid duplicated UI logic.
- Validation: Web tests 18/18; unified CLI tests 36/36; full unittest 824/824; Ruff, JavaScript syntax and diff checks passed; `/` and `/recipe` were inspected in the app browser.
- Exact next step: human diff review, then commit only with separate authorization.
- Local checkpoint commits: none. No commit or push is authorized.

## CI NumPy compatibility and workflow cleanup (2026-08-21)

- Current task: diagnose the latest `quality` failure and remove redundant CI work without weakening required gates.
- Branch: `codex/cli`; base commit: `2f3ee42`.
- Checklist: inspect remote jobs and logs; reproduce the failure across NumPy versions; correct the fixture; audit every workflow step; run focused, full, Web, lint, packaging, and artifact-install checks.
- Completed: 5 / 5 (100%).
- Root cause: the Animal-10N test fixture passed pixel value `716` directly to a `uint8` array. NumPy 1.26 silently wrapped it to `204`, while NumPy 2.x correctly raised `OverflowError` in both Python 3.10 and 3.12 CI jobs.
- Implementation: fixture values now use explicit modulo-256 arithmetic; GitHub Actions use Node-24-based `checkout@v6` and `setup-python@v6`; redundant pip upgrades, duplicate unified-CLI execution, and duplicate editable-install help smoke were removed. The wheel job keeps required training dependencies but no longer installs unrelated test/coverage/lint dependencies.
- Validation: NumPy 2.3.5 boundary check passed; dataset fixtures passed 5/5; ordinary full unittest passed 824/824; Web tests passed 20/20; Ruff and workflow YAML checks passed; wheel and sdist both built, installed into isolated environments, resolved the installed package, and passed help/list/validate CLI checks.
- Local limitation: coverage-instrumented tests on Windows intermittently hit an unrelated `PermissionError` while atomically replacing a sweep manifest. The same tests pass without coverage and passed in the failed Ubuntu CI run; no training or sweep code was changed under this task.
- Files modified by this task: `.github/workflows/quality.yml`, `tests/test_dataset_training_fixtures.py`, and this progress record. Files added: none.
- Exact next step: review the three-file diff, then commit only with separate authorization.

## 26-paper formal YAML exposure (2026-08-21)

- Current task: expose exactly one formal training YAML per paper in Web “New YAML”, while retaining four beginner templates and the full Recipe editor.
- Branch: `codex/cli`; base commit: `6ed461f` plus the pre-existing approved working-tree changes.
- Checklist: official protocol audit; missing formal YAMLs; paper-to-recipe defaults; safe dedicated-runner clone; Web mode split; focused/browser/full regression.
- Completed: 6 / 6 (100%).
- Public behavior: paper mode lists 26/26 papers and clones the selected formal configuration; generic mode keeps the reusable supervised loss/selector composer.
- Fidelity boundary: CNLCU and VolMinNet new defaults are `paper_protocol`; DLD remains `paper_oriented` because the official ViT-L/14 feature path is not present in the toolbox.
- Modular changes: CNLCU architecture is a reusable model, its decay is a generic scheduler, and no paper-name branch was added to the common experiment runner.
- Focused validation: CNLCU 2/2, DLD 4/4, VolMinNet 6/6, schema 4/4, unified CLI 40/40, Torch training 41/41, Web 21/21; browser interaction verified all 26 options and both creation modes.
- Final validation: full unittest 840/840; Web 21/21; runner package-data 7/7; Ruff passed; CNLCU, DLD and VolMinNet formal-config dry-runs passed; app-browser interaction verified all 26 options and both creation modes.
- Exact next step: human diff review and optional commit under separate authorization; no commit or push is authorized.

## Web YAML creation/editor closeout (2026-08-21)

- Current task: close the broken gap between “New YAML” and the Recipe editor.
- Branch: `codex/cli`; base commit: `6ed461f` plus the approved working-tree changes.
- Checklist: path-based load API; validated full-text save; 4+26 editor sources; automatic open after create; project-YAML reload; browser and regression verification.
- Completed: 6 / 6 (100%).
- Public behavior: a paper YAML created from the main page opens immediately in the editor; the textarea is editable; save, validate and run target the generated project YAML. The editor exposes four beginner templates and all 26 paper defaults without making project files built-in recipes.
- Safety: paths remain restricted to the repository, built-in recipes cannot be overwritten, and full YAML text is parsed and passed through `validate_config()` before writing.
- Browser evidence: created a CNLCU project YAML, automatically opened it, changed seed 1 to 23/24, saved it, reloaded it by project path, and completed `lnl validate`; the editor contained 31 options (4 templates + 26 paper defaults + the current project YAML).
- Final validation: Web 23/23, full unittest 840/840, Ruff, embedded JavaScript syntax and `git diff --check` passed.
- Cleanup: the browser-created project YAML was removed after validation; no generated training or configuration artifact remains.
- Exact next step: human diff review and optional commit under separate authorization; no commit or push is authorized.

## 26-paper noise compatibility preflight (2026-08-22)

- Current task: add the missing compatibility requirements for 14 papers without changing algorithms, experiment runners, adapters, YAML, CLI or Web.
- Branch/base: `codex/cli` at `7eaa452`.
- Checklist: declarative config-input contract; component-driven shared-runner providers; dedicated-runner providers; 26-paper/native-noise regression; full validation.
- Completed: 5 / 5 (100%).
- Public behavior: all 26 formal paper recipes now publish config-specific requirements during preflight. Missing rate pairs, known-T matrix, trusted supervision or external artifacts produce stable requirement codes before runner invocation.
- Safety boundary: compatibility describes the current executable path. Observed-only native noise is rejected when that path requires a manifest or aligned clean/noisy evidence; no formula, lifecycle, data adapter or clean-label boundary changed.
- Tests executed: compatibility 23/23, experiment service 3/3, unified CLI 47/47, full unittest 878/878; focused Ruff and `git diff --check` passed.
- Files added: none. Blockers: none. Local checkpoint commits: none.
- Exact next step: human review of the eight-file diff; commit or push requires separate authorization.
- Collaborator note: `training/runners.py`, `training/service.py` and shared documents are high-conflict; changes are limited to requirements providers, preflight validation and documentation.

## Web dataset-aware YAML workflow (2026-08-22)

- Current task: remove the duplicate YAML destination, make the editor contextual, and surface dataset/method compatibility while constructing YAML.
- Branch/base: `codex/cli` at `7eaa452`, layered on the approved uncommitted 26-paper compatibility work.
- Checklist: concrete-recipe batch query; server-side save gate; one-path editor; draft-preserving module switch; dataset-to-YAML link; focused/browser/full regression; documentation.
- Completed: 6 / 7 (86%); focused service 4/4 and Web 47/47 tests pass, and browser checks confirm CIFAR disables Binary Risk/FINE while enabling Loss Correction, a single YAML path, automatic editor collapse, and draft restoration.
- Safety boundary: dataset selection is a runtime `--data` override and is labeled as migration training; locked paper data/noise protocol fields, algorithms, runners, adapters and formal YAML remain unchanged.
- Files added: none. Local checkpoint commits: none. Push is not authorized.
- Exact next step: run full unittest/Ruff/final browser check, then update this record to 7/7 and hand off the reviewed diff.

## Dataset-first workflow 输入重构（2026-08-23）

- Current task: 按 Dataset-first 计划区分适配器事实、UNKNOWN 数据集声明和具体 recipe 实验输入。
- Branch: `codex/cli`; base: 当前工作区既有兼容性与 Web 修改；未提交、未推送。
- Checklist: adapter semantic hints；UNKNOWN-only 声明；具体 formal recipe；方法先验配置注入；实际预训练资源；Web 三段式 UI；测试与文档。
- Completed: 7 / 7（100%）。
- Public behavior: Web 不再允许把 method prior 或 pretrained role 写入 dataset catalog；正式 recipe 必须显式选择；兼容结果返回真实 input paths；原生 Clothing1M/Animal-10N 语义由 adapter hints 提供。
- Validation: Web `test_command_console.py` 47/47；数据适配器 46/46；Python 编译检查通过。完整 registry 测试当前受仓库既有缺失模块 `test_t_revision_workflow` 阻塞，未归因于本任务。
- Files modified in this task: `README.md`, `web/README.md`, `web/index.html`, `web/command_console.py`, `docs/data-flow-guide.md`, `docs/development-guide.md`, `docs/file-map.md`, `docs/toolbox-modularization-progress.md`, `src/lnl_toolbox/data/profile.py`, `src/lnl_toolbox/data/real_noise.py`, `src/lnl_toolbox/training/data_service.py`, `src/lnl_toolbox/training/compatibility.py`, `src/lnl_toolbox/training/service.py`, `tests/test_data_adapters.py`, `tests/test_registry.py`, `web/test_command_console.py`。其中多份文件已有协作者工作，本次只做局部增量。
- Exact next step: review final diff and resolve the pre-existing merged-test import issue before considering commit/push.

## 26 篇论文统一数据接头收口（2026-08-25）

- Current task: 以 `RunnerSpec` 为唯一 method requirement 真源，让 compatibility 与真实 runner 共享同一个 `DataRequirements`，并仅解除不改变现有数学流程的数据限制。
- Branch/base: `new-cli` at `3382934`; no branch creation, commit, or push is authorized.
- Checklist: canonical requirement source；RunnerSpec→runner identity；第一批六方法；其余二十方法；label/role boundary；tests/docs。Completed: 6 / 6（100%）。
- Public contract: `RunnerSpec.data_requirements(config)` 同时供 compatibility 与 `RunnerSpec.invoke()` 使用；`prepare_experiment_data(..., requirements=...)` 保持显式必填，DataService 不认识 method/runner。
- Safe removals: 可由 role/view/model input adapter 支持的 dataset-name、CIFAR、IMAGE、固定类别数和手工 feature snapshot 限制已解除。CWD binary、VolMinNet `C>=3`、Binary Risk/Importance binary、DLD multi-view、FINE robust strong-view、L2RW trusted supervision 等仍按 `implemented_variant_limit` 或算法 requirement 保留；没有新增论文 variant 或数学流程。
- Label boundary: noisy TRAIN 始终使用 observed target 且 batch 不暴露 clean label；clean/trusted labels 只在显式 evaluation、trusted/meta/curriculum role 使用。协议 TEST source 可提供 clean evaluation ground truth，TRAIN 派生 clean validation 仍须真实 clean target。
- Focused validation: registry `49/49`；data adapters `54/54`；CLI `89/89`；Quick Start `10/10`；training `72/72`（1 skipped）；estimators `49/49`；各 dedicated method suites已通过。
- Full validation: `949` tests executed; `946` passed, `1` skipped, and the two known baseline assertions below failed. All focused suites affected by this task pass.
- Known baseline failures outside this task: `test_config.py` still hard-codes 94 YAMLs although 95 are tracked; `test_noise.py` expects a generic length error while manifest validation raises the more specific out-of-namespace index error. Neither file nor noise/config implementation was modified.
- Files added: none. Four pre-existing untracked WebUI audit files remain untouched.
- Local checkpoint commits: none.
- Exact next step: human review of the final diff and the two explicitly recorded baseline failures; commit or publication requires separate authorization.
- Collaborator note: `training/runners.py`, `training/compatibility.py`, `training/data_service.py`, `training/experiment.py`, and shared docs are high-conflict; algorithms, losses, selectors, estimators, reproduction YAML, CLI/Web/Scratch, and legacy evidence runners remain unchanged.

## Quick Start 与 validation 语义收口（2026-08-28）

- Current task: 修复 Quick Start exact formal recipe、typed unsupported split、TEST-as-validation、JoCoR validation-size 和 PDL variant predicate 问题；禁用不可选方法卡片。
- Branch/base: `new-cli` at `3a7582d`; 不创建分支、不提交、不推送；工作区原有未提交内容保持不动。
- Completed: 7 / 7（100%）：exact matcher；typed split contract；无 validation 时不伪造 TEST；JoCoR；PDL predicate；Web 状态保护；回归验证（仅剩一项既有断言文本不一致）。
- Public behavior: Quick Start 的 `clean/native` 语义不再混同，formal 命中返回真实 recipe id；validation 仅来自 native validation 或 TRAIN 派生 split，零 validation 时 `PreparedData.validation_split=None` 且 manifest 省略 validation。
- Focused validation: data adapters 57/57；Quick Start service 11/11；PDL 20/20；JoCoR 7/7；Web assets/integration 6/6；CLI 89/89；data-contract 及 training focused suites通过。
- Full validation: 963/963 tests 通过；`test_noise` 的错误文本断言已更新为匹配当前更具体的 index-namespace 错误。未放宽生产代码的 typed-exception 边界。
- Files modified in this task: `src/lnl_toolbox/quickstart/templates.py`, `src/lnl_toolbox/quickstart/service.py`, `src/lnl_toolbox/data/contracts.py`, `src/lnl_toolbox/data/sources.py`, `src/lnl_toolbox/data/cifar_n.py`, `src/lnl_toolbox/data/real_noise.py`, `src/lnl_toolbox/training/data_service.py`, `src/lnl_toolbox/training/multi_model_experiment.py`, `src/lnl_toolbox/training/runners.py`, `src/lnl_toolbox/training/instance_transition_experiment.py`, `web/assets/quick_start.js`, `tests/test_data_adapters.py`, `tests/test_data_capability_contract.py`, `tests/test_experiment_service_data_contract.py`, `tests/test_pdl.py`, `tests/test_quickstart_service.py`, `tests/test_training.py`, `web/test_quick_start_assets.py`, `web/test_quick_start_integration.py`, and this progress record. No new files.
- Known collaborator scope: shared `training/data_service.py`, `training/runners.py`, Quick Start assets and progress documentation are high-conflict; no algorithm/loss/selector implementation was changed.
- Exact next step: human review；本轮功能和回归均已收口。无 commit 或 push 授权。

## CLI 启动、展示与交互编辑（2026-11-03）

- Branch/base: `scratch` / `01c1e54`；本轮未提交、未推送。原有 `.gitignore`、`data/cifar_n.py`、`training/dual_t_evidence_experiment.py` 及远端运维未跟踪文件保持不动。
- 实现：根包、core、data、training 使用兼容公开名称的惰性导入；CLI 重依赖进入命令后加载。`show` 与 `papers show` 共用入口；终端显示编辑菜单，非终端只输出摘要。新增中英文偏好与参数解释，列表保留论文来源链接。公开 `validate` 共用原运行验证路径；旧 Web review/dry-run 参数仍可调用但隐藏帮助。`data list [name]` 接替公开 status；删除 compose 模块、入口、四项专属测试及旧 Web 控制台按钮。
- 参数单一来源：原 Web 参数规则移到 `src/lnl_toolbox/parameters.py`，原 Web YAML 原样移到随包发布的 `src/lnl_toolbox/parameter_metadata.yaml`。Web 保留原函数入口与 HTTP 字段，CLI 不导入 Web 服务。未修改任何算法、损失、正式实验 YAML 或结果选择规则。
- 本轮文件：`src/lnl_toolbox/{__init__.py,core/__init__.py,data/__init__.py,training/__init__.py,parameters.py,parameter_metadata.yaml,cli/__init__.py,cli/main.py,cli/i18n.py,cli/parameter_help.py,cli/paper_editor.py}`；`web/{command_console.py,index.html,README.md}`；`tests/test_cli.py`；`pyproject.toml`；`README.md`；`docs/development-guide.md`；本记录。删除 `composition.py`，旧 Web 参数 YAML 为迁移而非删除规则。
- 性能：同一 Python 环境独立进程，help 修改前约 7.535 秒，修改后三次 0.937/1.016/0.893 秒，中位数 0.937 秒（约下降 87.6%）；list 中位数 1.462 秒，非交互 show 中位数 1.377 秒。前测含 import-time 诊断开销，不能当成严格冷缓存基准。独立进程断言 help/list/摘要没有导入 torch、torchvision、sklearn 或 Web 服务。
- 26 篇逐篇核对：APL、GCE、Binary Risk、Dual-T、Loss Correction、Importance Reweighting、CWD、PCSE、FINE、CDR、DSS、Co-teaching、JoCoR、CNLCU、MentorNet、T-Revision、MC-LDCE、VolMinNet、PDL、UPM、CAL、DLD、CA2C、DivideMix、L2RW、LEND。提取前后 Web schema 完全相同，未发现缺失英文解释。CLI 使用相同 formal 默认配置、权限、枚举、统一 seed 和预算/学习率显示规则；检查取消、保存保护、参数转换与分段自由学习率，不替换为 smoke。
- 自动化实际执行：主 tests 全量 1193 项通过；CLI 108 项通过；Web 142 项通过；Scratch 224 项通过。最终帮助/提示细改后增量 12 项通过；全部 argparse 英文帮助中文残留 0；配置的 Ruff 与 `git diff --check` 通过。扩展全仓 F821 检查发现的既有类型注解问题未纳入本任务，不能据此声称扩展规则也全部通过。构建 wheel 成功并核实共享规则、元数据及三个新 CLI 模块均入包；构建产物未跟踪。
- 真实浏览器：内置浏览器连接超时后使用已安装的 headless Edge，通过页面控件操作真实 HTTP 服务（不是 mock，不是 CLI 单独启动训练）。GCE 从已登记数据集选择、分段开关、非法 loss.q=2 的底部 CLI 错误输出、修正并预演，到点击开始完整执行一轮，19 次更新、退出码 0。输出目录 `artifacts/web-runs/3f378ea84754418eb081f42a4e406c8f`，job `f21d9e6d7ff0433da4077d4b8b7f38ac`。这是运行链路验证，不是准确率验收，也不是 26 篇逐篇真实训练。
- 停止补验：沙箱服务内 Windows taskkill 返回 Access denied，不能代表正常环境。已清理那次测试训练；正常权限本机临时服务再次经 Web 点击开始/停止，job `f3ac10a3745046fa9756607d10f56ff6` 为 `running=false, cancel_requested=true, returncode=1, state=cancelled`，未完成计划的 100 轮。未为测试环境限制改动训练或停止代码。
- 验收边界：原有 8765 服务未重启或替换；真实浏览器仅对 GCE 做代表性短训练，全部 26 篇完成的是参数/规则核对。未安装全局 CLI，命令检查使用本机既有 Python 的模块入口。生成的训练结果仅在忽略的 artifacts 中保留。旧独立训练入口与 Scratch 保留。

## 默认 YAML 命名与非默认配置整理（2026-11-06）

- Current task / goal: 26 篇 Web/CLI 默认配置采用 `论文名_数据集.yaml`；其他实验归入 smoke；清理无引用配置，并同步所有活动引用与测试。不改算法、默认参数或训练预算。
- Branch/base: `scratch` / `8e83feb`；沿用当前分支，不创建分支、不暂存、不提交、不推送。保留任务开始时 `.github/workflows/quality.yml`、`training/dual_t_evidence_experiment.py` 的改动和 `tests/test_cli.py` 中已有删除的验证断言，以及未跟踪远端运维文件。
- Approved scope: `configs/` 中上文批准的 78 次文件移动和两份自定义配置删除；目录登记、打包、CLI 模板发现、路径引用的测试及使用文档。登记检查工具 `scripts/check_paper_registration.py` 与 `tests/test_paper_registration.py` 已额外获得批准。AGENTS.md 仅更新一条使用示例的 YAML 路径，不改任何协作或审批规则。
- Checklist: (1) 引用与删除范围核实；(2) 默认配置 26 份改名；(3) 非默认配置 52 份归档及两份无引用文件删除；(4) 稳定 ID、profile 与打包路径；(5) 原文件哈希、默认入口及拼接路径审计；(6) 定向/Web/Scratch/打包/静态检查；(7) 主测试集与最终范围核对。已完成 7/7（100%）。
- Layout: `configs/reproduction/` 26 份默认配置；`configs/smoke/` 52 份非默认实验；`configs/experiment/` 保留一份 MentorNet 教师准备配置；algorithm/noise 保留 16 份片段。完整默认文件名列表见 `docs/file-map.md`。smoke 目录是本轮非默认配置收纳位置，不意味着其中每份配置都是短预算或合成数据。
- Deleted: `configs/experiment/binary-risk-natarajan-reproduction-custom.yaml`、`configs/experiment/gce-cifar10-noise02-reproduction-custom.yaml`；无代码、脚本、测试或目录登记引用，原受 Git 跟踪，可从历史恢复。历史 `archive/`、论文实现过程记录及已生成训练产物不迁移。
- Runtime changes: manifest 新增 `identities`，显式保留全部 68 个已登记 recipe 的 ID、profile 和保真度。`catalog.py` 读取该身份，不再因文件名变化改变已有命令；登记检查工具使用相同身份字段。旧 CLI 模板发现同时扫描 reproduction/smoke/experiment。论文目录的默认 recipe ID 无需更改。
- Verified invariants: 95 份保留的 YAML 全部 SHA-256 与迁移前一致（逐字节，包括换行）；68 个 recipe 的所有属性除 config_path 外完全一致；26 篇默认入口全部命中新文件；活动完整路径旧引用为 0，AST 检查已知配置的分段 Path 拼接缺失为 0。
- Checks executed: `python -B -m unittest test_config test_paper_registration` 31 项通过；`python -B scripts/check_paper_registration.py` 0 issues；Web 142 项通过；Scratch 224 项通过；主测试集 `unittest.defaultTestLoader.discover('tests', pattern='test_*.py')` 1197 项通过（472.231 秒；同一进程设置 torch intra/inter-op threads 为 2）；`python -B -m ruff check src tests web scripts` 与 `git diff --check` 通过。首轮广泛检查发现分段拼接的旧目录引用，已修正；首轮主测试因此中断，不能计作通过，上述 1197 项为修正后的完整重跑。
- Packaging: `pip wheel . --no-deps --no-build-isolation --no-index --wheel-dir <Temp>/lnl-yaml-relocation-wheel-20261106` 成功；解包检查 68 个登记 recipe 全部包含，默认目录恰好 26 份。wheel 仅用于验证，没有安装或提交。
- Files inspected/changed: `src/lnl_toolbox/catalog.py`、`src/lnl_toolbox/cli/__init__.py`、`src/lnl_toolbox/cli/data/recipe_catalog.json`、`scripts/check_paper_registration.py`、`pyproject.toml`；`tests/test_{ca2c,cal,cdr,cli,cnlcu,config,cwd,dss,estimators,importance_reweighting,jocor,l2rw,lend,loss_correction,mc_ldce,new_paper,paper_registration,pcse,registry,training}.py`；`web/test_command_console.py`、`web/test_training_status.py`、`src/lnl_toolbox/scratch/tests/test_formula_templates.py`；`AGENTS.md`、`README.md`、`docs/{README-usage,clean-training-architecture,clean-training-guide,development-guide,file-map,project-management-guide,t-revision,toolbox-modularization-progress}.md`、`papers/{dividemix,dual-t,importance-reweighting}/reproduction.md`。新路径 YAML 为原文件移动，没有新增算法、依赖或迁移层模块；临时审计脚本和 wheel 在系统 Temp，不是仓库新增源文件。
- Known limitations / integration: 直接使用旧 YAML 路径的外部脚本需改新路径；旧 recipe 命令保留。已运行的 Web 服务可能缓存旧路径，需要操作者重启；本轮不停止现有服务。此前论文参数对照发现的问题不在本次路径整理中修复。共享高冲突文件是 catalog、recipe manifest、pyproject、CLI 共享模板、已有 CLI 测试与进度文档，合并时需保留本轮任务外改动。
- Local checkpoint commits: none。Allowlist comparison: 变更符合上述批准范围，无额外仓库源文件；历史清理和发布均未授权。
- Exact next step: 交付路径整理结果；需要使用新路径时由操作者重启已有 Web 服务。论文参数差异另行处理；无 commit/push 步骤。

## 2026-11-07 PCSE / DLD 跨论文运行依赖删除

- Goal/scope: 非 Scratch 的 PCSE / DLD 不读取其他论文训练结果。PCSE 固定执行自身 CE 预训练；DLD 保留官方 ImageNet ResNet-34 权重入口和独立 frozen-model 流程。训练参数值不变，远端运行快照不动。
- Branch/base: `scratch`，开始时 HEAD `29b9752373827555a276cf75904358001dcfbdaf`。无新文件、无 commit/push。保留已有 CAL、Dual-T 和远端运维文件改动。
- Checklist: (1) 依赖实现删除；(2) 内置配置与元数据清理；(3) 专属测试删除及相关回归；(4) Web/CLI 文案残留清理；(5) 最终审计。完成 5/5（100%）。后续新增的本地正式配置一轮运行验证：PCSE、DLD 均完成，进程退出码均为 0。
- Changed implementation: `algorithms/pcse/config.py` 删除 mode/source schema；`algorithms/pcse/algorithm.py` 删除外部模型接管与来源状态；`training/pcse_experiment.py` 固定自身预训练；删除 `training/pcse_pretrained.py`（317 行外部适配实现，Git 历史可恢复）。`algorithms/dld/config.py`、`training/dld_pretrained.py`、`training/dld_experiment.py` 删除 UPM 加载分支，保留官方权重检查。上述路径均以 `src/lnl_toolbox/` 为根。
- Changed shared files: `training/runners.py`、`training/service.py`、`catalog.py` 删除对应前置运行要求；`core/config_schema.py`、`parameters.py` 删除旧模式规则；`parameter_metadata.yaml`、`web/lnl_parameter_metadata_registry.yaml` 删除旧选项。PCSE 默认及 multiclass smoke 配置删除 mode；volmin smoke 内容未变。README、Web README、DLD 文档及两份使用手册说明自身训练流程。
- Tests: `tests/test_pcse.py`、`test_dld.py` 删除外部来源专属回归；`test_registry.py`、`test_pipeline.py`、`test_cli.py`、`test_quickstart_service.py` 改为独立入口的断言。保留内部训练、恢复及 ImageNet 检查。
- Executed: PCSE/DLD/registry/pipeline 143 项通过（43.962 秒）；CLI/Quick Start 153 项通过（297.603 秒）；清理环境变量断言后相关 2 项重跑通过（3.182 秒）。Web 134 项中首次 133 通过、1 失败；原因是本轮元数据清理误伤参数路径，修正后该项重跑通过（26.863 秒）。未完整重跑 Web 134 项，未跑全站测试，也未启动 Web 真实训练。此前选定文件 Ruff 检查通过；最终审计待做。
- Additional files approved/completed: 用户随后要求“完后本地运行”，完成 `web/assets/quick_start.js` 删除 PCSE mode 判断并将“训练或加载模型”改为自身训练；`src/lnl_toolbox/cli/parameter_help.py` 删除旧模式说明。Web assets 33 项测试通过（60.310 秒），选定文件 Ruff 与 git diff --check 通过。
- Local actual training: `artifacts/local-formal-one-epoch-20261107/dld` 正式 `dld_cifar10.yaml`，仅覆盖 `dld.diffusion.epochs=1`、`trainer.epochs=1`；显式选择完整 CIFAR-10 别名 `local-cifar10`，不使用 mini 数据。完整执行官方 ImageNet ResNet34 特征、45000 样本预校正与一轮扩散训练（225 次更新），进程退出码 0。模型、噪声、学习率、分段、推理步数均未改，生成的路径与噪声事实由正常运行补充。
- PCSE actual training: 用户批准后，正式 `pcse_cifar10.yaml` 仅临时覆盖 pretraining/transition/ensemble epochs 为 1、scheduler.start_epoch/end_epoch 为 0/1，选完整 `local-cifar10`；结果位于 `artifacts/local-formal-one-epoch-20261107/pcse`。自身 CE 预训练 352 次更新、VolMin 矩阵估计 352 次更新、特征统计、GDA 和集成训练 100 次更新全部完成，退出码 0，最终测试样本数 10000。未读取其他论文结果。内置正式 YAML 未改为短测值。
- Integration: 当前修改均在已批准清单内。共享冲突点为 catalog、参数元数据、runner/service 及 CLI/Quick Start 测试。旧的带 mode/source PCSE 配置将明确报不支持，不再加载其他论文结果。官方 ImageNet 权重仍是 DLD 的外部文件需求，不是其他论文产物。全部训练产物本地忽略，不提交。
- Exact next step: 交付本地两篇正式配置缩短预算后的真实运行结果。无 commit/push。
