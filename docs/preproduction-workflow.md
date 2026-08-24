# 预制片计划工作链路

> 面向“预制片计划”的全链路梳理，以及 LLM 调用/模型切换的现状与本次改造。

## 1. 目标

把“厚版分镜脚本 + 已有素材”变成一份**可见、可编辑、一次确认后按镜头顺序制作**的预制片计划。确认前不发起任何生图，确认后只补“阻断制作的独立首帧”，再进入顺序渲染。

## 2. 端到端链路

### 2.1 导入厚版脚本

- 入口：POST /api/projects/import-thick-script
- 实现：script_importer.parse_shot_script → project_builder.build_film_project → 写出 <project>-plan.md
- 产出：FilmProject（含原始 source_section、R/S/P 素材映射、缺失码清单）
- 此阶段**不调用 LLM**。

### 2.2 用户选择 / 管理 LLM 客户端（本次新增独立页）

- 新页面：模型管理（侧栏 Model 管理）
- 数据源：环境变量 profiles（STUDIO_PLANNER_PROFILE_IDS）+ 数据库 llm_clients 表
- 默认按钮：POST /api/llm-clients/{id}/default → 写入 settings.active_planner，全局唯一生效
- 所有需要 LLM 的地方统一经 StudioServices.resolve_planner() 读取该默认客户端

### 2.3 H3 分镜分批补全

- 入口：POST /api/projects/{id}/h3-enrichment
- 实现：PlanningManager.start_imported_h3 创建 BatchPlanningRun，后台 enrich_imported_shots 每批 6 镜填充 H3 时间线字段
- 使用 planner_factory=services.resolve_planner，即**默认 LLM 客户端**
- 产出：project.shots 补齐 opening_state / ending_state / visual_beats / dialogue 等

### 2.4 生成预制片计划

- 入口：POST /api/projects/{id}/preproduction
- 实现：若 H3 批处理未完成则拒绝；否则 PreproductionPlanner.plan(project, assets) 逐镜给出转场判定、首帧来源、缺口与置信度
- 首帧优先级：手动首帧 > 系统黑场 > 连续镜头上一末帧 > 用户参考图 > GPT Image 2 补图 > 待人工复核
- 状态：awaiting_approval（无阻断）或 blocked
- 此阶段只做判定，**不调用生图/视频 provider**

### 2.5 用户确认与受控补图

- 入口：POST /api/projects/{id}/preproduction/approve
- 若需补图：PreproductionAssetGenerator.generate_approved_gaps 仅处理 generate_t2i 项；成功导入为普通资产并绑定镜头；失败 → blocked
- 无需补图：直接 become_ready
- 完成后状态 ready，渲染入口放行

### 2.6 顺序渲染

- 入口：POST /api/projects/{id}/render
- 前置：存在预制片计划时必须是 ready；旧项目走手动路径
- RenderManager 按 index 顺序、并发 1 提交 H3 任务

## 3. LLM 调用与模型路由

### 3.1 调用点

| 场景 | 入口 | 解析方式 |
| --- | --- | --- |
| 故事板构思 | /api/projects/plan、/api/projects/plan-async | resolve_planner() → 默认客户端 |
| H3 分批补全 | /api/projects/{id}/h3-enrichment | planning_manager 的 planner_factory = resolve_planner |
| 预制片（无批处理时直通） | /api/projects/{id}/preproduction | resolve_planner(profile, model) |

所有请求最终落在 PlannerService._request_json_wire，其 model 来自当前 PlannerService 绑定的 settings.planner_model。

### 3.2 之前的痛点

- 客户端只能通过环境变量配置；/api/planner-profiles 也不含数据库客户端。
- 模型切换下拉条只在“故事板”出现，未导入脚本前不可见。
- H3 批处理在 PlanningManager 内直接构造 PlannerService，只认环境 profiles，不能使用 UI 新增的客户端。
- 切换后，默认只影响部分调用点（同步 plan / preproduction），而批处理仍可能用旧 profile。

### 3.3 本次改造

- 新增 llm_clients 持久化表 + LLMClient 域模型。
- StudioServices.set_active_planner / resolve_planner / active_planner_view 统一支持环境与库内客户端。
- PlanningManager 通过 planner_factory 解析，因此 H3 批处理也遵循全局默认。
- 新增前端“模型管理”独立页，卡片提供“设为默认”“编辑”“删除”。
- “设为默认”后，**故事板构思、H3 分镜补全、预制片规划**全部使用该客户端。

## 4. 关键文件

- src/long_video_studio/preproduction.py — 计划领域唯一所有者
- src/long_video_studio/preproduction_assets.py — 受控补图
- src/long_video_studio/planning.py — 分批补全调度
- src/long_video_studio/services.py — 默认客户端解析
- src/long_video_studio/api.py — 计划 / 客户端 API
- web/src/main.jsx — 前端工作台与“模型管理”页
