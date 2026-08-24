# 预制片规划与受控自动补图实施计划

## Goal
实现前端显式的预制片规划阶段：用户导入厚版分镜脚本和大部分素材后，LLM 自动填写现有 Studio 的项目级、镜头级参数；系统显示跨镜判定、首帧来源、素材缺口与预计补图数。用户确认一次后，才用 GPT Image 2 补足阻断制作的独立首帧，随后按 1536×864、并发 1、镜头顺序制作长视频。

## Architecture
- 新增唯一领域所有者 PreproductionPlanner。它读取 FilmProject、脚本原文、素材和配置，输出可持久化、可审阅的预制片计划；不得渲染视频或静默调用生图。
- FilmProject 保存可选 preproduction_plan、版本、确认状态与批准生成资产的引用；旧项目无该字段时保持可读。
- API 负责生成、编辑、确认、受控补图和渲染门禁；Runner 只消费已批准计划并保持按镜头顺序执行。
- React 在故事板与制作之间新增显式页面；未确认前不能显示为可制作。

## Tech Stack
Python 3.11、FastAPI、Pydantic、SQLite JSON、React/Vite、现有 Responses-compatible Planner、现有 OpenAI-compatible Text-to-Image adapter、ComfyUI H3、pytest。

## Baseline / Authority Refs
- docs/architecture.md：控制面、提供方、首帧及顺序渲染边界。
- docs/image-edit-providers.md：有序素材与显式首帧优先级。
- README.md：continuation 模式、环境变量、串行渲染。
- 用户确认：预制片必须前端呈现；剪辑与动作和剪辑参考是跨镜判定权威；用户素材优先；仅补阻断制作的独立首帧；匹配剪辑、声音桥、亮度匹配和遮罩为视觉硬切；连续镜头用上一成片末帧；黑场不云端生图；确认一次才可付费补图。

## Compatibility Boundary
- 保持现有项目规划、镜头编辑、手动首帧、FL2VA/Ref2VA 和旧项目渲染可用。
- API 密钥仅从 .env 或部署密钥读取，绝不写入项目、SQLite、计划响应、日志或前端状态。
- 1536×864、并发 1、steps、超时、重试是固定运行档位，LLM 不得改写。
- 用户手动 start_frame_asset_id 永远优先。

## TDD Route
- Mode: off
- Decision: light
- Strict authority: not applicable
- Test posture: post-change regression with mock LLM/T2I transports
- Verification: focused pytest、Ruff、web format/build、一次 mock API workflow。

## Requirement Ready Check
- Goals: 将详细脚本和素材变成可见、可编辑、一次确认后可顺序制作的预制片计划。
- Acceptance: 参数全部可见；无确认不生图也不渲染；只补独立首帧；连续/硬切由脚本证据决定；密钥不泄露。
- Open question: 用户在实现后仅在本地 .env 填写 GPT Image 2 base URL、model 与 key，聊天中不传密钥。
- Decision: ready。

## Change Necessity
现有 Planner 可以创建故事板，T2I 可以生成零素材首帧，但没有可审阅计划、脚本证据、确认门槛或缺图闭环；配置无法实现这一交互。最小代码边界是复用现有 Planner、T2I、FilmProject，在新的规划服务和已有 API/UI 建立计划生命周期。Decision: code-change。

## Ownership and Retirement
PreproductionPlanner 是跨镜判定与首帧策略唯一所有者；API 和 Runner 只验证/消费。保留旧手动流程，不能用新计划字段静默破坏旧项目。不存在要删除的持久化数据或旧提供方。

## Task 1 — 可持久化领域契约
Files: modify src/long_video_studio/domain.py; create tests/test_preproduction_domain.py.

1. 新增 StartFrameSource：creator_asset、previous_boundary、system_black、generate_t2i、needs_review。
2. 新增 PreproductionStatus：draft、awaiting_approval、approved、generating_assets、ready、blocked。
3. 新增每镜计划模型，记录脚本原文证据、TransitionKind、首帧来源、上游镜头、候选/选中资产、缺口原因、置信度、补图许可、参数摘要。
4. 新增项目计划模型，记录版本、输入资产哈希、生成数、固定运行档位、阻断项、警告和确认时间；FilmProject 添加默认 None 的字段。
5. 测试历史项目 JSON 可读、手动首帧覆盖、首镜不能用上一末帧、未确认计划不能 ready。
6. Verify: PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q tests/test_preproduction_domain.py

## Task 2 — 证据驱动规划服务
Files: create src/long_video_studio/preproduction.py; modify src/long_video_studio/script_importer.py and its importer-to-project mapping so source_section retains each完整厚版镜头段； create tests/test_preproduction.py.

1. 实现 PreproductionPlanner.plan(project, assets)，只生成计划，绝不调用视频或图像 provider。
2. 解析 source_section 中的剪辑与动作和剪辑参考；镜内的单镜头无切不能当作跨镜证据。
3. 判定优先级：显式硬切/换场/闪回/黑场优先；匹配剪辑、声音桥、亮度匹配、遮罩均为视觉硬切；明确承接同一动作、空间、视线或轴线为连续；其余 needs_review。
4. 首帧优先级：手动首帧、系统黑场、连续镜头上一成片末帧、本镜用户参考图、独立镜头 GPT Image 2 补图、needs_review。
5. 计划保存原文证据与置信度，生成补图 prompt 但不发请求。
6. 用极乐城样例测试：1-01 黑场、1-08 到 1-09 匹配剪辑、硬切、明确承接、证据不足。
7. Verify: PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q tests/test_preproduction.py tests/test_script_importer.py

## Task 3 — 确认、补图与制作门禁 API
Files: modify src/long_video_studio/api.py, services.py, runner.py; create tests/test_preproduction_api.py.

1. 添加生成/读取/修改预制片计划 API。生成后状态 awaiting_approval。
2. 添加 approve API；只在用户确认后启动后台补图任务，且只处理 generate_t2i 项。成功图片作为普通资产导入并绑定镜头；失败项转 blocked，不能静默跳过。
3. 补图结束后计划进入 ready；存在计划时渲染入口只允许 ready。旧项目保持现有手动路径，但不能伪装为已确认的一键项目。
4. Runner 保持 index 顺序和并发 1；Runner 内禁止新增隐式 T2I。
5. 测试无确认无 T2I、确认后仅缺图生成、连续/硬切不生图、失败阻断、串行提交、响应不泄露密钥。
6. Verify: PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q tests/test_preproduction_api.py tests/test_runner_retry.py tests/test_api.py

## Task 4 — GPT Image 2 的通用提供方配置
Files: modify .env.example, docs/image-edit-providers.md, tests/test_text_to_image.py; local .env only after user local setup.

1. 复用 text_to_image.py 的 openai-compatible adapter，不创建供应商专用分支。
2. 在 .env.example 添加无密钥模板：STUDIO_T2I_PROVIDER=openai-compatible、STUDIO_T2I_BASE_URL、STUDIO_T2I_MODEL、STUDIO_T2I_API_KEY；说明仅批准后独立首帧使用。
3. 测试 model、Authorization header、URL/base64 响应和错误脱敏。
4. Verify: PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q tests/test_text_to_image.py

## Task 5 — 前端显式预制片工作台
Files: modify web/src/main.jsx and web/src/styles.css; add/update existing frontend test harness where available.

1. 在故事板和开始制作之间增加“预制片规划”阶段与生成按钮；不后台自动跳转。
2. 表格逐镜展示脚本证据、转场判定、首帧来源、完整参数摘要、素材绑定、缺口、补图原因、置信度。
3. 支持逐镜编辑/锁定；展示生成数、费用估算。无可靠价格时显示未配置价格，仅显示数量。
4. 确认补图必须有明确确认提示；锁定并开始制作仅在 ready 时可用；旧项目保留手动入口。
5. 补图后刷新资产、镜头缩略图和渲染进度。
6. Verify: cd web && npm run format && npm run build；手动验证未批准禁用、批准后才补图/制作。

## Task 6 — 文档、回归和真实冒烟
Files: modify README.md and docs/architecture.md; create docs/preproduction-workflow.md.

1. 文档化状态机、脚本判定、首帧优先级、确认和密钥边界。
2. 更新架构图：脚本+素材 → Planner → PreproductionPlanner → 可见计划/确认 → 可选 T2I → 顺序 Runner。
3. 用极乐城脚本做只规划冒烟，断言 1-01 系统黑场、1-08 到 1-09 视觉硬切、连续镜头不进补图清单。
4. 用户在本地配置 GPT Image 2 后，只对临时独立镜头做一次生图冒烟；不跑全片且不记录密钥。
5. Verify: make check；无法使用 bash 时运行等价 Ruff、pytest、web build。

## Risks and Rollback
- 剪辑误判：以原文证据、置信度、逐镜编辑修正；低置信度不自动补图。
- 云端费用：无确认不请求；计划展示数量；未知价格不伪造金额。
- Provider 兼容：复用 OpenAI-compatible 合约，先做单镜冒烟。
- 旧项目：计划字段可选，旧项目不自动迁移。
- 回滚：隐藏预制片入口并停止消费计划字段即可回到手动流程；已生成图片保留为普通资产，不自动删除。

## Execution Readiness View
- Intent Lock: 可见、可编辑、一次确认后的预制片闭环，不是静默后端自动化。
- Scope Fence: 不替换 H3/ComfyUI，不让 LLM 改机器运行档位，不做云端全量补图。
- Baseline Lock: architecture.md、image provider docs、现有手动首帧与顺序 runner。
- Batches: 领域契约/规划服务 → API/Provider → 前端 → 文档和冒烟。
- Drift rule: GPT Image 2 不兼容 OpenAI images API，或需求变成无确认自动付费时，停止并回到设计。

## Execution Route
- Decision: subagent-driven after implementation approval;领域、API、前端可按文件边界拆分，领域契约先落地。
- Fallback: inline sequential execution。
- User confirmation required: yes — 仅在用户本地写入 GPT Image 2 配置和发起实际付费生图前。
