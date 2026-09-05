# AGENTS.md — AI 编码协作指南

> 写给在本仓库工作的 AI 编码代理（DSH、Claude Code、Codex 等）。
> 人类协作者读 `README.md`；文件与实际代码冲突时，以代码为准。

## 仓库性质（先记住这些）

- **`dragan2023/ai-video-studio`（私有）是本产品的独立主仓库**，monorepo、单分支 `main`。
- 历史已于 2026-09 重置为全新历史（单一 init 提交），**不含任何上游 fork 记录**；
  旧双仓历史仅备份在本地 `<工作区>/_history-backup/`（nautilus-studio-git / infinite-canvas-git）。
- remote 只有 **`origin` = 本仓库**。**禁止**添加上游 remote、从上游 pull、向任何上游发 PR。
- 不要常态化 amend / force push；`main` 是唯一分支。

## 本地布局

```
AI视频全自动生产/                 # 工作区根（中文路径，shell 命令记得加引号；本身不是 git 仓库）
├── ai-video-studio/             # ★ 本仓库（studio/ + canvas/）
│   ├── studio/                  # 批量生产主线（FastAPI :7860 + Creator UI）
│   ├── canvas/                  # 无限画布工作台（Vite dev :3000）
│   ├── Start-AI-Video-Studio.bat# 一键启动入口
│   └── README.md / AGENTS.md
├── _history-backup/             # 旧仓库 .git 历史备份（勿提交、可随时删除）
└── uploads/ output/ data/       # 工作区级临时/产物目录，与 git 无关
```

> 注意：`studio/` 与 `canvas/` 过去是两个独立仓库（本地分支名都是 `main`），
> 2026-09 起合并为本 monorepo，**不要再按旧的双分支结构推送**。

## 常用命令

studio（在 `studio/` 下执行）：

```bash
pip install -e ".[dev]"     # 安装（Python ≥3.10，推荐 3.11；本目录 .venv 已就绪）
make lint                   # ruff format --check + ruff check（line-length 120）
make test                   # pytest -q（tests/ 仅本地存在，未入库）
make web-build              # cd web && npm ci && npm run build
make check                  # lint + test + web-build + web-audit
./Start-Studio.bat          # 仅启动 :7860（CLI 入口：nautilus-studio = long_video_studio.cli:main）
```

canvas（在 `canvas/web` 下执行）：

```bash
npm run dev                 # :3000
npm run build
```

一键启动（仓库根）：`Start-AI-Video-Studio.bat` → studio :7860 + canvas :3000，主操作在 studio 前端。

## 代码地图

- `studio/src/long_video_studio/` —— FastAPI 应用与领域逻辑；`adapters/` 为外部服务适配层
  （规划 LLM、T2I、图像编辑、MiniMax-H3 FL2VA/Ref2VA、MCP 等）
- `studio/web/` —— Creator UI（React 19 + Vite 6 + Tailwind 3）
- `studio/scripts/` —— MUSA GPU vLLM-Omni 服务启停、冒烟与校验脚本
- `studio/tests/`、`studio/docs/` —— pytest 测试与开发文档，**本地保留、未入库**
- `canvas/web/` —— 画布前端；`canvas/canvas-agent/`、`canvas/plugins/`、`canvas/.agents/` —— 画布代理与技能
- `studio/data/` —— 运行时生成（ignored）；`studio/import.body` —— 一次性调试转储（ignored，勿动）

## 环境变量契约

- 唯一来源 `studio/.env.example`（已入库）→ 复制为 `studio/.env`（**永不入库/上传/打印**）。
- `STUDIO_PLANNER_*`：可选规划 agent（OpenAI 兼容、多 profile，见 .env.example 注释）；
  无 key 回落内置确定性规划器；Web UI 只收到 profile 名称/模型/可用性，收不到 key。
- `STUDIO_T2I_*`：文生图端点（vLLM-Omni 或 OpenAI 兼容渠道）。
- H3：FL2VA 与 Ref2VA 是**独立模型分区**，可分节点/端口部署；
  改模型/硬件/精度/并行布局时必须更换 ETA 校准 profile 名。
- `STUDIO_GPU_SNAPSHOT_PATH`：GPU 遥测 JSON 快照；外部采集器原子更新，studio 只读。
- `canvas` 前端所需配置一律通过 `VITE_` 环境变量注入，不得硬编码密钥。

## 约定与风格

- 提交信息：`feat:|fix:|docs:|chore: + 中文描述`。
- Python：ruff（line-length 120；select E,F,I,UP,B,SIM；ignore B008）；兼容 3.10–3.12。
- web：prettier（`npm run format`），React 19。
- Windows：启动脚本 PowerShell/bat；路径含中文与空格，PS 里优先 `-LiteralPath`；
  新建 bat 只用 ASCII 内容避免编码问题。文件编码 UTF-8。
- `.venv/` 不可跨目录复制使用——目录迁移后必须重跑 `pip install -e .`。

## 领域速查（改代码前对号入座）

- **厚版脚本导入**：整份剧本 → 结构化工程；导入器在 server 端（素材库 character / location /
  prop / style / start-frame / audio 角色）。
- **层级规划**：创意总监 → 分镜导演 → 连续性审校；可选本地 H3/style skill pack，
  只送选中 pack 的连续性/音频摘录。
- **预制片（preproduction）**：规划产出的分镜/素材阶段；补图走 T2I / 图像编辑。
- **单镜重渲**：`POST /render?shot_id=<id>` 只重渲染指定镜头、不组装 final.mp4；
  `RenderJob.shot_ids` 支持子集；强制重渲只清目标镜。
- **studio ↔ canvas 闭环**：canvas 画布素材「入 studio」素材库；canvas 单镜重生成 ↔ studio
  单镜重渲端点；双向深链跳转。

## 禁区与注意事项

- 不提交/上传：`.env*`（`.env.example` 除外）、媒体文件、`data/`、`docs/`、`tests/`、
  `import.body`、`.dsh-project-memory/`、日志与缓存、`_history-backup/`。
- **不恢复** `.github/workflows`（CI/dependabot 已随历史重置移除；tests 未入库，CI 必挂）。
- 不把内网 registry、内网 IP（10.x/192.168.x）、机器本地模型路径写进代码。
- `.venv/`、`node_modules/`、`web/dist/` 是本地产物，不要手工修改或提交。
- 服务器无内置鉴权：不要把 7860/3000 暴露到公网。
