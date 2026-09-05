# AI Video Studio — AI 视频全自动生产线

一体化 AI 视频生产工作台：从「厚版剧本」到「成片」的完整流水线。

> 本仓库是本产品的**独立主仓库**（monorepo，单分支 `main`）：
>
> - `studio/` — 批量生产主线：脚本导入 → 层级规划 → 预制片 → H3 视频渲染，含主操作前端
> - `canvas/` — 无限画布工作台：素材整理 / 单镜重生成 / 与 studio 双向编辑、深链互通
>
> **日常使用**：双击仓库根目录的 `Start-AI-Video-Studio.bat`，一个入口同时拉起全部服务；
> 主要操作在 studio 前端 <http://localhost:7860>，画布 <http://localhost:3000/canvas> 按需辅助。

## 功能一览

- **厚版脚本导入** —— 整份剧本一次性导入为结构化工程（server 端素材库导入器）
- **层级批量规划** —— 创意总监 → 分镜导演 → 连续性审校的多级 LLM 规划；支持 Qwen / DeepSeek 等
  OpenAI 兼容渠道多 profile；无 key 时回落内置确定性规划器
- **预制片生产** —— 文生图（vLLM-Omni Qwen-Image / OpenAI 兼容渠道）、图像编辑（Qwen-Image-Edit）
  产出角色 / 场景 / 首帧素材，统一素材库管理
- **视频渲染** —— MiniMax-H3（FL2VA / Ref2VA 分区）桥接 MUSA GPU vLLM-Omni 服务；
  支持单镜强制重渲（`POST /render?shot_id=<id>`，不重组 final.mp4）
- **音频** —— Suno 音频客户端
- **画布工作台** —— 素材入库、单镜选择与重生成闭环、渲染控制台、双向深链

## 目录结构

```
ai-video-studio/
├── studio/                    # 批量生产主线（Python 3.10+ / FastAPI + React 19）
│   ├── src/long_video_studio/ # 后端核心：应用、规划/渲染/资产适配器、MCP、CLI 入口
│   ├── web/                   # Creator UI（React 19 + Vite 6 + Tailwind）
│   ├── scripts/               # MUSA vLLM-Omni 服务启停、冒烟与校验脚本
│   ├── Start-Studio.bat/.ps1  # 仅启动主工作流 :7860
│   └── Start-DualFrontend.ps1 # 启动主工作流 + 画布工作台
├── canvas/                    # 无限画布工作台（React / Vite / TypeScript）
│   ├── web/                   # 画布前端（dev :3000）
│   └── canvas-agent/          # 画布代理与技能
├── Start-AI-Video-Studio.bat  # ← 一键启动入口（拉起 :7860 + :3000）
├── README.md / AGENTS.md      # 本文件 / AI 编码协作指南
└── LICENSE / THIRD_PARTY.md   # 许可与第三方声明
```

> 按约定不入库：`.env`（密钥）、`docs/`、`tests/`、生成的媒体与运行数据（见各 `.gitignore`），
> 这些内容仅保留在本地开发机。

## 快速开始

环境要求：Python ≥ 3.10（推荐 3.11）、Node 22 + npm、ffmpeg。

```bash
# 1) 安装 studio 后端
cd studio
python -m venv .venv
pip install -e ".[dev]"

# 2) 配置环境变量（密钥只放本地 .env，永不入库）
copy .env.example .env        # macOS/Linux: cp .env.example .env

# 3) 启动
#    双击仓库根目录 Start-AI-Video-Studio.bat
#      → 主工作流  http://127.0.0.1:7860   ← 主要操作界面
#      → 画布工作台 http://localhost:3000/canvas
#    只需主线时也可运行 studio\Start-Studio.bat
```

常用命令（在 `studio/` 下执行，Makefile）：`make lint` / `make test` / `make web-build` / `make check`。
Docker：`studio/docker-compose.yml`、`canvas/docker-compose*.yml`。

## 关键配置（键名速查）

所有密钥只存本地 `.env`，完整说明见 `studio/.env.example`：

| 键前缀 | 用途 |
|---|---|
| `STUDIO_PLANNER_*` | 规划 agent（OpenAI 兼容；多 profile：`STUDIO_PLANNER_PROFILE_IDS=qwen,deepseek,...`） |
| `STUDIO_T2I_*` | 文生图端点（vLLM-Omni 或 OpenAI 兼容渠道） |
| H3 端点配置 | MiniMax-H3 渲染；FL2VA 与 Ref2VA 为独立分区，可分节点部署 |
| `STUDIO_GPU_SNAPSHOT_PATH` | GPU 遥测快照（外部采集器原子更新，studio 只读） |
| `VITE_*`（canvas） | 画布前端所需的渠道配置通过 Vite 环境变量注入，不硬编码 |

## 开发约定

- 提交信息：`feat:|fix:|docs:|chore: + 中文描述`
- Python：ruff（line-length 120）；兼容 3.10–3.12；测试 `cd studio && make test`
- AI 编码代理请读根目录 `AGENTS.md`

## 安全说明

- 服务器默认无内置鉴权，仅限本机使用或置于鉴权反代之后
- `.env`、生成媒体、内网 registry / 内网 IP / 机器本地模型路径一律不入库

## 许可与声明

本项目衍生自 Apache-2.0 协议的开源项目。按协议要求，相应许可文本与第三方声明
保留于 `LICENSE`、`THIRD_PARTY.md` 与 `canvas/LICENSE`。
