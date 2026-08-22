#!/usr/bin/env bash
# AI 视频全自动生产线 - Nautilus Studio 开发启动脚本
# 用法: bash dev.sh [port]
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src"
export PYTHONUTF8=1
set -a
source .env
set +a
PORT="7860"
if [ -n "$1" ]; then PORT="$1"; fi
exec .venv/Scripts/python -m uvicorn long_video_studio.app:create_app   --factory --host 127.0.0.1 --port "$PORT"
