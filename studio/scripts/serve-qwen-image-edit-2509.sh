#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Qwen-Image-Edit-2509 checkpoint directory}"
export MODEL_PATH
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen-Image-Edit-2509}"
export SERVED_MODEL_NAME
exec "$(dirname "$0")/serve-qwen-image-edit.sh"
