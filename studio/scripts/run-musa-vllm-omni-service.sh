#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_OMNI_MUSA_IMAGE:?set VLLM_OMNI_MUSA_IMAGE to the operator-approved image reference}"
: "${LEASED_GPU_IDS:?set LEASED_GPU_IDS to the leased physical GPU indices}"
: "${CONTAINER_NAME:?set CONTAINER_NAME to an hlease-prefixed task-owned name}"
: "${CONTAINER_OWNER:?set CONTAINER_OWNER to the lease owner}"
: "${CONTAINER_TICKET:?set CONTAINER_TICKET to the task or ticket identifier}"
: "${LEASE_HANDLE:?set LEASE_HANDLE to the active lease handle}"

if (($# == 0)); then
  echo "pass the vLLM-Omni service command after the script name" >&2
  exit 2
fi

DOCKER_BIN="${DOCKER_BIN:-docker}"
MODEL_ROOT_HOST="${MODEL_ROOT_HOST:-/mnt/nfs/models}"
MODEL_ROOT_CONTAINER="${MODEL_ROOT_CONTAINER:-/home/dist/models}"
SHM_SIZE="${SHM_SIZE:-1g}"
DETACH="${DETACH:-1}"

if [[ "${MODEL_ROOT_HOST}" != /* || "${MODEL_ROOT_CONTAINER}" != /* ]]; then
  echo "MODEL_ROOT_HOST and MODEL_ROOT_CONTAINER must be absolute paths" >&2
  exit 2
fi
if [[ ! "${VLLM_OMNI_MUSA_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "VLLM_OMNI_MUSA_IMAGE must be digest-qualified with @sha256:<64 hex characters>" >&2
  exit 2
fi
if [[ ! "${LEASED_GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "LEASED_GPU_IDS must be a comma-separated list of physical GPU indices" >&2
  exit 2
fi
if [[ "${CONTAINER_NAME}" != hlease-* ]]; then
  echo "CONTAINER_NAME must start with hlease-" >&2
  exit 2
fi
if [[ "${DETACH}" != "0" && "${DETACH}" != "1" ]]; then
  echo "DETACH must be 0 or 1" >&2
  exit 2
fi

docker_args=(
  run
  --rm
  --name "${CONTAINER_NAME}"
  --runtime=mthreads
  --network=host
  --ipc=host
  --shm-size "${SHM_SIZE}"
  --label "owner=${CONTAINER_OWNER}"
  --label "ticket=${CONTAINER_TICKET}"
  --label "lease_handle=${LEASE_HANDLE}"
  -e "MTHREADS_VISIBLE_DEVICES=${LEASED_GPU_IDS}"
  -e PYTHONUNBUFFERED=1
  -v "${MODEL_ROOT_HOST}:${MODEL_ROOT_CONTAINER}:ro"
)
if [[ "${DETACH}" == "1" ]]; then
  docker_args+=(--detach)
fi

exec "${DOCKER_BIN}" "${docker_args[@]}" "${VLLM_OMNI_MUSA_IMAGE}" "$@"
