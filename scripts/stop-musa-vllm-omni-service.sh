#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?set CONTAINER_NAME to the task-owned container name}"
: "${CONTAINER_OWNER:?set CONTAINER_OWNER to the lease owner}"
: "${CONTAINER_TICKET:?set CONTAINER_TICKET to the task or ticket identifier}"
: "${LEASE_HANDLE:?set LEASE_HANDLE to the lease handle recorded at launch}"

DOCKER_BIN="${DOCKER_BIN:-docker}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

if [[ "${CONTAINER_NAME}" != hlease-* ]]; then
  echo "CONTAINER_NAME must start with hlease-" >&2
  exit 2
fi
if [[ ! "${STOP_TIMEOUT}" =~ ^[0-9]+$ ]]; then
  echo "STOP_TIMEOUT must be a non-negative integer" >&2
  exit 2
fi

labels=$("${DOCKER_BIN}" inspect \
  --format '{{index .Config.Labels "owner"}}|{{index .Config.Labels "ticket"}}|{{index .Config.Labels "lease_handle"}}' \
  "${CONTAINER_NAME}")
expected="${CONTAINER_OWNER}|${CONTAINER_TICKET}|${LEASE_HANDLE}"
if [[ "${labels}" != "${expected}" ]]; then
  echo "container ownership labels do not match the requested lease; refusing to stop" >&2
  exit 3
fi

exec "${DOCKER_BIN}" stop --time "${STOP_TIMEOUT}" "${CONTAINER_NAME}"
