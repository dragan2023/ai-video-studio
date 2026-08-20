from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fake_command(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "args.txt"
    executable = tmp_path / "fake-command"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n', encoding="utf-8")
    executable.chmod(0o755)
    return executable, capture


def run_script(
    script: str,
    env: dict[str, str],
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    complete_env = os.environ.copy()
    complete_env.update(env)
    return subprocess.run(
        [str(ROOT / "scripts" / script), *args],
        check=check,
        capture_output=True,
        text=True,
        env=complete_env,
    )


def test_musa_container_wrapper_maps_only_explicit_devices(tmp_path: Path) -> None:
    docker, capture = fake_command(tmp_path)
    model_root = tmp_path / "models"
    model_root.mkdir()

    run_script(
        "run-musa-vllm-omni-service.sh",
        {
            "CAPTURE_PATH": str(capture),
            "CONTAINER_NAME": "hlease-nautilus-test-service",
            "CONTAINER_OWNER": "codex-test",
            "CONTAINER_TICKET": "NAUTILUS-TEST",
            "DETACH": "0",
            "DOCKER_BIN": str(docker),
            "LEASED_GPU_IDS": "2,3",
            "LEASE_HANDLE": "gpu-test-handle",
            "MODEL_ROOT_HOST": str(model_root),
            "VLLM_OMNI_MUSA_IMAGE": f"registry.example/vllm-omni:test@sha256:{'a' * 64}",
        },
        "vllm",
        "serve",
        "/home/dist/models/example",
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "MTHREADS_VISIBLE_DEVICES=2,3" in arguments
    assert "--privileged" not in arguments
    assert all(not argument.startswith("MUSA_VISIBLE_DEVICES=") for argument in arguments)
    assert all(not argument.startswith("CUDA_VISIBLE_DEVICES=") for argument in arguments)
    assert "owner=codex-test" in arguments
    assert "ticket=NAUTILUS-TEST" in arguments
    assert "lease_handle=gpu-test-handle" in arguments
    assert f"{model_root}:/home/dist/models:ro" in arguments
    assert arguments[-4:] == [
        f"registry.example/vllm-omni:test@sha256:{'a' * 64}",
        "vllm",
        "serve",
        "/home/dist/models/example",
    ]


def test_musa_container_wrapper_rejects_mutable_image_tag(tmp_path: Path) -> None:
    docker, capture = fake_command(tmp_path)
    result = run_script(
        "run-musa-vllm-omni-service.sh",
        {
            "CAPTURE_PATH": str(capture),
            "CONTAINER_NAME": "hlease-nautilus-test-service",
            "CONTAINER_OWNER": "codex-test",
            "CONTAINER_TICKET": "NAUTILUS-TEST",
            "DOCKER_BIN": str(docker),
            "LEASED_GPU_IDS": "2,3",
            "LEASE_HANDLE": "gpu-test-handle",
            "MODEL_ROOT_HOST": str(tmp_path),
            "VLLM_OMNI_MUSA_IMAGE": "registry.example/vllm-omni:mutable",
        },
        "vllm",
        "serve",
        "/home/dist/models/example",
        check=False,
    )

    assert result.returncode == 2
    assert "must be digest-qualified" in result.stderr
    assert not capture.exists()


def test_stop_wrapper_verifies_lease_ownership_before_stopping(tmp_path: Path) -> None:
    capture = tmp_path / "stop-args.txt"
    docker = tmp_path / "fake-docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = inspect ]; then
  printf '%s\\n' 'codex-test|NAUTILUS-TEST|gpu-test-handle'
  exit 0
fi
printf '%s\\n' "$@" > "$CAPTURE_PATH"
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    run_script(
        "stop-musa-vllm-omni-service.sh",
        {
            "CAPTURE_PATH": str(capture),
            "CONTAINER_NAME": "hlease-nautilus-test-service",
            "CONTAINER_OWNER": "codex-test",
            "CONTAINER_TICKET": "NAUTILUS-TEST",
            "DOCKER_BIN": str(docker),
            "LEASE_HANDLE": "gpu-test-handle",
        },
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "stop",
        "--time",
        "30",
        "hlease-nautilus-test-service",
    ]


def test_stop_wrapper_refuses_mismatched_lease_labels(tmp_path: Path) -> None:
    capture = tmp_path / "stop-args.txt"
    docker = tmp_path / "fake-docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = inspect ]; then
  printf '%s\\n' 'another-owner|NAUTILUS-TEST|gpu-test-handle'
  exit 0
fi
printf '%s\\n' "$@" > "$CAPTURE_PATH"
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    result = run_script(
        "stop-musa-vllm-omni-service.sh",
        {
            "CAPTURE_PATH": str(capture),
            "CONTAINER_NAME": "hlease-nautilus-test-service",
            "CONTAINER_OWNER": "codex-test",
            "CONTAINER_TICKET": "NAUTILUS-TEST",
            "DOCKER_BIN": str(docker),
            "LEASE_HANDLE": "gpu-test-handle",
        },
        check=False,
    )

    assert result.returncode == 3
    assert "ownership labels do not match" in result.stderr
    assert not capture.exists()


def test_qwen_image_launcher_sets_stable_served_name(tmp_path: Path) -> None:
    vllm, capture = fake_command(tmp_path)

    run_script(
        "serve-qwen-image.sh",
        {
            "CAPTURE_PATH": str(capture),
            "MODEL_PATH": "/models/Qwen-Image-2512",
            "SERVED_MODEL_NAME": "Qwen/Qwen-Image-2512",
            "TP_SIZE": "1",
            "VLLM_BIN": str(vllm),
        },
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["serve", "/models/Qwen-Image-2512", "--omni"]
    assert arguments[arguments.index("--served-model-name") + 1] == "Qwen/Qwen-Image-2512"
    assert arguments[arguments.index("--port") + 1] == "8094"


def test_qwen_image_edit_launcher_aligns_model_name_and_reference_limit(tmp_path: Path) -> None:
    vllm, capture = fake_command(tmp_path)

    run_script(
        "serve-qwen-image-edit.sh",
        {
            "CAPTURE_PATH": str(capture),
            "MAX_REFERENCE_IMAGES": "4",
            "MODEL_PATH": "/models/Qwen-Image-Edit-2511",
            "SERVED_MODEL_NAME": "Qwen/Qwen-Image-Edit-2511",
            "TP_SIZE": "1",
            "VLLM_BIN": str(vllm),
        },
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--served-model-name") + 1] == "Qwen/Qwen-Image-Edit-2511"
    assert arguments[arguments.index("--limit-mm-per-prompt") + 1] == '{"image":4}'
    assert arguments[arguments.index("--port") + 1] == "8093"


def test_qwen_image_edit_2509_wrapper_uses_2509_served_name(tmp_path: Path) -> None:
    vllm, capture = fake_command(tmp_path)

    run_script(
        "serve-qwen-image-edit-2509.sh",
        {
            "CAPTURE_PATH": str(capture),
            "MODEL_PATH": "/models/Qwen-Image-Edit-2509",
            "TP_SIZE": "1",
            "VLLM_BIN": str(vllm),
        },
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--served-model-name") + 1] == "Qwen/Qwen-Image-Edit-2509"
