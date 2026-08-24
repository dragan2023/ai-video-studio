from __future__ import annotations

from pathlib import Path

from long_video_studio.config import Settings


def test_default_text_to_image_timeout_covers_long_musa_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("STUDIO_T2I_TIMEOUT_SECONDS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.text_to_image_timeout_seconds == 7200


def test_named_planner_profiles_keep_keys_out_of_public_view(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STUDIO_PLANNER_PROFILE_IDS", "qwen")
    monkeypatch.setenv("STUDIO_PLANNER_QWEN_BASE_URL", "https://dashscope.example/v1")
    monkeypatch.setenv("STUDIO_PLANNER_QWEN_API_KEY", "private-qwen-key")
    monkeypatch.setenv("STUDIO_PLANNER_QWEN_MODEL", "qwen-plus")

    profile = Settings.from_env(project_root=tmp_path).planner_profile("qwen")

    assert profile.public() == {
        "id": "qwen", "display_name": "qwen", "model": "qwen-plus",
        "wire_api": "chat_completions", "available": True,
    }
    assert "private-qwen-key" not in str(profile.public())
