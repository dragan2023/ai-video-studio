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
