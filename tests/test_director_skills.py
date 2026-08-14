from __future__ import annotations

from pathlib import Path

from long_video_studio.director_skills import selected_skill_excerpt


def test_selected_skill_excerpt_uses_only_matching_pack(tmp_path: Path):
    animation = tmp_path / "3d-animation-short-generator"
    animation.mkdir()
    (animation / "SKILL.md").write_text(
        "# Shot table\nContinuity handoff and fixed landmark positions.\nCanvas UI instructions.\n",
        encoding="utf-8",
    )
    music = tmp_path / "music-video-subtitle-generator"
    music.mkdir()
    (music / "SKILL.md").write_text("# Beat grid\nMaster audio timeline.\n", encoding="utf-8")

    excerpt = selected_skill_excerpt(tmp_path, "animation")
    assert "fixed landmark positions" in excerpt
    assert "Master audio timeline" not in excerpt
    assert "Canvas UI instructions" not in excerpt


def test_missing_skill_directory_is_a_safe_noop(tmp_path: Path):
    assert selected_skill_excerpt(tmp_path / "missing", "animation") == ""
