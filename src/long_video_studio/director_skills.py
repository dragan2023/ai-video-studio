"""Small, safe bridge from local MiniMax-H3 skill packs to planner prompts.

The downloaded skills contain complete production workflows (canvas gates,
asset generation, and model fallbacks) that should not be pasted into every
LLM request.  This module selects one style pack and extracts only the
continuity, shot-table, prompt, camera, audio, and QC paragraphs relevant to a
shot planner.  Missing packs are normal in hosted deployments and fall back to
the built-in H3 contract.
"""

from __future__ import annotations

from pathlib import Path

STYLE_PACKS: dict[str, tuple[str, ...]] = {
    "animation": ("3d-animation-short-generator",),
    "music_video": ("music-video-subtitle-generator",),
    "commercial": ("brand-promo-video-generator", "minimalist-product-ad-generator"),
    "energetic": ("music-video-subtitle-generator",),
    "retro": ("handdrawn-live-video-generator",),
    "handdrawn": ("handdrawn-live-video-generator",),
    "paper": ("papercraft-stop-motion-explainer",),
    "papercraft": ("papercraft-stop-motion-explainer",),
    "collage": ("paper-collage-explainer-generator",),
    "custom": (),
    "cinematic": (),
    "documentary": (),
    "noir": (),
    "surreal": (),
}

REFERENCE_FILES: dict[str, tuple[str, ...]] = {
    "3d-animation-short-generator": (
        "references/shot-table-spec.md",
        "references/qc-checklist.md",
        "references/storyboard-guidelines.md",
    ),
    "co-op-game-intro-generator": ("references/h3-video-prompt-template.md",),
    "papercraft-stop-motion-explainer": ("references/shot-table-spec.md",),
    "paper-collage-explainer-generator": ("references/shot-table-spec.md",),
}

KEYWORDS = (
    "shot",
    "continuity",
    "reference",
    "anchor",
    "per-second",
    "camera",
    "motion",
    "audio",
    "dialogue",
    "sound",
    "quality",
    "negative",
    "identity",
    "landmark",
    "position",
)


def selected_skill_excerpt(
    skills_dir: Path | None,
    style_preset: str,
    *,
    max_chars: int = 4000,
) -> str:
    """Return a bounded excerpt from the style-matched local skill pack."""

    if skills_dir is None or not skills_dir.is_dir():
        return ""
    pack_names = STYLE_PACKS.get(style_preset.casefold(), ())
    excerpts: list[str] = []
    remaining = max_chars
    for pack_name in pack_names:
        if remaining <= 0:
            break
        pack_dir = skills_dir / pack_name
        paths = [pack_dir / name for name in REFERENCE_FILES.get(pack_name, ())]
        paths.append(pack_dir / "SKILL.md")
        pack_values: list[str] = []
        for skill_path in paths:
            if not skill_path.is_file():
                continue
            text = _extract_relevant_lines(skill_path.read_text(encoding="utf-8", errors="replace"))
            if text:
                pack_values.append(text)
        if not pack_values:
            continue
        pack_text = "\n".join(pack_values)
        excerpt = f"[{pack_name} selected director pack]\n{pack_text}"
        excerpt = excerpt[:remaining]
        excerpts.append(excerpt)
        remaining -= len(excerpt)
    return "\n\n".join(excerpts)


def _extract_relevant_lines(text: str) -> str:
    lines = text.splitlines()
    selected: set[int] = set()
    for index, line in enumerate(lines):
        lowered = line.casefold()
        if any(keyword in lowered for keyword in KEYWORDS) and not any(
            excluded in lowered for excluded in ("canvas", "choice card", "hub_generate", "pixar-inspired")
        ):
            selected.update(range(max(0, index - 1), min(len(lines), index + 3)))
    # Keep document order and collapse blank runs so the excerpt is a useful
    # instruction, not a random bag of keyword matches.
    values: list[str] = []
    blank = False
    for index, line in enumerate(lines):
        if index not in selected:
            continue
        lowered = line.casefold()
        if any(excluded in lowered for excluded in ("canvas", "choice card", "hub_generate", "pixar-inspired")):
            continue
        if not line.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        values.append(line.rstrip())
    return "\n".join(values).strip()
