"""Canonical directing styles shared by the planner and creator UI.

The H3 prompt-writing skills treat style as a global contract rather than a
single adjective.  Keeping that contract in one small, serializable module
prevents the API, the heuristic planner, and the web client from silently
drifting apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StyleContract:
    """A reusable style DNA for a complete film, not just one shot."""

    id: str
    label: str
    copy: str
    color: str
    medium: str
    palette: str
    lighting: str
    camera: str
    motion: str
    composition: str
    texture: str
    audio: str
    negative_constraints: tuple[str, ...]

    def render(self) -> str:
        """Render a dense style contract suitable for an LLM planner."""

        exclusions = ", ".join(self.negative_constraints)
        return "\n".join(
            [
                f"Style DNA: {self.label} ({self.id})",
                f"Medium: {self.medium}",
                f"Palette: {self.palette}",
                f"Lighting: {self.lighting}",
                f"Lens and camera language: {self.camera}",
                f"Motion rhythm: {self.motion}",
                f"Composition: {self.composition}",
                f"Material and image texture: {self.texture}",
                f"Audio strategy: {self.audio}",
                f"Global negative constraints: {exclusions}",
                (
                    "Apply this contract to the world bible and every shot. "
                    "Do not replace it with a new palette, light direction, lens grammar, "
                    "or motion rhythm unless the creator explicitly requests a change."
                ),
            ]
        )

    def compact(self) -> str:
        """Return the small, repeatable style lock embedded in shot prompts."""

        return (
            f"{self.label}; palette {self.palette}; lighting {self.lighting}; "
            f"camera {self.camera}; motion {self.motion}; texture {self.texture}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for the public API."""

        value = asdict(self)
        value["negative_constraints"] = list(self.negative_constraints)
        value["instructions"] = self.render()
        value["ui_instructions"] = self.compact()
        return value


def _style(
    style_id: str,
    label: str,
    copy: str,
    color: str,
    medium: str,
    palette: str,
    lighting: str,
    camera: str,
    motion: str,
    composition: str,
    texture: str,
    audio: str,
    *negative_constraints: str,
) -> StyleContract:
    return StyleContract(
        id=style_id,
        label=label,
        copy=copy,
        color=color,
        medium=medium,
        palette=palette,
        lighting=lighting,
        camera=camera,
        motion=motion,
        composition=composition,
        texture=texture,
        audio=audio,
        negative_constraints=tuple(negative_constraints),
    )


# Keep this order stable: it is also the order shown in the creator-facing UI.
STYLE_REGISTRY: dict[str, StyleContract] = {
    "cinematic": _style(
        "cinematic",
        "电影写实",
        "自然光 · 真实运动",
        "amber",
        "cinematic live-action realism",
        "natural skin tones, restrained warm neutrals, one motivated accent color",
        "motivated natural or practical key light, soft fill, readable shadow detail",
        "35mm/50mm grammar, deliberate wide-to-medium coverage, motivated push-in or pull-out",
        "setup → anticipation → action → brake → settle; one primary action per beat",
        "clear foreground, subject in the midground, geographic landmarks in the background",
        "natural skin, fabric and prop detail; subtle film grain, no plastic smoothing",
        "diegetic room tone and action sounds stay synchronized; dialogue remains intelligible",
        "random lens changes",
        "unmotivated camera resets",
        "plastic skin",
    ),
    "documentary": _style(
        "documentary",
        "纪录片",
        "手持感 · 生活质感",
        "mint",
        "observational documentary realism",
        "available-light neutrals with honest, location-specific color",
        "available light first, imperfect exposure retained, faces remain readable",
        "handheld medium shots, gentle reframing, occasional static observation",
        "real pauses and breath → small decision → physical response; no staged montage",
        "respect eyelines and real geography; allow useful negative space",
        "natural motion blur, lived-in surfaces, restrained grading",
        "room tone, footsteps, clothing and incidental sounds are foregrounded",
        "glossy commercial lighting",
        "perfectly stabilized camera",
        "artificial slow motion",
    ),
    "music_video": _style(
        "music_video",
        "音乐短片",
        "节奏感 · 强烈构图",
        "violet",
        "stylized cinematic music-video realism",
        "bold complementary palette with a locked accent color and controlled contrast",
        "designed keys and colored practicals, lighting changes only at planned beats",
        "wide/medium/close coverage with motivated tracking, arc, whip-pan or snap zoom",
        "beat-synced setup → hit → release; vary intensity while preserving identity and geography",
        "graphic silhouettes, layered depth, strong negative space and deliberate subject scale",
        "clean highlights, tactile wardrobe, controlled grain and intentional motion blur",
        "one master musical pulse; impacts and gestures land on the beat without replacing dialogue",
        "random flashing lights",
        "identity drift",
        "unmotivated cuts",
    ),
    "commercial": _style(
        "commercial",
        "品牌广告",
        "精致光线 · 高级质感",
        "rose",
        "premium cinematic product realism",
        "clean branded palette, controlled saturation, product accent color stays consistent",
        "soft key, shaped rim, polished highlights and an explicit product-facing light direction",
        "precise dolly, slider, pedestal and product close-up language; no accidental shake",
        "reveal → demonstrate → payoff; every gesture supports a single memorable benefit",
        "hero subject/product isolated with uncluttered background and readable silhouette",
        "high-detail materials, clean skin and fabric, subtle premium grain",
        "clear product/action Foley and a restrained music bed; speech is crisp and purposeful",
        "cluttered background",
        "unreadable product label",
        "generic stock-video gestures",
    ),
    "noir": _style(
        "noir",
        "黑色电影",
        "低调光 · 悬疑氛围",
        "blue",
        "neo-noir live-action realism",
        "near-black shadows, desaturated cool mids, restrained amber practicals",
        "single motivated side or back key, hard-edged shadows, selective eye light",
        "locked-off tension, slow dolly, lateral track and controlled close-up",
        "observation → hesitation → committed move → withheld reveal; tension beats stay legible",
        "oblique lines, doorframes and deep foreground occlusion frame the subject",
        "wet surfaces, textured blacks, fine grain and realistic low-light noise",
        "distant city/room tone, footsteps and object Foley carry suspense; silence is intentional",
        "flat front light",
        "cheerful high-key palette",
        "fog that hides the subject",
    ),
    "animation": _style(
        "animation",
        "手绘动画",
        "笔触感 · 想象力",
        "peach",
        "hand-drawn cinematic animation",
        "harmonized storybook palette, clear character accent colors and readable value groups",
        "soft illustrated shading with stable key direction and deliberately shaped highlights",
        "readable staging, multiplane pan, gentle truck and expressive but bounded camera moves",
        "anticipation → squash/stretch or expressive pose → follow-through → settle",
        "silhouette-first staging, clear poses, layered backgrounds and consistent scale",
        "visible brush texture, coherent line weight, no accidental photorealism",
        "designed Foley and expressive ambience synchronized to gestures; music supports the beat",
        "style switching between shots",
        "photorealistic skin",
        "inconsistent character proportions",
    ),
    "retro": _style(
        "retro",
        "复古胶片",
        "颗粒感 · 怀旧色调",
        "amber",
        "period-inspired analog film realism",
        "faded warm highlights, gentle cyan shadows, restrained vintage saturation",
        "soft motivated practicals, halation around highlights, period-consistent contrast",
        "patient 35mm/50mm framing, slow pans, dolly moves and occasional documentary handheld",
        "unhurried setup → tactile action → lingering settle; preserve period gesture",
        "classical balanced framing, practical landmarks and modest camera height",
        "fine grain, halation, slight gate weave and authentic material texture",
        "period room tone and tactile Foley; music stays inside the chosen era",
        "digital-sharp clinical image",
        "modern color grading",
        "anachronistic props",
    ),
    "surreal": _style(
        "surreal",
        "超现实",
        "梦境感 · 非日常",
        "violet",
        "grounded surreal cinematic realism",
        "recognizable base palette with one impossible accent or color event",
        "physically coherent key light plus one clearly motivated dreamlike anomaly",
        "slow orbit, impossible-but-readable perspective shift, controlled push-in and static holds",
        "ordinary setup → subtle anomaly → escalating transformation → calm landing",
        "real geography remains legible while scale, reflection or background logic bends deliberately",
        "tactile realism with selective dream distortion; faces and hands remain stable",
        "ordinary ambience subtly stretches; anomaly sounds are sparse, synchronized and never chaotic",
        "random hallucinated objects",
        "lost subject identity",
        "unreadable spatial relationships",
    ),
    "energetic": _style(
        "energetic",
        "高能短片",
        "强钩子 · 节奏峰值",
        "rose",
        "high-energy cinematic live action",
        "high-contrast saturated accents over a stable base palette",
        "bright readable key with motivated contrast and clear action separation",
        "tracking, arc, pedestal and rapid but physically plausible reframing",
        "hook → build → peak → brake → satisfying resolve; one readable action at a time",
        "strong silhouettes, clear action lanes, subject remains inside a stable geographic frame",
        "crisp detail, controlled motion blur and consistent wardrobe/prop texture",
        "rhythmic action Foley and breath lead the mix; music amplifies rather than masks events",
        "unreadable speed",
        "teleportation",
        "constant camera shake",
    ),
    "custom": _style(
        "custom",
        "自定义导演模板",
        "用户规则 · 可复用",
        "peach",
        "creator-defined cinematic language",
        "a deliberately chosen palette with stable subject accent colors",
        "a single motivated light direction and explicit contrast/temperature rule",
        "a consistent lens vocabulary and motivated camera movement",
        "setup → anticipation → primary action → settle unless the creator overrides it",
        "clear subject hierarchy, geography and continuity anchors",
        "one consistent medium/texture treatment across every shot",
        "a deliberate diegetic and musical strategy with synchronized action sound",
        "style drift",
        "unmotivated camera or lighting changes",
        "identity and wardrobe drift",
    ),
}


def get_style_contract(style_id: str | None, custom_instructions: str = "") -> StyleContract:
    """Resolve a preset ID, treating locally-created UI styles as custom."""

    normalized = (style_id or "").strip().lower()
    if normalized in STYLE_REGISTRY:
        return STYLE_REGISTRY[normalized]
    if custom_instructions.strip():
        return STYLE_REGISTRY["custom"]
    return STYLE_REGISTRY["cinematic"]


def style_prompt(style_id: str | None, custom_instructions: str = "") -> str:
    """Render a preset plus optional creator overrides for planner instructions."""

    contract = get_style_contract(style_id, custom_instructions)
    override = custom_instructions.strip()
    if not override:
        return contract.render()
    return f"{contract.render()}\nCreator overrides (highest priority): {override}"


def public_style_contracts() -> list[dict[str, Any]]:
    """Return canonical presets for the creator-facing API."""

    return [contract.as_dict() for contract in STYLE_REGISTRY.values()]
