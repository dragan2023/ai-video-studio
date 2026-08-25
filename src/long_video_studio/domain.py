from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from long_video_studio.h3_limits import H3_MAX_SHOT_SECONDS, H3_MIN_SHOT_SECONDS

# Project duration is an aggregate of short H3 shots, not one model request.
MAX_PROJECT_DURATION_SECONDS = 14_400


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class AssetKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"


class AssetRole(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    PROP = "prop"
    STYLE = "style"
    START_FRAME = "start_frame"
    AUDIO = "audio"
    REFERENCE = "reference"


class ContinuationMode(str, Enum):
    """Creator-facing trade-off for extending an already rendered clip."""

    # FL2VA-only generation is intentionally separate from the two Ref2VA
    # policies.  The project may either generate a fresh shot anchor (default)
    # or reuse the previous boundary frame for compatibility.
    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    QUALITY = "quality"


class UltraFastAnchorStrategy(str, Enum):
    """How an ultra-fast FL2VA shot obtains its opening frame."""

    INDEPENDENT = "independent"
    BOUNDARY = "boundary"


class UltraFastTransition(str, Enum):
    """Project-level edit applied between independent ultra-fast shots."""

    FADE_BLACK = "fade_black"
    DISSOLVE = "dissolve"
    HARD_CUT = "hard_cut"
    RANDOM = "random"


class AssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("asset"))
    sha256: str
    original_name: str
    display_name: str = ""
    media_type: str
    kind: AssetKind
    size_bytes: int
    stored_path: str | None = None
    external_path: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    roles: list[AssetRole] = Field(default_factory=lambda: [AssetRole.REFERENCE])
    source: Literal["upload", "path"] = "upload"
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def resolved_path(self) -> str:
        value = self.stored_path or self.external_path
        if not value:
            raise ValueError(f"asset {self.id} has no readable path")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        return sorted({item.strip().lower() for item in value if item.strip()})

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return " ".join(value.split())


class AssetUpdate(BaseModel):
    display_name: str | None = None
    caption: str | None = None
    tags: list[str] | None = None
    roles: list[AssetRole] | None = None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return sorted({item.strip().lower() for item in value if item.strip()})

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        return None if value is None else " ".join(value.split())


class AssetView(BaseModel):
    """Creator-facing asset metadata without server filesystem paths."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    sha256: str
    original_name: str
    display_name: str
    media_type: str
    kind: AssetKind
    size_bytes: int
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    caption: str
    tags: list[str]
    roles: list[AssetRole]
    source: Literal["upload", "path"]
    created_at: datetime


class ProjectBrief(BaseModel):
    title: str = "Untitled film"
    prompt: str = Field(min_length=3)
    duration_seconds: int = Field(default=60, ge=15, le=MAX_PROJECT_DURATION_SECONDS)
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    style: str = "cinematic realism"
    style_preset: str = "cinematic"
    style_instructions: str = ""
    language: str = "zh-CN"
    audience: str = "general"
    reference_asset_ids: list[str] = Field(default_factory=list)
    quality: Literal["draft", "final"] = "draft"
    subtitle_mode: Literal["none", "sidecar"] = "none"
    continuation_mode: ContinuationMode = ContinuationMode.FAST
    ultra_fast_anchor_strategy: UltraFastAnchorStrategy = UltraFastAnchorStrategy.INDEPENDENT
    ultra_fast_transition: UltraFastTransition = UltraFastTransition.FADE_BLACK
    ultra_fast_transition_seconds: float = Field(default=0.6, ge=0.1, le=2.0)


class ContinuityState(BaseModel):
    active_subject_ids: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    wardrobe: list[str] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    # These fields are optional so existing project records remain readable.
    # They make the spatial/identity handoff
    # explicit instead of forcing the compiler to infer it from prose.
    fixed_landmarks: list[str] = Field(default_factory=list)
    character_positions: list[str] = Field(default_factory=list)
    exited_characters: list[str] = Field(default_factory=list)
    performance: str = ""
    spatial_anchor: str = ""
    handoff: str = ""
    location: str = ""
    lighting: str = ""
    camera: str = ""
    action: str = ""
    audio: str = ""


class SubjectCard(BaseModel):
    """Stable identity binding shared by every shot director."""

    subject_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    visual_identity: str = ""
    wardrobe: str = ""
    reference_asset_ids: list[str] = Field(default_factory=list)
    speaker_id: str | None = None


class WorldBible(BaseModel):
    logline: str
    visual_style: str
    character_notes: list[str] = Field(default_factory=list)
    location_notes: list[str] = Field(default_factory=list)
    prop_notes: list[str] = Field(default_factory=list)
    audio_notes: list[str] = Field(default_factory=list)
    continuity_rules: list[str] = Field(default_factory=list)
    subjects: list[SubjectCard] = Field(default_factory=list)


class LLMClient(BaseModel):
    """A persisted, UI-managed LLM client used by the planner.

    Credentials stay server-side; ``public()`` never exposes ``api_key``.
    ``available`` requires a base URL and model so the UI can flag clients
    that cannot be used to call the planner.
    """

    id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(default="", max_length=120)
    base_url: str = Field(default="", max_length=400)
    api_key: str | None = Field(default=None)
    model: str = Field(default="", max_length=200)
    wire_api: str = Field(default="chat_completions", max_length=40)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        value = value.strip().lower()
        if not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError("llm client id may only contain letters, digits, hyphen and underscore")
        return value

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name or self.id,
            "base_url": self.base_url,
            "model": self.model or "",
            "wire_api": self.wire_api,
            "available": bool(self.base_url.strip() and self.model.strip()),
        }


class ShotTask(str, Enum):
    FL2VA = "fl2va"
    REF2VA = "ref2va"


class TransitionKind(str, Enum):
    """How a shot boundary should be authored and rendered."""

    CONTINUOUS = "continuous"
    CAMERA_MOVE = "camera_move"
    MATCH_CUT = "match_cut"
    OCCLUSION_CUT = "occlusion_cut"
    HARD_CUT = "hard_cut"
    ANCHOR = "anchor"


class ShotStatus(str, Enum):
    PLANNED = "planned"
    READY = "ready"
    RENDERING = "rendering"
    COMPLETE = "complete"
    FAILED = "failed"


class StartFrameSource(str, Enum):
    """Declared origin of a shot's opening frame in a preproduction plan."""

    CREATOR_ASSET = "creator_asset"
    PREVIOUS_BOUNDARY = "previous_boundary"
    SYSTEM_BLACK = "system_black"
    GENERATE_T2I = "generate_t2i"
    NEEDS_REVIEW = "needs_review"


class PreproductionStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    GENERATING_ASSETS = "generating_assets"
    READY = "ready"
    BLOCKED = "blocked"


class PreproductionRuntimeProfile(BaseModel):
    """Fixed, creator-visible runtime settings; never provider configuration."""

    resolution: str = "1536x864"
    max_concurrency: int = Field(default=1, ge=1, le=1)
    inference_steps: int = Field(default=12, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)


class PreproductionShotPlan(BaseModel):
    """The reviewable preproduction decision for one storyboard shot."""

    shot_id: str = Field(min_length=1)
    shot_index: int = Field(ge=0)
    script_evidence: str = ""
    transition_kind: TransitionKind = TransitionKind.CONTINUOUS
    start_frame_source: StartFrameSource = StartFrameSource.NEEDS_REVIEW
    source_shot_id: str | None = None
    candidate_asset_ids: list[str] = Field(default_factory=list)
    selected_asset_id: str | None = None
    gap_reason: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    generation_permitted: bool = False
    parameter_summary: str = ""

    @model_validator(mode="after")
    def validate_start_frame_source(self) -> PreproductionShotPlan:
        if self.start_frame_source == StartFrameSource.PREVIOUS_BOUNDARY:
            if not self.source_shot_id:
                raise ValueError("previous_boundary start frames require source_shot_id")
            if self.source_shot_id == self.shot_id:
                raise ValueError("previous_boundary source_shot_id must refer to another shot")
        return self


class PreproductionPlan(BaseModel):
    """Persistent approval record for the visible preproduction workflow."""

    version: int = Field(default=1, ge=1)
    asset_input_fingerprint: str = ""
    generated_image_count: int = Field(default=0, ge=0)
    runtime_profile: PreproductionRuntimeProfile = Field(default_factory=PreproductionRuntimeProfile)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    approved_at: datetime | None = None
    status: PreproductionStatus = PreproductionStatus.DRAFT
    shot_plans: list[PreproductionShotPlan] = Field(default_factory=list)

    @property
    def shots(self) -> list[PreproductionShotPlan]:
        """Compatibility-friendly shorthand for consumers displaying plan rows."""

        return self.shot_plans

    @model_validator(mode="after")
    def validate_plan(self) -> PreproductionPlan:
        seen_shot_ids: set[str] = set()
        seen_indexes: set[int] = set()
        for shot_plan in self.shot_plans:
            if shot_plan.shot_id in seen_shot_ids:
                raise ValueError("preproduction plan has duplicate shot_id")
            if shot_plan.shot_index in seen_indexes:
                raise ValueError("preproduction plan has duplicate shot_index")
            seen_shot_ids.add(shot_plan.shot_id)
            seen_indexes.add(shot_plan.shot_index)

        if self.shot_plans:
            first_shot = min(self.shot_plans, key=lambda item: item.shot_index)
            if first_shot.start_frame_source == StartFrameSource.PREVIOUS_BOUNDARY:
                raise ValueError("the first shot cannot use a previous_boundary start frame")

        if self.status == PreproductionStatus.READY:
            if self.blockers:
                raise ValueError("a plan with blockers cannot be ready")
            if self.approved_at is None:
                raise ValueError("a ready plan requires approval")
        return self

    def become_ready(self) -> PreproductionPlan:
        """Advance an approved asset workflow to ready after its blockers clear."""

        if self.status not in {PreproductionStatus.APPROVED, PreproductionStatus.GENERATING_ASSETS}:
            raise ValueError("only approved or generating_assets plans may become ready")
        if self.blockers:
            raise ValueError("a plan with blockers cannot become ready")
        if self.approved_at is None:
            raise ValueError("an approved plan requires approved_at before becoming ready")
        return type(self).model_validate(
            {**self.model_dump(mode="python"), "status": PreproductionStatus.READY}
        )


class DialogueLine(BaseModel):
    """One explicit spoken line.

    Visual direction never belongs here. Keeping speech in a typed collection
    lets model adapters render H3 dialogue tags without guessing from quotes or
    narrative prose.
    """

    speaker: str = Field(min_length=1)
    # ``speaker`` is creator-facing text and may be an alias returned by an
    # agent.  These optional fields are the canonical runtime binding.  The
    # planner fills them from ``WorldBible.subjects`` before a shot reaches a
    # model adapter, while keeping old project records readable.
    subject_id: str | None = Field(default=None, min_length=1)
    speaker_id: str | None = Field(default=None, min_length=1)
    text: str = Field(min_length=1)
    language: str = "Chinese"
    delivery: str = "natural"
    mode: Literal["on_screen", "off_screen", "voice_over"] = "on_screen"
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)

    @field_validator("speaker", "text", "language", "delivery", mode="before")
    @classmethod
    def strip_dialogue_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_timing(self) -> DialogueLine:
        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("dialogue end_seconds must be greater than start_seconds")
        return self


class StoryboardBeat(BaseModel):
    """One observable timeline beat in a model-facing storyboard shot."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    # A static, black, or sound-led beat may intentionally have no headline action.
    visual_action: str = ""
    state_change: str = ""
    camera: str = ""
    sound: str = ""
    # H3's shot-table skill calls these out separately from the headline
    # action. Defaults keep the fields optional for existing storyboards.
    performance: str = ""
    spatial_anchor: str = ""
    handoff: str = ""

    @model_validator(mode="after")
    def validate_timing(self) -> StoryboardBeat:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("storyboard beat end_seconds must be greater than start_seconds")
        return self


class ShotSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("shot"))
    index: int = Field(ge=0)
    title: str
    purpose: str
    source_section: str = ""
    # H3 accepts a nominal 15-second output request and aligns it to 362 frames
    # / about 15.083 seconds. Ref2VA reference inputs are trimmed separately.
    duration_seconds: float = Field(ge=H3_MIN_SHOT_SECONDS, le=H3_MAX_SHOT_SECONDS)
    task: ShotTask = ShotTask.FL2VA
    transition_kind: TransitionKind = TransitionKind.CONTINUOUS
    prompt: str = Field(
        min_length=1,
        description="Visual-only scene, subject, action, lighting, composition, and camera direction.",
    )
    audio_prompt: str = ""
    music_prompt: str = ""
    dialogue: list[DialogueLine] = Field(default_factory=list)
    opening_state: str = ""
    ending_state: str = ""
    continuity_handoff: str = ""
    reference_anchors: list[str] = Field(default_factory=list)
    hook: str = ""
    visual_beats: list[StoryboardBeat] = Field(default_factory=list)
    negative_prompt: str = ""
    subtitle_text: str | None = None
    camera: str = "medium shot, stable cinematic camera"
    reference_asset_ids: list[str] = Field(default_factory=list)
    start_frame_asset_id: str | None = None
    audio_asset_id: str | None = None
    continuity_from_shot_id: str | None = None
    continuation_mode: ContinuationMode | None = None
    continuity_in: ContinuityState = Field(default_factory=ContinuityState)
    continuity_out: ContinuityState = Field(default_factory=ContinuityState)
    seed: int = 42
    fps: int = 24
    inference_steps: int = 12
    flow_shift: float = 12.0
    status: ShotStatus = ShotStatus.PLANNED
    selected_take_path: str | None = None
    anchor_frame_path: str | None = None
    anchor_prompt: str = Field(
        default="",
        description=(
            "Complete direct-to-image opening-frame prompt. Image Edit prompts bind each ordered reference by "
            "ordinal and display name; zero-reference T2I prompts stand alone. Runtime adapters do not expand it. "
            "The planner normalizes agent output to Studio's 1000-character preflight before persistence."
        ),
    )
    boundary_frame_path: str | None = None
    # Render telemetry is persisted with the shot so the Studio can show the
    # measured cost of the latest attempt after a page reload.  These fields
    # are runtime metadata, not creator-editable storyboard content.
    render_started_at: datetime | None = None
    render_completed_at: datetime | None = None
    render_duration_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_reference_contract(self) -> ShotSpec:
        if self.start_frame_asset_id and self.start_frame_asset_id not in self.reference_asset_ids:
            self.reference_asset_ids.insert(0, self.start_frame_asset_id)
        return self

    @field_validator("prompt", "audio_prompt", "music_prompt", "anchor_prompt", mode="before")
    @classmethod
    def strip_prompt_fields(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_prompt_separation(self) -> ShotSpec:
        for value in (self.prompt, self.audio_prompt, self.music_prompt, self.anchor_prompt):
            lowered = value.casefold()
            if "<d>" in lowered or "</d>" in lowered:
                raise ValueError("dialogue tags are only allowed in dialogue.text")
        for line in self.dialogue:
            if line.end_seconds is not None and line.end_seconds > self.duration_seconds:
                raise ValueError("dialogue timing must fit within the shot duration")
        for beat in self.visual_beats:
            if beat.end_seconds > self.duration_seconds:
                raise ValueError("storyboard beat timing must fit within the shot duration")
        return self


def resolved_continuation_mode(project: FilmProject, shot: ShotSpec) -> ContinuationMode:
    """Resolve a per-shot override against the project's creator choice."""

    return shot.continuation_mode or project.brief.continuation_mode


def uses_independent_ultra_fast_anchor(project: FilmProject, shot: ShotSpec) -> bool:
    """Return whether this shot should receive a fresh FL2VA opening frame."""

    return (
        resolved_continuation_mode(project, shot) == ContinuationMode.ULTRA_FAST
        and project.brief.ultra_fast_anchor_strategy == UltraFastAnchorStrategy.INDEPENDENT
        and not shot.start_frame_asset_id
    )


def effective_video_task(
    shot: ShotSpec,
    *,
    ref2va_configured: bool,
    fl2va_configured: bool,
    continuation_mode: ContinuationMode | None = None,
) -> ShotTask:
    """Select the runtime task without changing the storyboard's creative IR.

    FL2VA remains the internal compatibility fallback when it is the only
    configured continuation backend. A creator-selected start frame always
    forces FL2VA because that explicit composition must not be replaced by a
    previous-video reference.
    """

    if shot.start_frame_asset_id:
        return ShotTask.FL2VA
    if continuation_mode == ContinuationMode.ULTRA_FAST:
        # Both ultra-fast strategies are FL2VA-only.  Independent shots may
        # intentionally have no continuity_from_shot_id at all.
        return ShotTask.FL2VA
    is_generated_clip_continuation = bool(shot.continuity_from_shot_id)
    if not is_generated_clip_continuation:
        return shot.task
    if shot.task == ShotTask.REF2VA or ref2va_configured:
        return ShotTask.REF2VA
    if fl2va_configured:
        return ShotTask.FL2VA
    # Ref2VA is the intended continuation path. Returning it when neither
    # endpoint is configured makes preflight name the missing primary backend
    # instead of silently presenting FL2VA as the normal route.
    return ShotTask.REF2VA


class TimelineClip(BaseModel):
    shot_id: str
    start_seconds: float
    duration_seconds: float


class PlannerTraceEvent(BaseModel):
    """Bounded planner diagnostic event shown in the Studio debug console."""

    id: str = Field(default_factory=lambda: new_id("trace"))
    created_at: datetime = Field(default_factory=utc_now)
    stage: str
    status: Literal["started", "request", "response", "completed", "failed", "fallback", "client"]
    message: str = ""
    request_payload: str | None = None
    response_payload: str | None = None
    error: str | None = None
    duration_ms: float | None = None


class BatchPlanningStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETE = "complete"


class BatchPlanningRun(BaseModel):
    """Persisted, credential-free state for imported-script H3 enrichment."""

    status: BatchPlanningStatus = BatchPlanningStatus.QUEUED
    profile_id: str = "default"
    model: str = ""
    batch_size: int = Field(default=6, ge=1, le=24)
    completed_shot_ids: list[str] = Field(default_factory=list)
    failed_shot_ids: list[str] = Field(default_factory=list)
    current_batch_start: int = Field(default=0, ge=0)
    last_error: str = ""
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def completed_count(self) -> int:
        return len(self.completed_shot_ids)


class FilmProject(BaseModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    brief: ProjectBrief
    world_bible: WorldBible
    shots: list[ShotSpec]
    timeline: list[TimelineClip] = Field(default_factory=list)
    status: Literal["planning", "planned", "compiled", "rendering", "complete", "failed"] = "planned"
    planner_trace: list[PlannerTraceEvent] = Field(default_factory=list)
    preproduction_plan: PreproductionPlan | None = None
    batch_planning_run: BatchPlanningRun | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def build_timeline(self) -> FilmProject:
        cursor = 0.0
        self.timeline = []
        for index, shot in enumerate(sorted(self.shots, key=lambda item: item.index)):
            shot.index = index
            self.timeline.append(
                TimelineClip(
                    shot_id=shot.id,
                    start_seconds=cursor,
                    duration_seconds=shot.duration_seconds,
                )
            )
            cursor += shot.duration_seconds
        return self


class ModelCapability(BaseModel):
    id: str
    display_name: str
    task: str
    endpoint: str | None = None
    available: bool = False
    max_duration_seconds: float | None = None
    supports_audio: bool = False
    supports_multiple_references: bool = False
    recommended_gpus: int = 0
    notes: list[str] = Field(default_factory=list)


class ExecutionStage(BaseModel):
    id: str = Field(default_factory=lambda: new_id("stage"))
    shot_id: str | None = None
    kind: Literal["keyframe", "video", "continuity_check", "assembly"]
    capability_id: str
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    estimated_seconds: float | None = None


class DeploymentRequest(BaseModel):
    capability_id: str
    endpoint: str | None = None
    recommended_gpus: int = 0
    shot_ids: list[str] = Field(default_factory=list)
    status: Literal["ready", "unconfigured"]
    rationale: str


class ExecutionPlan(BaseModel):
    project_id: str
    stages: list[ExecutionStage]
    deployments: list[DeploymentRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimated_seconds: float


class RenderJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    project_id: str
    # D7-M：单镜渲染支持。None=全项目顺序渲染；非空=只渲染这些 shot（其余不动、不组装 final.mp4）。
    shot_ids: list[str] | None = None
    status: Literal["queued", "running", "complete", "failed"] = "queued"
    progress: float = Field(default=0, ge=0, le=1)
    current_shot_id: str | None = None
    current_service_id: Literal["fl2va", "ref2va", "image_edit", "t2i"] | None = None
    message: str = "queued"
    output_path: str | None = None
    subtitle_path: str | None = None
    error: str | None = None
    force_rerender: bool = False
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=0, ge=0)
    estimated_seconds: float | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class RenderObservation(BaseModel):
    """Append-only successful render timing used to calibrate future ETAs."""

    id: str = Field(default_factory=lambda: new_id("observation"))
    source_key: str
    project_id: str
    shot_id: str
    render_profile: str
    task: ShotTask
    continuation_mode: str
    aspect_ratio: str
    duration_seconds: float = Field(gt=0)
    inference_steps: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class ShotRenderEstimate(BaseModel):
    shot_id: str
    task: ShotTask
    continuation_mode: str
    estimated_seconds: float = Field(ge=0)
    remaining_seconds: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    source: Literal["history", "configured"]
    confidence: Literal["low", "medium", "high"]


class ProjectRenderEstimate(BaseModel):
    project_id: str
    total_seconds: float = Field(ge=0)
    remaining_seconds: float = Field(ge=0)
    assembly_seconds: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    source: Literal["history", "configured", "mixed"]
    shots: list[ShotRenderEstimate] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
