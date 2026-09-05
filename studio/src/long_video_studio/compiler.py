from __future__ import annotations

from typing import TYPE_CHECKING

from long_video_studio.adapters.image_edit import known_multi_image_support
from long_video_studio.anchor_policy import IMAGE_EDIT_ANCHOR_MODES, anchor_selected
from long_video_studio.config import Settings
from long_video_studio.dialogue_harness import (
    prepare_dialogue,
    validate_active_speakers,
)
from long_video_studio.domain import (
    AssetKind,
    ContinuationMode,
    DeploymentRequest,
    ExecutionPlan,
    ExecutionStage,
    FilmProject,
    ModelCapability,
    ShotSpec,
    ShotTask,
    effective_video_task,
    resolved_continuation_mode,
    uses_independent_ultra_fast_anchor,
)
from long_video_studio.h3_limits import H3_MAX_SHOT_SECONDS

if TYPE_CHECKING:
    from long_video_studio.estimator import RenderEstimator
    from long_video_studio.repository import StudioRepository


class FilmCompiler:
    """Compile creator-level Film IR into an infrastructure-facing execution plan."""

    def __init__(
        self,
        settings: Settings,
        estimator: RenderEstimator | None = None,
        repository: StudioRepository | None = None,
    ):
        self.settings = settings
        self.estimator = estimator
        self.repository = repository

    def capabilities(self) -> list[ModelCapability]:
        image_edit_configured = bool(
            self.settings.image_edit_provider not in {"", "disabled", "none"}
            and self.settings.image_edit_base_url
            and self.settings.image_edit_model
        )
        known_multi_image = (
            known_multi_image_support(self.settings.image_edit_model) if self.settings.image_edit_model else None
        )
        invalid_reference_limit = known_multi_image is False and self.settings.image_edit_max_references > 1
        image_edit_available = image_edit_configured and not invalid_reference_limit
        text_to_image_available = bool(
            self.settings.text_to_image_provider not in {"", "disabled", "none"}
            and self.settings.text_to_image_base_url
        )
        image_edit_notes = [
            f"Provider: {self.settings.image_edit_provider}.",
            "Supports local vLLM-Omni and hosted OpenAI-compatible adapters.",
        ]
        if invalid_reference_limit:
            image_edit_notes.append(
                "Configured checkpoint is single-image; set max references to 1 or use Qwen-Image-Edit-2509."
            )
        return [
            ModelCapability(
                id="qwen-image-edit",
                display_name="Image Edit Provider",
                task="image_edit",
                endpoint=self.settings.image_edit_base_url,
                available=image_edit_available,
                supports_multiple_references=(
                    image_edit_available
                    and self.settings.image_edit_max_references > 1
                    and known_multi_image is not False
                ),
                # The Studio also accepts hosted providers. Do not claim a
                # self-hosted GPU count until that exact backend is validated.
                recommended_gpus=0,
                notes=image_edit_notes,
            ),
            ModelCapability(
                id="qwen-image-t2i",
                display_name="Text-to-Image Provider",
                task="text_to_image",
                endpoint=self.settings.text_to_image_base_url,
                available=text_to_image_available,
                recommended_gpus=0,
                notes=[
                    f"Provider: {self.settings.text_to_image_provider}.",
                    "Generates an authorized opening frame when no material image is selected.",
                ],
            ),
            ModelCapability(
                id="minimax-h3-fl2va",
                display_name="MiniMax-H3 FL2VA",
                task="fl2va",
                endpoint=self.settings.h3_endpoint("fl2va"),
                available=self.settings.h3_configured("fl2va"),
                max_duration_seconds=H3_MAX_SHOT_SECONDS,
                supports_audio=True,
                recommended_gpus=8,
                notes=["First-frame-led video and audio generation."],
            ),
            ModelCapability(
                id="minimax-h3-ref2va",
                display_name="MiniMax-H3 Ref2VA",
                task="ref2va",
                endpoint=self.settings.h3_endpoint("ref2va"),
                available=self.settings.h3_configured("ref2va"),
                max_duration_seconds=H3_MAX_SHOT_SECONDS,
                supports_audio=True,
                supports_multiple_references=True,
                recommended_gpus=8,
                notes=["Reference image plus audio/video conditioning."],
            ),
            ModelCapability(
                id="continuity-qc",
                display_name="Continuity supervisor",
                task="quality_control",
                available=True,
                notes=["MVP validates media boundaries; VLM scoring is an extension point."],
            ),
            ModelCapability(
                id="ffmpeg",
                display_name="Timeline renderer",
                task="assembly",
                available=True,
                notes=["Concatenates selected takes and extracts stable boundary frames."],
            ),
        ]

    def compile(self, project: FilmProject) -> ExecutionPlan:
        over_limit = [
            (shot.index + 1, shot.duration_seconds)
            for shot in project.shots
            if shot.duration_seconds > H3_MAX_SHOT_SECONDS
        ]
        if over_limit:
            raise ValueError(
                f"H3 output-duration ceiling is {H3_MAX_SHOT_SECONDS:g} seconds per shot; refusing to render: "
                + ", ".join(f"shot {index}={duration:g}s" for index, duration in over_limit)
            )
        capability_map = {capability.id: capability for capability in self.capabilities()}
        stages: list[ExecutionStage] = []
        warnings: list[str] = []
        video_stage_by_shot: dict[str, str] = {}
        total_estimate = 0.0
        image_edit = capability_map["qwen-image-edit"]
        text_to_image = capability_map["qwen-image-t2i"]
        anchor_mode = self.settings.image_edit_anchor_mode
        if image_edit.available and anchor_mode not in IMAGE_EDIT_ANCHOR_MODES:
            warnings.append(f"unsupported STUDIO_IMAGE_EDIT_ANCHOR_MODE: {anchor_mode}")

        for position, shot in enumerate(sorted(project.shots, key=lambda value: value.index)):
            if shot.dialogue:
                legacy_dialogue = not project.world_bible.subjects
                prepared = prepare_dialogue(
                    shot.dialogue,
                    shot.duration_seconds,
                    project.world_bible,
                )
                active_subject_ids = shot.continuity_in.active_subject_ids
                if legacy_dialogue and not active_subject_ids and not shot.continuity_in.characters:
                    active_subject_ids = [
                        line.subject_id for line in prepared.lines if line.mode == "on_screen" and line.subject_id
                    ]
                    warnings.append(
                        f"shot {shot.index + 1} used a one-time legacy dialogue roster migration; "
                        "re-plan the project to persist canonical SubjectCards"
                    )
                validate_active_speakers(
                    prepared.lines,
                    prepared.world_bible,
                    shot.continuity_in.characters,
                    active_subject_ids=active_subject_ids,
                    shot_index=shot.index,
                )
                continuity_in = shot.continuity_in.model_copy(update={"active_subject_ids": active_subject_ids})
                shot = shot.model_copy(update={"dialogue": list(prepared.lines), "continuity_in": continuity_in})
            is_ultra_independent = uses_independent_ultra_fast_anchor(project, shot)
            dependencies: list[str] = []
            if shot.continuity_from_shot_id and not shot.start_frame_asset_id:
                previous_stage = video_stage_by_shot.get(shot.continuity_from_shot_id)
                if previous_stage:
                    dependencies.append(previous_stage)

            resolved_mode = resolved_continuation_mode(project, shot)
            continuation_mode = (
                resolved_mode
                if not shot.start_frame_asset_id
                and (shot.continuity_from_shot_id or resolved_mode == ContinuationMode.ULTRA_FAST)
                else None
            )
            runtime_task = effective_video_task(
                shot,
                ref2va_configured=self.settings.h3_configured("ref2va"),
                fl2va_configured=self.settings.h3_configured("fl2va"),
                continuation_mode=continuation_mode,
            )
            is_fl2va = runtime_task == ShotTask.FL2VA
            is_boundary_frame_continuation = (
                bool(shot.continuity_from_shot_id)
                and continuation_mode == ContinuationMode.ULTRA_FAST
                and not is_ultra_independent
            )
            has_explicit_start = bool(shot.start_frame_asset_id)
            has_image_references = self._has_image_references(shot)
            has_start_reference = has_explicit_start or has_image_references
            missing_non_continuity_start = is_fl2va and not shot.continuity_from_shot_id and not has_start_reference
            selected_by_mode = is_fl2va and (
                is_ultra_independent
                or (not is_boundary_frame_continuation and anchor_selected(shot, position, anchor_mode))
            )
            if selected_by_mode and not shot.anchor_prompt:
                raise ValueError(f"shot {shot.index + 1} requires a planner-authored anchor_prompt")

            # Anchor creation is a creative requirement independent of service
            # availability.  Route selected images/boundary frames to Image
            # Edit and zero-image shots to the dedicated T2I provider.
            needs_keyframe = selected_by_mode or missing_non_continuity_start
            if needs_keyframe:
                uses_image_edit = has_image_references or bool(shot.continuity_from_shot_id)
                keyframe_capability = image_edit if uses_image_edit else text_to_image
                keyframe_stage = ExecutionStage(
                    shot_id=shot.id,
                    kind="keyframe",
                    capability_id=keyframe_capability.id,
                    inputs={
                        "reference_asset_ids": shot.reference_asset_ids,
                        "instruction": shot.continuity_in.model_dump(mode="json"),
                        "anchor_prompt": shot.anchor_prompt,
                        "anchor_mode": anchor_mode,
                    },
                    depends_on=list(dependencies),
                    estimated_seconds=20,
                )
                stages.append(keyframe_stage)
                # The generated frame is the sole input to FL2VA.  Keeping the
                # predecessor on the keyframe stage makes the dependency graph
                # explicit for continuous shots and avoids running Image Edit
                # before the previous boundary exists.
                dependencies = [keyframe_stage.id]
                total_estimate += keyframe_stage.estimated_seconds or 0
                if not keyframe_capability.available:
                    route = "Image Edit" if uses_image_edit else "T2I"
                    warnings.append(
                        f"Shot {position + 1} requires a generated anchor frame; {route} is not configured."
                    )
            elif missing_non_continuity_start:
                warnings.append(
                    f"Shot {position + 1} has no non-continuity start image; "
                    f"anchor mode '{anchor_mode}' does not select it."
                )

            capability_id = "minimax-h3-ref2va" if runtime_task == ShotTask.REF2VA else "minimax-h3-fl2va"
            capability = capability_map[capability_id]
            if not capability.available:
                warnings.append(f"{capability.display_name} endpoint is not configured; render stays plan-only.")
            if (
                shot.continuity_from_shot_id
                and not shot.start_frame_asset_id
                and runtime_task == ShotTask.FL2VA
                and shot.task == ShotTask.FL2VA
                and not is_boundary_frame_continuation
            ):
                warnings.append(
                    f"Shot {position + 1} uses the internal FL2VA boundary fallback because "
                    "the Ref2VA endpoint is not configured."
                )
            if is_boundary_frame_continuation:
                warnings.append(f"Shot {position + 1} uses 极速续写: the previous boundary frame is sent to FL2VA.")
            elif is_ultra_independent:
                warnings.append(f"Shot {position + 1} uses 极速续写: a fresh opening frame is generated for FL2VA.")
            stage_continuation_mode = (
                continuation_mode
                if continuation_mode == ContinuationMode.ULTRA_FAST or runtime_task == ShotTask.REF2VA
                else None
            )
            estimate = self._estimate_video_seconds(
                shot.duration_seconds,
                shot.inference_steps,
                stage_continuation_mode,
            )
            video_stage = ExecutionStage(
                shot_id=shot.id,
                kind="video",
                capability_id=capability_id,
                depends_on=dependencies,
                inputs={
                    "anchor_prompt": shot.anchor_prompt,
                    "prompt": shot.prompt,
                    "audio_prompt": shot.audio_prompt,
                    "music_prompt": shot.music_prompt,
                    "dialogue": [line.model_dump(mode="json") for line in shot.dialogue],
                    "negative_prompt": shot.negative_prompt,
                    "duration_seconds": shot.duration_seconds,
                    "transition_kind": shot.transition_kind.value,
                    "fps": shot.fps,
                    "inference_steps": shot.inference_steps,
                    "seed": shot.seed,
                    "reference_asset_ids": shot.reference_asset_ids,
                    "start_frame_asset_id": shot.start_frame_asset_id,
                    "audio_asset_id": shot.audio_asset_id,
                    "continuity_from_shot_id": shot.continuity_from_shot_id,
                    "continuation_mode": stage_continuation_mode.value if stage_continuation_mode else None,
                },
                estimated_seconds=estimate,
            )
            stages.append(video_stage)
            video_stage_by_shot[shot.id] = video_stage.id
            total_estimate += estimate

            qc_stage = ExecutionStage(
                shot_id=shot.id,
                kind="continuity_check",
                capability_id="continuity-qc",
                depends_on=[video_stage.id],
                inputs={
                    "expected_in": shot.continuity_in.model_dump(mode="json"),
                    "expected_out": shot.continuity_out.model_dump(mode="json"),
                },
                estimated_seconds=2,
            )
            stages.append(qc_stage)
            total_estimate += 2

        assembly = ExecutionStage(
            kind="assembly",
            capability_id="ffmpeg",
            depends_on=list(video_stage_by_shot.values()),
            inputs={"timeline": [clip.model_dump(mode="json") for clip in project.timeline]},
            estimated_seconds=max(5, project.brief.duration_seconds * 0.15),
        )
        stages.append(assembly)
        total_estimate += assembly.estimated_seconds or 0
        deployments: list[DeploymentRequest] = []
        for capability_id in (
            "minimax-h3-fl2va",
            "minimax-h3-ref2va",
            "qwen-image-edit",
            "qwen-image-t2i",
        ):
            capability = capability_map[capability_id]
            shot_ids = [
                stage.shot_id
                for stage in stages
                if stage.kind == "video" and stage.capability_id == capability_id and stage.shot_id
            ]
            if not shot_ids and capability_id in {"qwen-image-edit", "qwen-image-t2i"}:
                shot_ids = [
                    stage.shot_id
                    for stage in stages
                    if stage.kind == "keyframe" and stage.capability_id == capability_id and stage.shot_id
                ]
            if not shot_ids:
                continue
            deployments.append(
                DeploymentRequest(
                    capability_id=capability_id,
                    endpoint=capability.endpoint,
                    recommended_gpus=capability.recommended_gpus,
                    shot_ids=shot_ids,
                    status="ready" if capability.available else "unconfigured",
                    rationale=(
                        "Keep this capability warm and batch dependent shots where possible."
                        if capability_id.startswith("minimax-h3")
                        else "Prepare anchor frames before the dependent video stages."
                    ),
                )
            )
        calibrated_total = (
            self.estimator.estimate_project(project, include_completed=True).total_seconds
            if self.estimator is not None
            else total_estimate
        )
        return ExecutionPlan(
            project_id=project.id,
            stages=stages,
            deployments=deployments,
            warnings=list(dict.fromkeys(warnings)),
            estimated_seconds=round(calibrated_total, 1),
        )

    def _has_image_references(self, shot: ShotSpec) -> bool:
        if not shot.reference_asset_ids:
            return False
        if self.repository is None:
            return bool(shot.reference_asset_ids)
        return any(
            (asset := self.repository.get_asset(asset_id)) is not None and asset.kind is AssetKind.IMAGE
            for asset_id in shot.reference_asset_ids
        )

    @staticmethod
    def _estimate_video_seconds(
        duration_seconds: float,
        steps: int,
        continuation_mode: ContinuationMode | None = None,
    ) -> float:
        # Current 4x S5000 TP4/TE4/VAE-PP4/Flash baselines, normalized to
        # 15 seconds and 50 steps. Runtime Studio estimates use RenderEstimator
        # and historical medians; this is the repository-free compiler fallback.
        reference_seconds = {
            None: 594.3,
            ContinuationMode.ULTRA_FAST: 594.3,
            ContinuationMode.FAST: 594.3,
            ContinuationMode.QUALITY: 2570.7,
        }[continuation_mode]
        denoise = reference_seconds * (steps / 50) * (duration_seconds / 15)
        return round(denoise, 1)
