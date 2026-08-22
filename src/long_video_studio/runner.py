from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from long_video_studio.adapters.comfyui_h3 import ComfyUIH3Client
from long_video_studio.adapters.h3 import H3Client
from long_video_studio.adapters.image_edit import (
    ImageEditProvider,
    ImageEditReference,
    ImageEditRequest,
    provider_from_settings,
)
from long_video_studio.adapters.media import MediaTools
from long_video_studio.adapters.text_to_image import (
    TextToImageProvider,
    TextToImageRequest,
    text_to_image_provider_from_settings,
)
from long_video_studio.anchor_policy import IMAGE_EDIT_ANCHOR_MODES, anchor_selected
from long_video_studio.config import Settings
from long_video_studio.domain import (
    AssetKind,
    ContinuationMode,
    FilmProject,
    RenderJob,
    ShotSpec,
    ShotStatus,
    ShotTask,
    UltraFastAnchorStrategy,
    UltraFastTransition,
    effective_video_task,
    resolved_continuation_mode,
    uses_independent_ultra_fast_anchor,
    utc_now,
)
from long_video_studio.estimator import RenderEstimator
from long_video_studio.h3_context import stable_speaker_ids
from long_video_studio.repository import StudioRepository


class RenderManager:
    CONTINUATION_TAIL_SECONDS = 5.0
    CONTINUATION_REF2VA_RULE = (
        "Continue from the moment immediately after the reference video's final frame. "
        "Do not replay, restage, summarize, or repeat any action that already occurred "
        "in the reference video. Begin with the next new action while preserving character "
        "identity, scene geometry, camera direction, motion, and audio continuity."
    )

    def __init__(
        self,
        settings: Settings,
        repository: StudioRepository,
        *,
        estimator: RenderEstimator | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.estimator = estimator or RenderEstimator(settings, repository)
        self.media = MediaTools(settings.ffmpeg_binary, settings.ffprobe_binary)
        self.image_edit_provider: ImageEditProvider | None = None
        self.image_edit_provider_error: str | None = None
        try:
            self.image_edit_provider = provider_from_settings(settings)
        except ValueError as error:
            self.image_edit_provider_error = str(error)
        self.text_to_image_provider: TextToImageProvider | None = None
        self.text_to_image_provider_error: str | None = None
        try:
            self.text_to_image_provider = text_to_image_provider_from_settings(settings)
        except ValueError as error:
            self.text_to_image_provider_error = str(error)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(settings.render_max_concurrency)

    def _h3_configured(self, task: str) -> bool:
        if self.settings.h3_backend == "comfyui":
            return bool(self.settings.comfyui_url and self.settings.comfyui_workflow)
        return bool(self.settings.h3_fl2va_url if task == "fl2va" else self.settings.h3_ref2va_url)

    def _h3_client(self, endpoint: str | None):
        if self.settings.h3_backend == "comfyui":
            if not self.settings.comfyui_url or not self.settings.comfyui_workflow:
                raise RuntimeError("STUDIO_COMFYUI_URL and STUDIO_COMFYUI_WORKFLOW are required")
            return ComfyUIH3Client(
                self.settings.comfyui_url,
                self.settings.comfyui_workflow,
                timeout_seconds=self.settings.h3_timeout_seconds,
                width=self.settings.comfyui_width,
                height=self.settings.comfyui_height,
                steps=self.settings.comfyui_steps,
            )
        if not endpoint:
            raise RuntimeError("H3 endpoint is not configured")
        return H3Client(
            endpoint,
            self.settings.h3_timeout_seconds,
            self.settings.h3_flow_shift,
            self.settings.h3_quality,
        )

    def submit(self, project_id: str, *, force: bool = False) -> RenderJob:
        project = self.repository.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        for job_id, task in tuple(self._tasks.items()):
            if task.done():
                self._tasks.pop(job_id, None)
                continue
            active_job = self.repository.get_job(job_id)
            if active_job and active_job.project_id == project_id:
                return active_job
        estimate = self.estimator.estimate_project(project, include_completed=True)
        job = self.repository.save_job(
            RenderJob(
                project_id=project_id,
                force_rerender=force,
                estimated_seconds=estimate.total_seconds,
            )
        )
        # FastAPI invokes the render route on its event-loop thread.  Resolve
        # that loop explicitly so a worker-thread refactor cannot lose it.
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run(job.id))
        self._tasks[job.id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(job.id, None))
        return job

    async def _run(self, job_id: str) -> None:
        async with self._semaphore:
            await self._run_with_slot(job_id)

    async def _run_with_slot(self, job_id: str) -> None:
        job = self.repository.get_job(job_id)
        if not job:
            return
        project = self.repository.get_project(job.project_id)
        if not project:
            return
        job.status = "running"
        job.message = "starting render"
        job.started_at = utc_now()
        self.repository.save_job(job)
        project.status = "rendering"
        self.repository.save_project(project)
        output_dir = self.settings.output_dir / project.id
        output_dir.mkdir(parents=True, exist_ok=True)
        if job.force_rerender:
            self._clear_forced_render_state(project, output_dir)
            project.updated_at = utc_now()
            self.repository.save_project(project)
        rendered: list[Path] = []
        rendered_by_shot: dict[str, Path] = {}
        boundary_frames: dict[str, Path] = {}
        project_speaker_ids = stable_speaker_ids(project.shots, project.world_bible)
        ordered_shots = sorted(project.shots, key=lambda value: value.index)
        width, height = self._video_canvas(project.brief.aspect_ratio)
        active_shot: ShotSpec | None = None
        active_started_monotonic: float | None = None
        try:
            for position, shot in enumerate(ordered_shots):
                job.current_shot_id = shot.id
                job.progress = position / max(len(project.shots), 1)
                job.message = f"rendering shot {position + 1}/{len(project.shots)}"
                self.repository.save_job(job)
                output_path = output_dir / f"shot-{position + 1:03d}.mp4"
                reusable_take = None if job.force_rerender else self.reusable_take_path(shot)
                if reusable_take is not None:
                    self._set_job_service(job, None)
                    shot.status = ShotStatus.COMPLETE
                    rendered.append(reusable_take)
                    rendered_by_shot[shot.id] = reusable_take
                    boundary = (
                        Path(shot.boundary_frame_path)
                        if shot.boundary_frame_path
                        else output_dir / f"shot-{position + 1:03d}-boundary.png"
                    )
                    if not boundary.is_file():
                        await asyncio.to_thread(
                            self.media.extract_last_stable_frame,
                            reusable_take,
                            boundary,
                        )
                    shot.boundary_frame_path = str(boundary)
                    boundary_frames[shot.id] = boundary
                    job.progress = (position + 1) / len(project.shots)
                    job.message = f"reused shot {position + 1}/{len(project.shots)}"
                    self.repository.save_job(job)
                    self.repository.save_project(project)
                    continue
                shot.status = ShotStatus.RENDERING
                active_shot = shot
                active_started_monotonic = time.monotonic()
                shot.render_started_at = utc_now()
                shot.render_completed_at = None
                shot.render_duration_seconds = None
                self.repository.save_project(project)
                resolved_mode = resolved_continuation_mode(project, shot)
                continuation_mode = (
                    resolved_mode
                    if not shot.start_frame_asset_id
                    and (shot.continuity_from_shot_id or resolved_mode == ContinuationMode.ULTRA_FAST)
                    else None
                )
                runtime_task = effective_video_task(
                    shot,
                    ref2va_configured=self._h3_configured("ref2va"),
                    fl2va_configured=self._h3_configured("fl2va"),
                    continuation_mode=continuation_mode,
                )
                is_ref2va_continuation = bool(
                    runtime_task == ShotTask.REF2VA and shot.continuity_from_shot_id and not shot.start_frame_asset_id
                )
                if runtime_task == ShotTask.FL2VA:
                    anchor = await self._maybe_make_anchor(
                        project,
                        shot,
                        position,
                        boundary_frames,
                        output_dir,
                        job=job,
                    )
                    start_frame = anchor or self._start_frame(shot, boundary_frames)
                    prepared_start = output_dir / f"shot-{position + 1:03d}-start-{width}x{height}.png"
                    await asyncio.to_thread(
                        self.media.fit_image_to_canvas,
                        start_frame,
                        prepared_start,
                        width,
                        height,
                    )
                    if not self._h3_configured("fl2va"):
                        raise RuntimeError("H3 FL2VA backend is not configured")
                    self._set_job_service(job, "fl2va")
                    await self._h3_client(self.settings.h3_fl2va_url).generate_fl2va(
                        shot,
                        prepared_start,
                        output_path,
                        width=width,
                        height=height,
                        async_job=True,
                        brief=project.brief,
                        world_bible=project.world_bible,
                        speaker_ids=project_speaker_ids,
                    )
                elif is_ref2va_continuation:
                    if not self._h3_configured("ref2va"):
                        raise RuntimeError("H3 Ref2VA backend is not configured")
                    self._set_job_service(job, "ref2va")
                    image, media = await self._continuation_ref2va_inputs(
                        project,
                        shot,
                        position,
                        rendered_by_shot,
                        boundary_frames,
                        output_dir,
                    )
                    request_shot = self._with_continuation_rule(shot)
                    await self._h3_client(self.settings.h3_ref2va_url).generate_ref2va(
                        request_shot,
                        image,
                        media,
                        output_path,
                        width=width,
                        height=height,
                        async_job=True,
                        brief=project.brief,
                        world_bible=project.world_bible,
                        previous_shot=next(
                            (candidate for candidate in ordered_shots if candidate.id == shot.continuity_from_shot_id),
                            None,
                        ),
                        speaker_ids=project_speaker_ids,
                    )
                else:
                    if not self._h3_configured("ref2va"):
                        raise RuntimeError("H3 Ref2VA backend is not configured")
                    self._set_job_service(job, "ref2va")
                    image, media = self._ref2va_inputs(shot)
                    await self._h3_client(self.settings.h3_ref2va_url).generate_ref2va(
                        shot,
                        image,
                        media,
                        output_path,
                        width=width,
                        height=height,
                        async_job=True,
                        brief=project.brief,
                        world_bible=project.world_bible,
                        speaker_ids=project_speaker_ids,
                    )
                shot.selected_take_path = str(output_path)
                shot.status = ShotStatus.COMPLETE
                rendered.append(output_path)
                rendered_by_shot[shot.id] = output_path
                boundary = output_dir / f"shot-{position + 1:03d}-boundary.png"
                await asyncio.to_thread(self.media.extract_last_stable_frame, output_path, boundary)
                shot.boundary_frame_path = str(boundary)
                boundary_frames[shot.id] = boundary
                shot.render_completed_at = utc_now()
                if active_started_monotonic is not None:
                    shot.render_duration_seconds = round(time.monotonic() - active_started_monotonic, 3)
                self.repository.save_project(project)
                self._set_job_service(job, None)
                self.estimator.observe(project, shot)
                active_shot = None
                active_started_monotonic = None

            final_path = output_dir / "final.mp4"
            scene_transitions = self._ultra_fast_scene_transitions(project)
            continuous_boundaries = (
                [False] * max(0, len(project.shots) - 1)
                if scene_transitions is not None
                else [
                    bool(shot.continuity_from_shot_id and not shot.start_frame_asset_id) for shot in project.shots[1:]
                ]
            )
            await asyncio.to_thread(
                self.media.concatenate,
                rendered,
                final_path,
                transition_seconds=(
                    project.brief.ultra_fast_transition_seconds
                    if scene_transitions is not None
                    else self.settings.transition_seconds
                ),
                continuous_boundaries=continuous_boundaries,
                scene_transitions=scene_transitions,
            )
            subtitle_path = self._write_sidecar_subtitles(project, output_dir)
            project.status = "complete"
            self.repository.save_project(project)
            job.status = "complete"
            job.current_service_id = None
            job.progress = 1
            job.current_shot_id = None
            job.message = "render complete"
            job.output_path = str(final_path)
            job.subtitle_path = str(subtitle_path) if subtitle_path else None
            job.completed_at = utc_now()
            self.repository.save_job(job)
        except Exception as error:  # noqa: BLE001 - background job must persist failures
            project.status = "failed"
            for shot in project.shots:
                if shot.status == ShotStatus.RENDERING:
                    shot.status = ShotStatus.FAILED
            if active_shot is not None and active_started_monotonic is not None:
                active_shot.render_completed_at = utc_now()
                active_shot.render_duration_seconds = round(time.monotonic() - active_started_monotonic, 3)
            self.repository.save_project(project)
            job.status = "failed"
            job.current_service_id = None
            job.error = self._format_error(error)
            job.message = "render failed"
            job.completed_at = utc_now()
            self.repository.save_job(job)

    def active_project_ids(self) -> set[str]:
        active: set[str] = set()
        for job_id, task in self._tasks.items():
            if task.done():
                continue
            job = self.repository.get_job(job_id)
            if job:
                active.add(job.project_id)
        return active

    @staticmethod
    def _format_error(error: BaseException) -> str:
        """Persist a useful failure even when an exception has no message."""

        detail = str(error).strip() or repr(error)
        return f"{type(error).__name__}: {detail}"

    @staticmethod
    def _ultra_fast_scene_transitions(project: FilmProject) -> list[str] | None:
        brief = project.brief
        if (
            brief.continuation_mode != ContinuationMode.ULTRA_FAST
            or brief.ultra_fast_anchor_strategy != UltraFastAnchorStrategy.INDEPENDENT
        ):
            return None
        boundary_count = max(0, len(project.shots) - 1)
        if brief.ultra_fast_transition == UltraFastTransition.HARD_CUT:
            return ["hard_cut"] * boundary_count
        if brief.ultra_fast_transition == UltraFastTransition.FADE_BLACK:
            return ["fade_black"] * boundary_count
        if brief.ultra_fast_transition == UltraFastTransition.DISSOLVE:
            return ["dissolve"] * boundary_count
        rng = random.Random(f"{project.id}:ultra-fast-transitions")
        return [rng.choice(("fade_black", "dissolve", "fade")) for _ in range(boundary_count)]

    async def shutdown(self) -> None:
        active = [(job_id, task) for job_id, task in self._tasks.items() if not task.done()]
        for _, task in active:
            task.cancel()
        if active:
            await asyncio.gather(*(task for _, task in active), return_exceptions=True)
        for job_id, _ in active:
            job = self.repository.get_job(job_id)
            if not job or job.status not in {"queued", "running"}:
                continue
            job.status = "failed"
            job.current_service_id = None
            job.error = "Studio stopped before the render completed"
            job.message = "render interrupted"
            job.completed_at = utc_now()
            self.repository.save_job(job)
            project = self.repository.get_project(job.project_id)
            if project and project.status == "rendering":
                project.status = "failed"
                for shot in project.shots:
                    if shot.status == ShotStatus.RENDERING:
                        shot.status = ShotStatus.FAILED
                project.updated_at = utc_now()
                self.repository.save_project(project)

    @staticmethod
    def _write_sidecar_subtitles(project, output_dir: Path) -> Path | None:
        if project.brief.subtitle_mode != "sidecar":
            return None
        entries: list[str] = []
        elapsed = 0.0
        sequence = 1
        for shot in sorted(project.shots, key=lambda value: value.index):
            text = (shot.subtitle_text or "").strip()
            start = elapsed
            elapsed += shot.duration_seconds
            if not text:
                continue
            end = elapsed
            entries.extend(
                [
                    str(sequence),
                    f"{RenderManager._srt_time(start)} --> {RenderManager._srt_time(end)}",
                    text,
                    "",
                ]
            )
            sequence += 1
        if not entries:
            return None
        path = output_dir / "final.srt"
        path.write_text("\n".join(entries), encoding="utf-8")
        return path

    @staticmethod
    def _srt_time(seconds: float) -> str:
        millis = max(0, int(round(seconds * 1000)))
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds_value, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{millis:03d}"

    @staticmethod
    def _video_canvas(aspect_ratio: str) -> tuple[int, int]:
        # MiniMax-H3 rounds canvases to multiples of 32. These shapes preserve
        # roughly equal pixel area while honoring landscape/portrait/square.
        return {
            "16:9": (1280, 704),
            "9:16": (704, 1280),
            "1:1": (960, 960),
        }[aspect_ratio]

    @staticmethod
    def reusable_take_path(shot) -> Path | None:
        if not shot.selected_take_path or shot.status != ShotStatus.COMPLETE:
            return None
        path = Path(shot.selected_take_path)
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        return path

    @staticmethod
    def _clear_forced_render_state(project: FilmProject, output_dir: Path) -> None:
        """Make the explicit “again” action visibly and semantically fresh."""

        for shot in project.shots:
            shot.status = ShotStatus.PLANNED
            shot.selected_take_path = None
            shot.boundary_frame_path = None
            shot.anchor_frame_path = None
            shot.render_started_at = None
            shot.render_completed_at = None
            shot.render_duration_seconds = None
        for path in output_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.startswith("shot-") or path.name in {"final.mp4", "final.srt", "concat.txt"}:
                path.unlink()

    def _start_frame(self, shot, boundary_frames: dict[str, Path]) -> Path:
        # A creator-selected start frame is an explicit composition decision
        # and therefore wins even when the shot also carries continuity
        # metadata from an earlier storyboard plan.
        if shot.start_frame_asset_id:
            asset = self.repository.get_asset(shot.start_frame_asset_id)
            if asset:
                return Path(asset.resolved_path)
        if shot.continuity_from_shot_id and shot.continuity_from_shot_id in boundary_frames:
            return boundary_frames[shot.continuity_from_shot_id]
        for asset_id in shot.reference_asset_ids:
            asset = self.repository.get_asset(asset_id)
            if asset and asset.kind == AssetKind.IMAGE:
                return Path(asset.resolved_path)
        raise RuntimeError(f"shot {shot.id} has no start frame")

    def _set_job_service(self, job: RenderJob | None, service_id: str | None) -> None:
        if job is None or job.current_service_id == service_id:
            return
        job.current_service_id = service_id
        job.updated_at = utc_now()
        self.repository.save_job(job)

    async def _maybe_make_anchor(
        self,
        project,
        shot,
        position: int,
        boundary_frames: dict[str, Path],
        output_dir: Path,
        *,
        job: RenderJob | None = None,
    ) -> Path | None:
        """Build an FL2VA anchor through Image Edit or zero-reference T2I."""

        mode = self.settings.image_edit_anchor_mode
        if mode not in IMAGE_EDIT_ANCHOR_MODES:
            raise RuntimeError(f"unsupported STUDIO_IMAGE_EDIT_ANCHOR_MODE: {mode}")
        is_ultra_independent = uses_independent_ultra_fast_anchor(project, shot)
        if (
            shot.continuity_from_shot_id
            and not shot.start_frame_asset_id
            and resolved_continuation_mode(project, shot) == ContinuationMode.ULTRA_FAST
            and not is_ultra_independent
        ):
            # Preserve the legacy boundary-frame -> FL2VA strategy when the
            # creator explicitly selects it.
            return None
        if not is_ultra_independent and not anchor_selected(shot, position, mode):
            return None
        if not shot.anchor_prompt:
            raise RuntimeError(f"shot {shot.index + 1} requires a planner-authored anchor_prompt")

        references = self._anchor_references(
            shot,
            boundary_frames,
            # In independent ultra-fast mode the previous boundary is edited
            # into a new composition; it is not used directly as FL2VA input.
            include_boundary=True,
        )
        anchor_path = output_dir / f"shot-{position + 1:03d}-anchor.png"
        width = {"16:9": 1280, "9:16": 720, "1:1": 1024}[project.brief.aspect_ratio]
        height = {"16:9": 720, "9:16": 1280, "1:1": 1024}[project.brief.aspect_ratio]
        if references:
            if self.image_edit_provider_error:
                raise RuntimeError(self.image_edit_provider_error)
            provider = self.image_edit_provider
            if provider is None or not provider.configured:
                raise RuntimeError("Image Edit anchor requested but provider is not configured")
            self._set_job_service(job, "image_edit")
            await provider.edit(
                ImageEditRequest(
                    # The planner owns the complete direct-to-Qwen prompt. Keep
                    # adapters transport-only so a second template cannot overflow
                    # the model's 1024-token input limit or alter creative intent.
                    prompt=shot.anchor_prompt,
                    references=tuple(references),
                    output_path=anchor_path,
                    width=width,
                    height=height,
                    negative_prompt=shot.negative_prompt or None,
                    extra_body={
                        "num_inference_steps": self.settings.image_edit_steps,
                        "true_cfg_scale": self.settings.image_edit_true_cfg_scale,
                        "guidance_scale": self.settings.image_edit_guidance_scale,
                    },
                )
            )
        else:
            if self.text_to_image_provider_error:
                raise RuntimeError(self.text_to_image_provider_error)
            provider = self.text_to_image_provider
            if provider is None or not provider.configured:
                raise RuntimeError("T2I anchor requested but provider is not configured; set STUDIO_T2I_BASE_URL")
            self._set_job_service(job, "t2i")
            await provider.generate(
                TextToImageRequest(
                    prompt=shot.anchor_prompt,
                    negative_prompt=shot.negative_prompt,
                    output_path=anchor_path,
                    width=width,
                    height=height,
                    seed=shot.seed,
                )
            )
        shot.anchor_frame_path = str(anchor_path)
        self.repository.save_project(project)
        return anchor_path

    def _anchor_references(
        self,
        shot,
        boundary_frames: dict[str, Path],
        *,
        include_boundary: bool = True,
    ) -> list[ImageEditReference]:
        references: list[ImageEditReference] = []
        seen: set[Path] = set()

        def add(path: Path, label: str, role: str, tags: tuple[str, ...] = (), caption: str | None = None):
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            references.append(ImageEditReference(path, label, role, tags, caption))

        if include_boundary and shot.continuity_from_shot_id and shot.continuity_from_shot_id in boundary_frames:
            add(boundary_frames[shot.continuity_from_shot_id], "previous shot boundary", "continuity")
        if shot.start_frame_asset_id:
            asset = self.repository.get_asset(shot.start_frame_asset_id)
            if asset and asset.kind == AssetKind.IMAGE:
                add(
                    Path(asset.resolved_path),
                    asset.display_name or asset.caption or Path(asset.original_name).stem,
                    "start_frame",
                    tuple(asset.tags),
                    asset.caption,
                )
        for asset_id in shot.reference_asset_ids:
            asset = self.repository.get_asset(asset_id)
            if not asset or asset.kind != AssetKind.IMAGE:
                continue
            role = next(
                (
                    getattr(value, "value", value)
                    for value in asset.roles
                    if getattr(value, "value", value) != "reference"
                ),
                "reference",
            )
            add(
                Path(asset.resolved_path),
                asset.display_name or asset.caption or Path(asset.original_name).stem,
                role,
                tuple(asset.tags),
                asset.caption,
            )
        return references

    def _ref2va_inputs(self, shot) -> tuple[Path, Path]:
        image: Path | None = None
        media: Path | None = None
        for asset_id in shot.reference_asset_ids:
            asset = self.repository.get_asset(asset_id)
            if not asset:
                continue
            if asset.kind == AssetKind.IMAGE and not image:
                image = Path(asset.resolved_path)
            elif asset.kind in {AssetKind.AUDIO, AssetKind.VIDEO} and not media:
                media = Path(asset.resolved_path)
        if shot.audio_asset_id:
            asset = self.repository.get_asset(shot.audio_asset_id)
            if asset:
                media = Path(asset.resolved_path)
        if not image or not media:
            raise RuntimeError(f"shot {shot.id} requires image plus audio/video references")
        return image, media

    async def _continuation_ref2va_inputs(
        self,
        project: FilmProject,
        shot: ShotSpec,
        position: int,
        rendered_by_shot: dict[str, Path],
        boundary_frames: dict[str, Path],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        source_id = shot.continuity_from_shot_id
        if not source_id:
            raise RuntimeError(f"shot {shot.id} has no continuation source")
        source_video = rendered_by_shot.get(source_id)
        source_boundary = boundary_frames.get(source_id)
        if not source_video or not source_boundary:
            raise RuntimeError(f"shot {shot.id} continuation source {source_id} is not rendered")

        mode = resolved_continuation_mode(project, shot)
        if mode == ContinuationMode.QUALITY:
            reference_path = output_dir / f"shot-{position + 1:03d}-continuation-ref2va-max15s.mp4"
            normalized = await asyncio.to_thread(
                self.media.normalize_ref2va_video,
                source_video,
                reference_path,
            )
            return source_boundary, normalized

        tail_path = output_dir / (f"shot-{position + 1:03d}-continuation-tail-{self.CONTINUATION_TAIL_SECONDS:g}s.mp4")
        await asyncio.to_thread(
            self.media.extract_tail,
            source_video,
            tail_path,
            self.CONTINUATION_TAIL_SECONDS,
        )
        return source_boundary, tail_path

    @classmethod
    def _with_continuation_rule(cls, shot: ShotSpec) -> ShotSpec:
        """Build an ephemeral Ref2VA request without mutating storyboard text."""

        prompt = shot.prompt.rstrip()
        if cls.CONTINUATION_REF2VA_RULE not in prompt:
            prompt = f"{prompt}\n\nCONTINUATION CONSTRAINT:\n{cls.CONTINUATION_REF2VA_RULE}"
        return shot.model_copy(deep=True, update={"prompt": prompt})
