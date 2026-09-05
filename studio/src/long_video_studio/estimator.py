from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import median

from long_video_studio.config import Settings
from long_video_studio.domain import (
    ContinuationMode,
    FilmProject,
    ProjectRenderEstimate,
    RenderObservation,
    ShotRenderEstimate,
    ShotSpec,
    ShotStatus,
    ShotTask,
    effective_video_task,
    resolved_continuation_mode,
    uses_independent_ultra_fast_anchor,
    utc_now,
)
from long_video_studio.repository import StudioRepository


class RenderEstimator:
    """Profile-aware ETA calibration backed by append-only successful timings."""

    def __init__(self, settings: Settings, repository: StudioRepository):
        self.settings = settings
        self.repository = repository

    def backfill(self) -> int:
        """Preserve legacy shot telemetry before projects can be deleted."""

        count = 0
        for project in self.repository.list_projects():
            for shot in project.shots:
                if self.observe(project, shot):
                    count += 1
        return count

    def observe(self, project: FilmProject, shot: ShotSpec) -> bool:
        if (
            shot.status != ShotStatus.COMPLETE
            or shot.render_duration_seconds is None
            or shot.render_completed_at is None
            or shot.render_duration_seconds <= 0
        ):
            return False
        task = self._runtime_task(project, shot)
        mode = self._mode(project, shot)
        completed = shot.render_completed_at.isoformat()
        observation = RenderObservation(
            source_key=f"{project.id}:{shot.id}:{completed}",
            project_id=project.id,
            shot_id=shot.id,
            render_profile=self.settings.render_profile,
            task=task,
            continuation_mode=mode,
            aspect_ratio=project.brief.aspect_ratio,
            duration_seconds=shot.duration_seconds,
            inference_steps=shot.inference_steps,
            elapsed_seconds=shot.render_duration_seconds,
            created_at=shot.render_completed_at,
        )
        before = {item.source_key for item in self.repository.list_render_observations(self.settings.render_profile)}
        self.repository.save_render_observation(observation)
        return observation.source_key not in before

    def estimate_project(
        self,
        project: FilmProject,
        *,
        now: datetime | None = None,
        include_completed: bool = False,
    ) -> ProjectRenderEstimate:
        now = now or utc_now()
        estimates = [
            self.estimate_shot(
                project,
                shot,
                now=now,
                include_completed=include_completed,
            )
            for shot in sorted(project.shots, key=lambda item: item.index)
        ]
        assembly_seconds = max(12.0, 2.0 * len(project.shots)) * self.settings.render_estimate_scale
        total_seconds = sum(item.estimated_seconds for item in estimates) + assembly_seconds
        remaining_seconds = sum(item.remaining_seconds for item in estimates)
        if include_completed or remaining_seconds > 0 or not self._final_output_exists(project):
            remaining_seconds += assembly_seconds
        sources = {item.source for item in estimates}
        source = next(iter(sources)) if len(sources) == 1 else "mixed"
        if not sources:
            source = "configured"
        return ProjectRenderEstimate(
            project_id=project.id,
            total_seconds=round(total_seconds, 3),
            remaining_seconds=round(max(0.0, remaining_seconds), 3),
            assembly_seconds=round(assembly_seconds, 3),
            sample_count=max((item.sample_count for item in estimates), default=0),
            source=source,
            shots=estimates,
            generated_at=now,
        )

    def estimate_shot(
        self,
        project: FilmProject,
        shot: ShotSpec,
        *,
        now: datetime | None = None,
        include_completed: bool = False,
    ) -> ShotRenderEstimate:
        task = self._runtime_task(project, shot)
        mode = self._mode(project, shot)
        samples = self._samples(task, mode, project.brief.aspect_ratio)
        if samples:
            baseline = self._robust_median(samples)
            source = "history"
        else:
            baseline = (
                self.settings.render_ref2va_baseline_seconds
                if task == ShotTask.REF2VA
                else self.settings.render_fl2va_baseline_seconds
            )
            source = "configured"
        estimated = (
            baseline
            * (shot.duration_seconds / 10.0)
            * (shot.inference_steps / 50.0)
            * self.settings.render_estimate_scale
        )
        remaining = estimated
        if not include_completed and self._shot_output_exists(shot):
            remaining = 0.0
        elif not include_completed and shot.status == ShotStatus.RENDERING and shot.render_started_at:
            elapsed = max(0.0, ((now or utc_now()) - shot.render_started_at).total_seconds())
            remaining = max(0.0, estimated - elapsed)
        sample_count = len(samples)
        confidence = "high" if sample_count >= 5 else "medium" if sample_count >= 2 else "low"
        return ShotRenderEstimate(
            shot_id=shot.id,
            task=task,
            continuation_mode=mode,
            estimated_seconds=round(max(0.0, estimated), 3),
            remaining_seconds=round(max(0.0, remaining), 3),
            sample_count=sample_count,
            source=source,
            confidence=confidence,
        )

    def _samples(self, task: ShotTask, mode: str, aspect_ratio: str) -> list[float]:
        observations = [
            item
            for item in self.repository.list_render_observations(self.settings.render_profile)
            if item.task == task and item.elapsed_seconds > 0
        ]
        buckets = (
            [item for item in observations if item.continuation_mode == mode and item.aspect_ratio == aspect_ratio],
            [item for item in observations if item.continuation_mode == mode],
            [item for item in observations if item.aspect_ratio == aspect_ratio],
            observations,
        )
        selected = next((bucket for bucket in buckets if bucket), [])
        return [
            item.elapsed_seconds / (item.duration_seconds / 10.0) / (item.inference_steps / 50.0) for item in selected
        ]

    @staticmethod
    def _robust_median(samples: list[float]) -> float:
        center = median(samples)
        if len(samples) < 5:
            return center
        deviations = [abs(value - center) for value in samples]
        mad = median(deviations)
        if mad == 0:
            return center
        trimmed = [value for value in samples if abs(value - center) <= 3.5 * mad]
        return median(trimmed or samples)

    def _runtime_task(self, project: FilmProject, shot: ShotSpec) -> ShotTask:
        resolved_mode = resolved_continuation_mode(project, shot)
        continuation_mode = (
            resolved_mode
            if not shot.start_frame_asset_id
            and (shot.continuity_from_shot_id or resolved_mode == ContinuationMode.ULTRA_FAST)
            else None
        )
        return effective_video_task(
            shot,
            ref2va_configured=self.settings.h3_configured("ref2va"),
            fl2va_configured=self.settings.h3_configured("fl2va"),
            continuation_mode=continuation_mode,
        )

    @staticmethod
    def _mode(project: FilmProject, shot: ShotSpec) -> str:
        if uses_independent_ultra_fast_anchor(project, shot):
            return "ultra_fast"
        if not shot.continuity_from_shot_id:
            return "initial"
        return resolved_continuation_mode(project, shot).value

    @staticmethod
    def _shot_output_exists(shot: ShotSpec) -> bool:
        return bool(
            shot.status == ShotStatus.COMPLETE and shot.selected_take_path and Path(shot.selected_take_path).is_file()
        )

    def _final_output_exists(self, project: FilmProject) -> bool:
        return (self.settings.output_dir / project.id / "final.mp4").is_file()
