from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace

from long_video_studio.config import Settings
from long_video_studio.domain import (
    BatchPlanningRun,
    BatchPlanningStatus,
    FilmProject,
    PlannerTraceEvent,
    ProjectBrief,
    WorldBible,
    utc_now,
)
from long_video_studio.planner import PlannerError, PlannerService
from long_video_studio.repository import StudioRepository

logger = logging.getLogger(__name__)


class PlanningManager:
    """Runs independent project planners without tying them to one browser request."""

    def __init__(
        self,
        settings: Settings,
        repository: StudioRepository,
        planner: PlannerService,
        planner_factory: Callable[..., PlannerService] | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.planner = planner
        self._planner_factory = planner_factory
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(settings.planner_project_concurrency)
        self._recover_interrupted_projects()

    def _current_planner(self) -> PlannerService:
        if self._planner_factory is not None:
            return self._planner_factory()
        return self.planner

    def _resolve_planner(self, *, profile_id: str, model: str = "") -> PlannerService:
        """Resolve a planner for a named client, honoring the app-wide factory."""
        if self._planner_factory is not None:
            return self._planner_factory(profile_id=profile_id, model=model)
        profile = self.settings.planner_profile(profile_id)
        if not profile.public()["available"]:
            raise ValueError(f"Planner profile is not configured: {profile_id}")
        return PlannerService(
            replace(
                self.settings,
                planner_base_url=profile.base_url,
                planner_api_key=profile.api_key,
                planner_model=model or profile.model,
                planner_wire_api=profile.wire_api,
            ),
            self.repository,
        )

    async def start(self, brief: ProjectBrief) -> FilmProject:
        draft = FilmProject(
            brief=brief,
            world_bible=WorldBible(
                logline=brief.title or brief.prompt,
                visual_style=brief.style,
            ),
            shots=[],
            status="planning",
        )
        self.repository.save_project(draft)
        task = asyncio.create_task(
            self._run(draft.id, brief),
            name=f"studio-plan-{draft.id}",
        )
        self._tasks[draft.id] = task
        task.add_done_callback(lambda completed, project_id=draft.id: self._discard(project_id, completed))
        return draft

    async def start_imported_h3(
        self, project_id: str, *, profile_id: str = "default", model: str = "", retry: bool = False,
    ) -> FilmProject:
        project = self.repository.get_project(project_id)
        if not project:
            raise KeyError(project_id)
        if self.active(project_id):
            return project
        # Validate the client resolves before we lock a batch run in; this
        # accepts env profiles and persisted LLM clients through the factory.
        try:
            preview = self._resolve_planner(profile_id=profile_id, model=model)
        except (ValueError, PlannerError) as error:
            raise ValueError(f"Planner profile is not configured: {profile_id}") from error
        resolved_model = model or preview.settings.planner_model or ""
        prior = project.batch_planning_run
        if prior and prior.status == BatchPlanningStatus.COMPLETE and not retry:
            return project
        run = BatchPlanningRun(
            status=BatchPlanningStatus.RUNNING,
            profile_id=profile_id,
            model=resolved_model,
            batch_size=prior.batch_size if prior else 6,
            completed_shot_ids=[] if retry else (prior.completed_shot_ids if prior else []),
            failed_shot_ids=[],
            started_at=utc_now(),
        )
        project.batch_planning_run = run
        self.repository.save_project(project)
        task = asyncio.create_task(self._run_imported_h3(project_id), name=f"studio-imported-h3-{project_id}")
        self._tasks[project_id] = task
        task.add_done_callback(lambda completed, item=project_id: self._discard(item, completed))
        return project

    async def _run_imported_h3(self, project_id: str) -> None:
        try:
            async with self._semaphore:
                project = self.repository.get_project(project_id)
                if not project or not project.batch_planning_run:
                    return
                run = project.batch_planning_run
                completed = set(run.completed_shot_ids)
                pending = [
                    shot for shot in sorted(project.shots, key=lambda item: item.index) if shot.id not in completed
                ]
                if not pending:
                    run.status = BatchPlanningStatus.COMPLETE
                    run.current_batch_start = len(project.shots)
                    run.last_error = ""
                    run.updated_at = utc_now()
                    self.repository.save_project(project)
                    return
                batch = pending[:run.batch_size]
                run.status = BatchPlanningStatus.RUNNING
                run.current_batch_start = batch[0].index
                run.last_error = ""
                run.updated_at = utc_now()
                self.repository.save_project(project)
                planner = self._resolve_planner(profile_id=run.profile_id, model=run.model)

                async def on_shot_progress(shot_id: str) -> None:
                    # Persist each completed shot so the UI shows X/70 during a
                    # batch rather than staying at 0/70 until the whole batch
                    # finishes (each shot can take ~90s on the LLM).
                    latest = self.repository.get_project(project_id)
                    if not latest or not latest.batch_planning_run:
                        return
                    batch_run = latest.batch_planning_run
                    if shot_id not in batch_run.completed_shot_ids:
                        batch_run.completed_shot_ids = [*batch_run.completed_shot_ids, shot_id]
                    batch_run.updated_at = utc_now()
                    self.repository.save_project(latest)

                enriched = await planner.enrich_imported_shots(
                    project, {shot.id for shot in batch}, on_progress=on_shot_progress,
                )
                project = self.repository.get_project(project_id) or project
                run = project.batch_planning_run
                project.shots = enriched.shots
                run.completed_shot_ids = list(dict.fromkeys([
                    *run.completed_shot_ids,
                    *(shot.id for shot in batch),
                ]))
                completed_after = set(run.completed_shot_ids)
                remaining = [
                    shot for shot in sorted(project.shots, key=lambda item: item.index)
                    if shot.id not in completed_after
                ]
                if remaining:
                    # One batch per invocation: persist the enriched shots then
                    # pause so the creator reviews this batch before generating
                    # the next one. The UI exposes a "续生成" action on pause.
                    run.status = BatchPlanningStatus.PAUSED
                    run.last_error = (
                        f"第 {len(run.completed_shot_ids) // run.batch_size} 批已完成；"
                        "请审阅本批分镜，确认后点击续生成。"
                    )
                else:
                    run.status = BatchPlanningStatus.COMPLETE
                    run.current_batch_start = len(project.shots)
                    run.last_error = ""
                run.updated_at = utc_now()
                self.repository.save_project(project)
        except asyncio.CancelledError:
            project = self.repository.get_project(project_id)
            if project and project.batch_planning_run:
                project.batch_planning_run.status = BatchPlanningStatus.PAUSED
                project.batch_planning_run.updated_at = utc_now()
                self.repository.save_project(project)
            raise
        except Exception as error:
            logger.exception("imported H3 batch failed: %s", project_id)
            project = self.repository.get_project(project_id)
            if project and project.batch_planning_run:
                project.batch_planning_run.status = BatchPlanningStatus.FAILED
                project.batch_planning_run.last_error = str(error)
                project.batch_planning_run.updated_at = utc_now()
                self.repository.save_project(project)

    def active(self, project_id: str) -> bool:
        task = self._tasks.get(project_id)
        return bool(task and not task.done())

    def active_project_ids(self) -> list[str]:
        return sorted(project_id for project_id, task in self._tasks.items() if not task.done())

    async def cancel(self, project_id: str) -> bool:
        task = self._tasks.get(project_id)
        if task is None or task.done():
            return False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def shutdown(self) -> None:
        active = [(project_id, task) for project_id, task in self._tasks.items() if not task.done()]
        tasks = [task for _, task in active]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for project_id, _ in active:
            project = self.repository.get_project(project_id)
            if project and project.status == "planning":
                self._mark_failed(project_id, "构思因 Studio 服务关闭而中断，请重新构思")

    async def _run(self, project_id: str, brief: ProjectBrief) -> None:
        try:
            async with self._semaphore:
                await self._current_planner().plan(brief, project_id=project_id)
        except asyncio.CancelledError:
            self._mark_failed(project_id, "构思已取消")
            raise
        except Exception as error:  # planner records the structured failure trace
            logger.exception("project planner failed: %s", project_id)
            self._mark_failed(project_id, str(error))

    def _mark_failed(self, project_id: str, error: str) -> None:
        project = self.repository.get_project(project_id)
        if not project:
            return
        project.status = "failed"
        project.updated_at = utc_now()
        if not project.planner_trace or project.planner_trace[-1].status != "failed":
            project.planner_trace.append(
                PlannerTraceEvent(
                    stage="planner",
                    status="failed",
                    message="planning task did not complete",
                    error=error,
                )
            )
        self.repository.save_project(project)
        logger.warning("project %s planning failed: %s", project_id, error)

    def _recover_interrupted_projects(self) -> None:
        for project in self.repository.list_projects():
            if project.batch_planning_run and project.batch_planning_run.status == BatchPlanningStatus.RUNNING:
                project.batch_planning_run.status = BatchPlanningStatus.PAUSED
                project.batch_planning_run.last_error = "Studio 服务重启；可从已完成镜头继续。"
                project.batch_planning_run.updated_at = utc_now()
                self.repository.save_project(project)
            if project.status == "planning":
                self._mark_failed(
                    project.id,
                    "构思任务因 Studio 服务重启而中断，请重新构思",
                )

    def _discard(self, project_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(project_id) is completed:
            self._tasks.pop(project_id, None)
