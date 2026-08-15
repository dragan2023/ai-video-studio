from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from long_video_studio.config import Settings
from long_video_studio.domain import FilmProject, PlannerTraceEvent, ProjectBrief, WorldBible, utc_now
from long_video_studio.planner import PlannerService
from long_video_studio.repository import StudioRepository

logger = logging.getLogger(__name__)


class PlanningManager:
    """Runs independent project planners without tying them to one browser request."""

    def __init__(
        self,
        settings: Settings,
        repository: StudioRepository,
        planner: PlannerService,
    ):
        self.repository = repository
        self.planner = planner
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(settings.planner_project_concurrency)
        self._recover_interrupted_projects()

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
                await self.planner.plan(brief, project_id=project_id)
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
            if project.status == "planning":
                self._mark_failed(
                    project.id,
                    "构思任务因 Studio 服务重启而中断，请重新构思",
                )

    def _discard(self, project_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(project_id) is completed:
            self._tasks.pop(project_id, None)
