from __future__ import annotations

import asyncio
from dataclasses import replace

from long_video_studio.domain import FilmProject, ProjectBrief, WorldBible
from long_video_studio.repository import StudioRepository
from long_video_studio.runner import RenderManager


def test_render_manager_runs_two_projects_and_queues_the_third(settings, monkeypatch):
    async def scenario() -> None:
        configured = replace(settings, render_max_concurrency=2)
        repository = StudioRepository(configured.database_path)
        manager = RenderManager(configured, repository)
        projects = [
            repository.save_project(
                FilmProject(
                    brief=ProjectBrief(prompt=f"Concurrent film {index}"),
                    world_bible=WorldBible(logline=f"Film {index}", visual_style="Natural"),
                    shots=[],
                )
            )
            for index in range(3)
        ]
        started: list[str] = []
        two_started = asyncio.Event()
        release = asyncio.Event()

        async def controlled_run(job_id: str) -> None:
            job = repository.get_job(job_id)
            assert job is not None
            started.append(job.project_id)
            if len(started) == 2:
                two_started.set()
            await release.wait()

        monkeypatch.setattr(manager, "_run_with_slot", controlled_run)
        for project in projects:
            manager.submit(project.id)

        await asyncio.wait_for(two_started.wait(), timeout=1)
        await asyncio.sleep(0.02)
        assert set(started) == {projects[0].id, projects[1].id}

        release.set()
        await asyncio.gather(*list(manager._tasks.values()))
        assert set(started) == {project.id for project in projects}

    asyncio.run(scenario())
