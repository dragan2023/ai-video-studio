from __future__ import annotations

import asyncio
from types import SimpleNamespace

from long_video_studio.domain import RenderJob, ShotSpec
from long_video_studio.runner import RenderManager


def test_h3_retry_persists_retry_count():
    saved = []
    manager = RenderManager.__new__(RenderManager)
    manager.settings = SimpleNamespace(render_retry_attempts=3, render_retry_backoff_seconds=0)
    manager.repository = SimpleNamespace(save_job=saved.append)
    job = RenderJob(project_id="project", max_retries=2)
    shot = ShotSpec(index=0, title="retry", purpose="test", duration_seconds=4, prompt="test")
    attempts = {"count": 0}

    async def operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = asyncio.run(manager._with_retries(operation, job, shot))
    assert result == "ok"
    assert attempts["count"] == 3
    assert job.retry_count == 2
    assert len(saved) == 2
