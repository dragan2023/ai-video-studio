from __future__ import annotations

from dataclasses import dataclass

from long_video_studio.assets import AssetService
from long_video_studio.compiler import FilmCompiler
from long_video_studio.config import Settings
from long_video_studio.estimator import RenderEstimator
from long_video_studio.planner import PlannerService
from long_video_studio.repository import StudioRepository
from long_video_studio.service_status import ServiceStatusCollector


@dataclass
class StudioServices:
    settings: Settings
    repository: StudioRepository
    assets: AssetService
    planner: PlannerService
    compiler: FilmCompiler
    estimator: RenderEstimator
    service_status: ServiceStatusCollector

    @classmethod
    def create(cls, settings: Settings) -> StudioServices:
        settings.ensure_directories()
        repository = StudioRepository(settings.database_path)
        estimator = RenderEstimator(settings, repository)
        estimator.backfill()
        return cls(
            settings=settings,
            repository=repository,
            assets=AssetService(settings, repository),
            planner=PlannerService(settings, repository),
            compiler=FilmCompiler(settings, estimator=estimator, repository=repository),
            estimator=estimator,
            service_status=ServiceStatusCollector(settings),
        )
