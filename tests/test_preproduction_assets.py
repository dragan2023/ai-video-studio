import asyncio

from test_assets import png_bytes

from long_video_studio.adapters.text_to_image import TextToImageRequest
from long_video_studio.assets import AssetService
from long_video_studio.domain import FilmProject, PreproductionStatus, ProjectBrief, ShotSpec, WorldBible
from long_video_studio.preproduction import PreproductionPlanner
from long_video_studio.preproduction_assets import PreproductionAssetGenerator
from long_video_studio.repository import StudioRepository


class FakeT2I:
    configured = True

    async def generate(self, request: TextToImageRequest):
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(png_bytes().read())
        return request.output_path


def test_approved_gap_generation_binds_asset_and_makes_plan_ready(settings, monkeypatch):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    project = FilmProject(
        brief=ProjectBrief(prompt="A river at night."),
        world_bible=WorldBible(logline="River", visual_style="cinematic"),
        shots=[ShotSpec(index=0, title="River", purpose="Set place", duration_seconds=5, prompt="A river at night.", source_section="剪辑与动作：硬切至河面。")],
    )
    plan = PreproductionPlanner().plan(project, [])
    project.preproduction_plan = plan.model_copy(update={"status": PreproductionStatus.GENERATING_ASSETS})
    monkeypatch.setattr("long_video_studio.preproduction_assets.text_to_image_provider_from_settings", lambda _settings: FakeT2I())

    result = asyncio.run(PreproductionAssetGenerator(settings, repository, assets).generate_approved_gaps(project))
    assert result.preproduction_plan.status == PreproductionStatus.READY
    assert result.preproduction_plan.generated_image_count == 0
    assert result.shots[0].start_frame_asset_id
    assert repository.get_asset(result.shots[0].start_frame_asset_id)


def test_plan_execution_bindings_materialize_black_and_continuity(settings):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    project = FilmProject(
        brief=ProjectBrief(prompt="A black title becomes a continuing image."),
        world_bible=WorldBible(logline="Title", visual_style="cinematic"),
        shots=[
            ShotSpec(index=0, title="Black", purpose="Title", duration_seconds=5, prompt="Black title.", source_section="剪辑与动作：本镜黑场字幕镜。"),
            ShotSpec(index=1, title="Continue", purpose="Continue", duration_seconds=5, prompt="The image continues.", source_section="剪辑与动作：承接上一镜动作余势，保持空间轴线。"),
        ],
    )
    project.preproduction_plan = PreproductionPlanner().plan(project, [])
    bound = PreproductionAssetGenerator(settings, repository, assets).apply_execution_bindings(project)
    assert bound.shots[0].start_frame_asset_id
    black = repository.get_asset(bound.shots[0].start_frame_asset_id)
    assert black and assets.resolve_content_path(black).read_bytes().startswith(b"\x89PNG")
    assert bound.shots[1].continuity_from_shot_id == bound.shots[0].id
