import asyncio
from dataclasses import replace
from pathlib import Path

from test_assets import png_bytes

from long_video_studio.adapters.image_edit import (
    ImageEditCapabilities,
    ImageEditRequest,
)
from long_video_studio.adapters.text_to_image import TextToImageRequest
from long_video_studio.assets import AssetService
from long_video_studio.domain import (
    AssetRole,
    AssetUpdate,
    ContinuationMode,
    FilmProject,
    ProjectBrief,
    ShotSpec,
    ShotTask,
    WorldBible,
)
from long_video_studio.repository import StudioRepository
from long_video_studio.runner import RenderManager


class FakeImageEditProvider:
    def __init__(self):
        self.request: ImageEditRequest | None = None

    @property
    def configured(self):
        return True

    @property
    def capabilities(self):
        return ImageEditCapabilities("fake", "test", True, 4)

    async def edit(self, request: ImageEditRequest):
        self.request = request
        request.output_path.write_bytes(png_bytes().getvalue())
        return request.output_path


class FakeTextToImageProvider:
    def __init__(self):
        self.request: TextToImageRequest | None = None

    @property
    def configured(self):
        return True

    async def generate(self, request: TextToImageRequest):
        self.request = request
        request.output_path.write_bytes(png_bytes().getvalue())
        return request.output_path


def test_missing_first_frame_generates_anchor_from_ordered_scene_and_character_references(settings):
    configured = replace(settings, image_edit_anchor_mode="first-shot")
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    scene = assets.ingest_stream(
        png_bytes(),
        "scene.png",
        "image/png",
        roles=[AssetRole.LOCATION],
        tags=["old town"],
    )
    character = assets.ingest_stream(
        png_bytes("royalblue"),
        "hero.png",
        "image/png",
        roles=[AssetRole.CHARACTER],
        tags=["red coat"],
    )
    scene = assets.update(scene.id, AssetUpdate(display_name="Old Town"))
    character = assets.update(character.id, AssetUpdate(display_name="Bai Lu"))
    shot = ShotSpec(
        index=0,
        title="Arrival",
        purpose="Introduce the hero",
        prompt="The hero steps into the rainy old town.",
        anchor_prompt=(
            "A single cinematic opening still in the rainy old town: Bai Lu stands in the foreground, "
            "wearing the referenced red coat, with the Old Town architecture clearly visible behind her."
        ),
        negative_prompt="text, logo, watermark",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        reference_asset_ids=[scene.id, character.id],
    )
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A rainy arrival."),
            world_bible=WorldBible(logline="Arrival", visual_style="cinematic"),
            shots=[shot],
        )
    )
    provider = FakeImageEditProvider()
    manager = RenderManager(configured, repository)
    manager.image_edit_provider = provider
    output_dir = configured.output_dir / project.id
    output_dir.mkdir(parents=True)

    anchor = asyncio.run(manager._maybe_make_anchor(project, shot, 0, {}, output_dir))

    assert anchor == output_dir / "shot-001-anchor.png"
    assert provider.request is not None
    assert [reference.role for reference in provider.request.references] == [
        "location",
        "character",
    ]
    assert [reference.label for reference in provider.request.references] == [
        "Old Town",
        "Bai Lu",
    ]
    assert "red coat" in provider.request.references[1].tags
    assert provider.request.prompt == shot.anchor_prompt
    assert (provider.request.width, provider.request.height) == (1280, 720)
    assert provider.request.negative_prompt == "text, logo, watermark"
    assert provider.request.extra_body == {
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
    }
    persisted = repository.get_project(project.id)
    assert persisted.shots[0].anchor_frame_path == str(anchor)
    assert persisted.shots[0].anchor_prompt == shot.anchor_prompt


def test_missing_planner_anchor_fails_closed(settings):
    configured = replace(settings, image_edit_anchor_mode="first-shot")
    repository = StudioRepository(configured.database_path)
    shot = ShotSpec(
        index=0,
        title="Missing anchor",
        purpose="Prove validation",
        prompt="The actor enters the room.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
    )
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="An entrance."),
            world_bible=WorldBible(logline="Entrance", visual_style="cinematic"),
            shots=[shot],
        )
    )
    manager = RenderManager(configured, repository)
    manager.image_edit_provider = FakeImageEditProvider()
    output_dir = configured.output_dir / project.id
    output_dir.mkdir(parents=True)

    try:
        asyncio.run(manager._maybe_make_anchor(project, shot, 0, {}, output_dir))
    except RuntimeError as exc:
        assert "planner-authored anchor_prompt" in str(exc)
    else:
        raise AssertionError("missing planner anchor must fail closed")


def test_zero_material_anchor_uses_text_to_image_only(settings):
    configured = replace(settings, image_edit_anchor_mode="first-shot")
    repository = StudioRepository(configured.database_path)
    shot = ShotSpec(
        index=0,
        title="New world",
        purpose="Establish a completely new setting",
        prompt="A creator enters a sunlit empty studio.",
        anchor_prompt="A 16:9 cinematic opening still of a creator entering a sunlit empty studio.",
        negative_prompt="text, logo, watermark",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        seed=23,
        reference_asset_ids=[],
    )
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A new studio."),
            world_bible=WorldBible(logline="Arrival", visual_style="cinematic"),
            shots=[shot],
        )
    )
    image_edit = FakeImageEditProvider()
    text_to_image = FakeTextToImageProvider()
    manager = RenderManager(configured, repository)
    manager.image_edit_provider = image_edit
    manager.text_to_image_provider = text_to_image
    output_dir = configured.output_dir / project.id
    output_dir.mkdir(parents=True)

    anchor = asyncio.run(manager._maybe_make_anchor(project, shot, 0, {}, output_dir))

    assert anchor == output_dir / "shot-001-anchor.png"
    assert image_edit.request is None
    assert text_to_image.request is not None
    assert text_to_image.request.prompt == shot.anchor_prompt
    assert text_to_image.request.negative_prompt == shot.negative_prompt
    assert text_to_image.request.seed == 23
    assert (text_to_image.request.width, text_to_image.request.height) == (1280, 720)


def test_ultra_fast_edits_previous_boundary_into_a_fresh_anchor(settings, tmp_path):
    configured = replace(settings, image_edit_anchor_mode="first-shot")
    repository = StudioRepository(configured.database_path)
    first = ShotSpec(
        index=0,
        title="Scene one",
        purpose="Open the story",
        prompt="A detective enters a rain-soaked alley.",
        anchor_prompt="A detective at the entrance to a rain-soaked alley, cinematic 16:9.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
    )
    second = ShotSpec(
        index=1,
        title="Scene two",
        purpose="Reveal the suspect",
        prompt="Inside a bright cafe, the suspect opens a sealed letter.",
        anchor_prompt="A bright cafe interior; the suspect opens a sealed letter, cinematic 16:9.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        continuity_from_shot_id=first.id,
    )
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(
                prompt="A detective short drama.",
                duration_seconds=15,
                continuation_mode=ContinuationMode.ULTRA_FAST,
            ),
            world_bible=WorldBible(logline="A sealed letter", visual_style="cinematic"),
            shots=[first, second],
        )
    )
    previous_boundary = tmp_path / "previous-boundary.png"
    previous_boundary.write_bytes(png_bytes("green").getvalue())
    image_edit = FakeImageEditProvider()
    text_to_image = FakeTextToImageProvider()
    manager = RenderManager(configured, repository)
    manager.image_edit_provider = image_edit
    manager.text_to_image_provider = text_to_image
    output_dir = configured.output_dir / project.id
    output_dir.mkdir(parents=True)

    anchor = asyncio.run(
        manager._maybe_make_anchor(
            project,
            second,
            1,
            {first.id: previous_boundary},
            output_dir,
        )
    )

    assert anchor == output_dir / "shot-002-anchor.png"
    assert image_edit.request is not None
    assert [reference.path for reference in image_edit.request.references] == [
        previous_boundary
    ]
    assert image_edit.request.prompt == second.anchor_prompt
    assert text_to_image.request is None


def test_explicit_start_frame_bypasses_image_edit(settings):
    configured = replace(settings, image_edit_anchor_mode="scene-cuts")
    repository = StudioRepository(configured.database_path)
    assets = AssetService(configured, repository)
    start = assets.ingest_stream(
        png_bytes(),
        "creator-start.png",
        "image/png",
        roles=[AssetRole.START_FRAME],
    )
    shot = ShotSpec(
        index=0,
        title="Creator opening",
        purpose="Honor the selected frame",
        prompt="Continue naturally from the exact creator frame.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        start_frame_asset_id=start.id,
        reference_asset_ids=[start.id],
    )
    project = repository.save_project(
        FilmProject(
            brief=ProjectBrief(prompt="A creator-controlled opening."),
            world_bible=WorldBible(logline="Opening", visual_style="cinematic"),
            shots=[shot],
        )
    )
    provider = FakeImageEditProvider()
    manager = RenderManager(configured, repository)
    manager.image_edit_provider = provider
    output_dir = configured.output_dir / project.id
    output_dir.mkdir(parents=True)

    anchor = asyncio.run(manager._maybe_make_anchor(project, shot, 0, {}, output_dir))

    assert anchor is None
    assert provider.request is None


def test_explicit_start_frame_wins_over_previous_boundary(settings, tmp_path):
    repository = StudioRepository(settings.database_path)
    assets = AssetService(settings, repository)
    start = assets.ingest_stream(
        png_bytes(),
        "creator-start.png",
        "image/png",
        roles=[AssetRole.START_FRAME],
    )
    previous_boundary = tmp_path / "previous-boundary.png"
    previous_boundary.write_bytes(png_bytes("royalblue").getvalue())
    shot = ShotSpec(
        index=1,
        title="Creator override",
        purpose="Honor the selected frame",
        prompt="Begin from the exact creator-selected composition.",
        duration_seconds=4,
        task=ShotTask.FL2VA,
        start_frame_asset_id=start.id,
        reference_asset_ids=[start.id],
        continuity_from_shot_id="shot_previous",
    )
    manager = RenderManager(settings, repository)

    selected = manager._start_frame(shot, {"shot_previous": previous_boundary})

    assert selected == Path(start.resolved_path)
