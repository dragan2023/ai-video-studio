"""Controlled approved-only T2I asset generation for preproduction plans."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import struct
import zlib

from long_video_studio.adapters.text_to_image import TextToImageRequest, text_to_image_provider_from_settings
from long_video_studio.assets import AssetService
from long_video_studio.config import Settings
from long_video_studio.domain import AssetRole, FilmProject, PreproductionStatus, StartFrameSource
from long_video_studio.repository import StudioRepository


class PreproductionAssetGenerator:
    """Generate only user-approved generate_t2i start-frame gaps."""

    def __init__(self, settings: Settings, repository: StudioRepository, assets: AssetService) -> None:
        self.settings = settings
        self.repository = repository
        self.assets = assets

    def apply_execution_bindings(self, project: FilmProject) -> FilmProject:
        """Materialize free plan sources into existing Runner shot inputs."""
        plan = project.preproduction_plan
        if not plan:
            return project
        shots = {shot.id: shot for shot in project.shots}
        updated_rows = []
        for row in plan.shot_plans:
            shot = shots[row.shot_id]
            if row.start_frame_source == StartFrameSource.PREVIOUS_BOUNDARY and row.source_shot_id:
                shots[shot.id] = shot.model_copy(update={"continuity_from_shot_id": row.source_shot_id})
            elif row.start_frame_source == StartFrameSource.SYSTEM_BLACK:
                asset = self._system_black_asset()
                shots[shot.id] = shot.model_copy(update={"start_frame_asset_id": asset.id})
                row = row.model_copy(update={"selected_asset_id": asset.id})
            updated_rows.append(row)
        return project.model_copy(update={
            "shots": list(shots.values()),
            "preproduction_plan": plan.model_copy(update={"shot_plans": updated_rows}),
        })

    def _system_black_asset(self):
        pixels = b"".join(b"\x00" + (b"\x00\x00\x00" * self.settings.comfyui_width) for _ in range(self.settings.comfyui_height))
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.settings.comfyui_width, self.settings.comfyui_height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(pixels, 6)) + chunk(b"IEND", b"")
        return self.assets.ingest_stream(
            BytesIO(png), "system-black-start-frame.png", "image/png",
            tags=["system", "black-frame"], roles=[AssetRole.START_FRAME],
        )

    async def generate_approved_gaps(self, project: FilmProject) -> FilmProject:
        plan = project.preproduction_plan
        if not plan or plan.status != PreproductionStatus.GENERATING_ASSETS:
            raise ValueError("preproduction plan is not approved for asset generation")
        provider = text_to_image_provider_from_settings(self.settings)
        rows = [row for row in plan.shot_plans if row.start_frame_source == StartFrameSource.GENERATE_T2I]
        if rows and not provider.configured:
            return self._blocked(project, "T2I provider is not configured")
        try:
            generated_rows = []
            shots = {shot.id: shot for shot in project.shots}
            for row in rows:
                if not row.generation_permitted:
                    return self._blocked(project, f"shot {row.shot_index + 1} is not permitted for T2I")
                shot = shots[row.shot_id]
                output = self.settings.output_dir / project.id / "preproduction" / f"{shot.id}-start.png"
                image = await provider.generate(TextToImageRequest(
                    prompt=shot.anchor_prompt or shot.prompt, negative_prompt=shot.negative_prompt,
                    output_path=output, width=self.settings.comfyui_width, height=self.settings.comfyui_height, seed=shot.seed,
                ))
                with Path(image).open("rb") as stream:
                    asset = self.assets.ingest_stream(
                        stream, f"{shot.title}-generated-start.png", "image/png",
                        tags=["preproduction", "generated-start-frame"],
                        roles=[AssetRole.REFERENCE, AssetRole.START_FRAME],
                    )
                shots[shot.id] = shot.model_copy(update={
                    "start_frame_asset_id": asset.id,
                    "reference_asset_ids": list(dict.fromkeys([*shot.reference_asset_ids, asset.id])),
                })
                generated_rows.append(row.model_copy(update={
                    "start_frame_source": StartFrameSource.CREATOR_ASSET,
                    "selected_asset_id": asset.id,
                    "gap_reason": "generated from approved T2I request",
                    "generation_permitted": False,
                }))
            generated_by_id = {row.shot_id: row for row in generated_rows}
            final_rows = [generated_by_id.get(row.shot_id, row) for row in plan.shot_plans]
            final_plan = plan.model_copy(update={
                "status": PreproductionStatus.READY, "generated_image_count": 0,
                "blockers": [], "shot_plans": final_rows,
            })
            return project.model_copy(update={"shots": list(shots.values()), "preproduction_plan": final_plan})
        except Exception as error:
            return self._blocked(project, f"approved T2I generation failed: {type(error).__name__}: {error}")

    @staticmethod
    def _blocked(project: FilmProject, reason: str) -> FilmProject:
        assert project.preproduction_plan is not None
        return project.model_copy(update={
            "preproduction_plan": project.preproduction_plan.model_copy(
                update={"status": PreproductionStatus.BLOCKED, "blockers": [reason]}
            )
        })
