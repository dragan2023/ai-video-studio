from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from long_video_studio.adapters.h3 import H3Client
from long_video_studio.anchor_policy import anchor_selected
from long_video_studio.asset_importer import detect_missing_codes, import_code_assets, scan_asset_codes
from long_video_studio.domain import (
    MAX_PROJECT_DURATION_SECONDS,
    AssetKind,
    AssetRecord,
    AssetRole,
    AssetUpdate,
    AssetView,
    ContinuationMode,
    ContinuityState,
    DialogueLine,
    ExecutionPlan,
    FilmProject,
    LLMClient,
    PlannerTraceEvent,
    PreproductionPlan,
    PreproductionStatus,
    ProjectBrief,
    ProjectRenderEstimate,
    RenderJob,
    ShotSpec,
    ShotStatus,
    ShotTask,
    StoryboardBeat,
    SubjectCard,
    TransitionKind,
    UltraFastAnchorStrategy,
    UltraFastTransition,
    WorldBible,
    effective_video_task,
    resolved_continuation_mode,
    uses_independent_ultra_fast_anchor,
    utc_now,
)
from long_video_studio.h3_limits import H3_MAX_SHOT_SECONDS, H3_MIN_SHOT_SECONDS
from long_video_studio.plan_exporter import write_plan
from long_video_studio.planner import PlannerError
from long_video_studio.planning import PlanningManager
from long_video_studio.project_builder import build_film_project
from long_video_studio.runner import RenderManager
from long_video_studio.script_importer import parse_shot_script
from long_video_studio.services import StudioServices
from long_video_studio.style_registry import public_style_contracts


class ImportPathRequest(BaseModel):
    path: str
    recursive: bool = True
    copy_into_library: bool | None = None
    tags: list[str] = Field(default_factory=list)
    roles: list[AssetRole] = Field(default_factory=lambda: [AssetRole.REFERENCE])


class ThickScriptImportView(BaseModel):
    project: FilmProject
    missing_codes: list[str] = Field(default_factory=list)
    imported_codes: list[str] = Field(default_factory=list)
    plan_path: str


class H3BatchPlanningRequest(BaseModel):
    profile_id: str = "default"
    model: str = Field(default="", max_length=200)
    retry: bool = False


class PreproductionRequest(BaseModel):
    profile_id: str = Field(default="", max_length=64)
    model: str = Field(default="", max_length=200)


class ActivePlannerRequest(BaseModel):
    profile_id: str = Field(default="default", max_length=64)
    model: str = Field(default="", max_length=200)


class LLMClientCreate(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(default="", max_length=120)
    base_url: str = Field(default="", max_length=400)
    api_key: str | None = Field(default=None, max_length=500)
    model: str = Field(default="", max_length=200)
    wire_api: str = Field(default="chat_completions", max_length=40)


class LLMClientUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    base_url: str | None = Field(default=None, max_length=400)
    api_key: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    wire_api: str | None = Field(default=None, max_length=40)


class DefaultLLMClientRequest(BaseModel):
    model: str = Field(default="", max_length=200)


class ShotUpdate(BaseModel):
    title: str | None = None
    purpose: str | None = None
    duration_seconds: float | None = Field(
        default=None,
        ge=H3_MIN_SHOT_SECONDS,
        le=H3_MAX_SHOT_SECONDS,
    )
    task: ShotTask | None = None
    transition_kind: TransitionKind | None = None
    prompt: str | None = None
    anchor_prompt: str | None = None
    audio_prompt: str | None = None
    music_prompt: str | None = None
    dialogue: list[DialogueLine] | None = None
    opening_state: str | None = None
    ending_state: str | None = None
    continuity_handoff: str | None = None
    reference_anchors: list[str] | None = None
    hook: str | None = None
    visual_beats: list[StoryboardBeat] | None = None
    negative_prompt: str | None = None
    subtitle_text: str | None = None
    camera: str | None = None
    reference_asset_ids: list[str] | None = None
    start_frame_asset_id: str | None = None
    audio_asset_id: str | None = None
    continuity_from_shot_id: str | None = None
    continuation_mode: ContinuationMode | None = None
    continuity_in: ContinuityState | None = None
    continuity_out: ContinuityState | None = None
    seed: int | None = None
    fps: int | None = Field(default=None, ge=1, le=120)
    inference_steps: int | None = Field(default=None, ge=1, le=100)
    flow_shift: float | None = None


class ShotOrderUpdate(BaseModel):
    """D7：镜头顺序重排请求——shot_ids 必须恰好包含项目全部镜头，顺序即新时间线顺序。"""

    shot_ids: list[str] = Field(min_length=1)


class ProjectBriefUpdate(BaseModel):
    title: str | None = None
    prompt: str | None = Field(default=None, min_length=3)
    duration_seconds: int | None = Field(default=None, ge=15, le=MAX_PROJECT_DURATION_SECONDS)
    aspect_ratio: Literal["16:9", "9:16", "1:1"] | None = None
    style: str | None = None
    style_preset: str | None = None
    style_instructions: str | None = None
    language: str | None = None
    audience: str | None = None
    reference_asset_ids: list[str] | None = None
    quality: Literal["draft", "final"] | None = None
    subtitle_mode: Literal["none", "sidecar"] | None = None
    continuation_mode: ContinuationMode | None = None
    ultra_fast_anchor_strategy: UltraFastAnchorStrategy | None = None
    ultra_fast_transition: UltraFastTransition | None = None
    ultra_fast_transition_seconds: float | None = Field(default=None, ge=0.1, le=2.0)


class WorldBibleUpdate(BaseModel):
    logline: str | None = None
    visual_style: str | None = None
    character_notes: list[str] | None = None
    location_notes: list[str] | None = None
    prop_notes: list[str] | None = None
    audio_notes: list[str] | None = None
    continuity_rules: list[str] | None = None
    subjects: list[SubjectCard] | None = None


class ProjectUpdate(BaseModel):
    brief: ProjectBriefUpdate | None = None
    world_bible: WorldBibleUpdate | None = None


def _services(request: Request) -> StudioServices:
    return request.app.state.services


def _project_view(project: FilmProject) -> FilmProject:
    """Keep large planner diagnostics on the dedicated trace endpoint."""

    return project.model_copy(update={"planner_trace": []})


def _runner(request: Request) -> RenderManager:
    return request.app.state.render_manager


def _planning_manager(request: Request) -> PlanningManager:
    return request.app.state.planning_manager


async def _openai_compatible_health(base_url: str | None) -> bool:
    if not base_url:
        return False
    root = base_url.rstrip("/")
    if root.endswith("/v1/images/generations"):
        root = root[: -len("/v1/images/generations")]
    elif root.endswith("/v1"):
        root = root[:-3]
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{root}/health")
        return response.status_code < 400
    except httpx.HTTPError:
        return False


def create_api_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    async def health(request: Request) -> dict[str, object]:
        services = _services(request)
        probes = []
        labels = []
        if services.settings.h3_fl2va_url:
            probes.append(
                H3Client(
                    services.settings.h3_fl2va_url,
                    services.settings.h3_timeout_seconds,
                    services.settings.h3_flow_shift,
                ).health()
            )
            labels.append("fl2va")
        if services.settings.h3_ref2va_url:
            probes.append(
                H3Client(
                    services.settings.h3_ref2va_url,
                    services.settings.h3_timeout_seconds,
                    services.settings.h3_flow_shift,
                ).health()
            )
            labels.append("ref2va")
        results = await asyncio.gather(*probes) if probes else []
        healthy = dict(zip(labels, results, strict=True))
        capabilities = {item.id: item for item in services.compiler.capabilities()}
        t2i_configured = capabilities["qwen-image-t2i"].available
        t2i_healthy = (
            await _openai_compatible_health(services.settings.text_to_image_base_url) if t2i_configured else False
        )
        return {
            "status": "ok",
            "planner": services.settings.planner_wire_api if services.settings.planner_base_url else "heuristic",
            "planner_source": services.settings.planner_source,
            "planner_model": services.active_planner_view().get("resolved_model") or services.settings.planner_model,
            "fl2va_configured": services.settings.h3_configured("fl2va"),
            "fl2va_healthy": healthy.get("fl2va", False),
            "ref2va_configured": services.settings.h3_configured("ref2va"),
            "ref2va_healthy": healthy.get("ref2va", False),
            "image_edit_provider": services.settings.image_edit_provider,
            "image_edit_configured": capabilities["qwen-image-edit"].available,
            "image_edit_max_references": services.settings.image_edit_max_references,
            "t2i_provider": services.settings.text_to_image_provider,
            "t2i_model": services.settings.text_to_image_model,
            "t2i_configured": t2i_configured,
            "t2i_healthy": t2i_healthy,
            "render_estimate_scale": services.settings.render_estimate_scale,
            "render_profile": services.settings.render_profile,
            "render_max_concurrency": services.settings.render_max_concurrency,
            "planner_project_concurrency": services.settings.planner_project_concurrency,
        }

    @router.get("/capabilities")
    def capabilities(request: Request):
        return _services(request).compiler.capabilities()

    @router.get("/services/status")
    async def services_status(request: Request) -> dict[str, object]:
        services = _services(request)
        return await services.service_status.collect(
            planning_project_ids=_planning_manager(request).active_project_ids(),
            active_jobs=services.repository.list_active_jobs(),
        )

    @router.get("/style-presets")
    def style_presets() -> list[dict[str, object]]:
        """Return the canonical directing contracts used by the planner."""

        return public_style_contracts()

    @router.post("/assets/upload", response_model=list[AssetView])
    def upload_assets(
        request: Request,
        files: list[UploadFile] = File(...),
        tags: str = Form(""),
        roles: str = Form("reference"),
    ) -> list[AssetRecord]:
        services = _services(request)
        parsed_tags = [item.strip() for item in tags.split(",") if item.strip()]
        try:
            parsed_roles = [AssetRole(item.strip()) for item in roles.split(",") if item.strip()]
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        imported: list[AssetRecord] = []
        for upload in files:
            imported.append(
                services.assets.ingest_stream(
                    upload.file,
                    upload.filename or "asset.bin",
                    upload.content_type,
                    tags=parsed_tags,
                    roles=parsed_roles,
                )
            )
        return imported

    @router.post("/assets/import-path", response_model=list[AssetView])
    def import_path(request: Request, payload: ImportPathRequest) -> list[AssetRecord]:
        try:
            return _services(request).assets.import_path(
                payload.path,
                recursive=payload.recursive,
                copy_into_library=payload.copy_into_library,
                tags=payload.tags,
                roles=payload.roles,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/assets", response_model=list[AssetView])
    def list_assets(
        request: Request,
        q: str = "",
        kind: AssetKind | None = None,
        role: AssetRole | None = None,
    ) -> list[AssetRecord]:
        return _services(request).assets.search(q, kind=kind, role=role)

    @router.patch("/assets/{asset_id}", response_model=AssetView)
    def update_asset(request: Request, asset_id: str, payload: AssetUpdate) -> AssetRecord:
        try:
            return _services(request).assets.update(asset_id, payload)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="asset not found") from error

    @router.delete("/assets/{asset_id}")
    def delete_asset(request: Request, asset_id: str) -> dict[str, bool]:
        services = _services(request)
        asset = services.repository.get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        references: list[str] = []
        for project in services.repository.list_projects():
            if asset_id in project.brief.reference_asset_ids:
                references.append(f"{project.id}:project")
            for shot in project.shots:
                if asset_id in shot.reference_asset_ids or asset_id in {
                    shot.start_frame_asset_id,
                    shot.audio_asset_id,
                }:
                    references.append(f"{project.id}:{shot.id}")
        if references:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=("素材仍被项目或分镜引用，请先从相关项目中移除：" + ", ".join(sorted(set(references)))),
            )
        return {"deleted": services.assets.delete(asset_id)}

    @router.get("/assets/{asset_id}/content")
    def asset_content(request: Request, asset_id: str) -> FileResponse:
        asset = _services(request).repository.get_asset(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="asset not found")
        try:
            path = _services(request).assets.resolve_content_path(asset)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="asset content is missing") from None
        return FileResponse(path, media_type=asset.media_type, filename=asset.original_name)

    @router.post("/projects/import-thick-script", response_model=ThickScriptImportView)
    def import_thick_script(
        request: Request,
        script: UploadFile = File(...),
        asset_root: str = Form(""),
        title: str = Form(""),
    ) -> ThickScriptImportView:
        services = _services(request)
        try:
            raw_text = script.file.read().decode("utf-8-sig")
            parsed = parse_shot_script(raw_text)
            entries = scan_asset_codes(Path(asset_root)) if asset_root.strip() else {}
            imported = import_code_assets(services.assets, entries) if entries else {}
            project_title = title.strip() or Path(script.filename or "厚版脚本").stem
            project = build_film_project(
                parsed.shots,
                imported,
                title=project_title,
            )
            referenced = {code for shot in parsed.shots for code in shot.ordered_ref_codes}
            missing = detect_missing_codes(referenced, entries)
            project_path = services.settings.output_dir / f"{project.id}-plan.md"
            code_hints = {asset_id: code for code, asset_id in imported.items()}
            write_plan(
                project,
                project_path,
                code_hints=code_hints,
                missing_codes=missing,
                max_shots=5,
            )
            project.planner_trace.append(
                PlannerTraceEvent(
                    stage="thick-script-import",
                    status="completed",
                    message=f"parsed {len(project.shots)} shots; missing assets: {len(missing)}",
                )
            )
            services.repository.save_project(project)
            return ThickScriptImportView(
                project=_project_view(project),
                missing_codes=missing,
                imported_codes=sorted(imported),
                plan_path=str(project_path),
            )
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=422, detail="script must be UTF-8 Markdown") from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.post("/projects/plan", response_model=FilmProject)
    async def plan_project(request: Request, brief: ProjectBrief) -> FilmProject:
        services = _services(request)
        draft = FilmProject(
            brief=brief,
            world_bible=WorldBible(
                logline=brief.title or brief.prompt,
                visual_style=brief.style,
            ),
            shots=[],
            status="planning",
        )
        services.repository.save_project(draft)
        try:
            return _project_view(await services.resolve_planner().plan(brief, project_id=draft.id))
        except KeyError as error:
            failed = services.repository.get_project(draft.id) or draft
            failed.status = "failed"
            failed.updated_at = utc_now()
            services.repository.save_project(failed)
            raise HTTPException(
                status_code=422,
                detail={"message": str(error), "project_id": failed.id},
            ) from error
        except PlannerError as error:
            failed = services.repository.get_project(draft.id) or draft
            failed.status = "failed"
            failed.updated_at = utc_now()
            services.repository.save_project(failed)
            raise HTTPException(
                status_code=502,
                detail={"message": str(error), "project_id": failed.id},
            ) from error

    @router.post("/projects/plan-async", response_model=FilmProject, status_code=202)
    async def plan_project_async(request: Request, brief: ProjectBrief) -> FilmProject:
        return _project_view(await _planning_manager(request).start(brief))

    @router.get("/planning/active", response_model=list[str])
    def active_planning_projects(request: Request) -> list[str]:
        return _planning_manager(request).active_project_ids()

    @router.get("/projects", response_model=list[FilmProject])
    def list_projects(request: Request) -> list[FilmProject]:
        return [_project_view(project) for project in _services(request).repository.list_projects()]

    @router.get("/projects/{project_id}", response_model=FilmProject)
    def get_project(request: Request, project_id: str) -> FilmProject:
        project = _services(request).repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return _project_view(project)

    @router.delete("/projects/{project_id}")
    async def delete_project(request: Request, project_id: str) -> dict[str, object]:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        latest_job = services.repository.get_latest_job(project_id)
        if project_id in _runner(request).active_project_ids() or (
            latest_job and latest_job.status in {"queued", "running"}
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="项目正在渲染，完成或失败后才能删除。",
            )

        output_root = services.settings.output_dir.resolve()
        output_path = services.settings.output_dir / project_id
        resolved_output = output_path.resolve()
        if resolved_output.parent != output_root or output_path.is_symlink():
            raise HTTPException(status_code=409, detail="project output path is unsafe")

        await _planning_manager(request).cancel(project_id)
        output_deleted = False
        if resolved_output.is_dir():
            try:
                shutil.rmtree(resolved_output)
            except OSError as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to delete project output: {error}",
                ) from error
            output_deleted = True
        deleted = services.repository.delete_project(project_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="project not found")
        return {
            "deleted": True,
            "project_id": project_id,
            "output_deleted": output_deleted,
        }

    @router.get("/projects/{project_id}/planner-trace", response_model=list[PlannerTraceEvent])
    def planner_trace(request: Request, project_id: str) -> list[PlannerTraceEvent]:
        project = _services(request).repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return project.planner_trace

    @router.patch("/projects/{project_id}", response_model=FilmProject)
    def update_project(request: Request, project_id: str, payload: ProjectUpdate) -> FilmProject:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        brief = project.brief
        if payload.brief is not None:
            brief = ProjectBrief.model_validate({**brief.model_dump(), **payload.brief.model_dump(exclude_unset=True)})
        world_bible = project.world_bible
        if payload.world_bible is not None:
            world_bible = WorldBible.model_validate(
                {**world_bible.model_dump(), **payload.world_bible.model_dump(exclude_unset=True)}
            )
        project = FilmProject.model_validate(
            {
                **project.model_dump(),
                "brief": brief,
                "world_bible": world_bible,
                "status": "planned",
                "updated_at": utc_now(),
            }
        )
        return services.repository.save_project(project)

    @router.patch("/projects/{project_id}/shots/order", response_model=FilmProject)
    def reorder_shots(request: Request, project_id: str, payload: ShotOrderUpdate) -> FilmProject:
        """D7：按 shot_ids 顺序重排镜头，重写 index 并触发 timeline 重建。

        与 update_shot 一致，重排重置受影响镜头的渲染状态；磁盘产物不删除。
        注意：必须声明在 /shots/{shot_id} 之前，避免 "order" 被当作 shot_id 捕获。
        """
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        existing = {shot.id for shot in project.shots}
        if set(payload.shot_ids) != existing or len(payload.shot_ids) != len(existing):
            raise HTTPException(status_code=422, detail="shot_ids must include every shot exactly once")
        by_id = {shot.id: shot for shot in project.shots}
        reordered = []
        for index, shot_id in enumerate(payload.shot_ids):
            shot = by_id[shot_id]
            reordered.append(
                shot.model_copy(
                    update={
                        "index": index,
                        "status": ShotStatus.PLANNED,
                        "selected_take_path": None,
                        "anchor_frame_path": None,
                        "boundary_frame_path": None,
                        "render_started_at": None,
                        "render_completed_at": None,
                        "render_duration_seconds": None,
                    }
                )
            )
        project = FilmProject.model_validate(
            {
                **project.model_dump(mode="python"),
                "shots": reordered,
                "status": "planned",
                "updated_at": utc_now(),
            }
        )
        return services.repository.save_project(project)

    @router.patch("/projects/{project_id}/shots/{shot_id}", response_model=FilmProject)
    def update_shot(
        request: Request,
        project_id: str,
        shot_id: str,
        payload: ShotUpdate,
    ) -> FilmProject:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        for position, shot in enumerate(project.shots):
            if shot.id == shot_id:
                try:
                    updates = payload.model_dump(exclude_unset=True)
                    new_duration = updates.get("duration_seconds")
                    if new_duration is not None and new_duration != shot.duration_seconds:
                        ratio = new_duration / shot.duration_seconds
                        if "dialogue" not in updates:
                            updates["dialogue"] = [
                                line.model_copy(
                                    update={
                                        "start_seconds": round(line.start_seconds * ratio, 3),
                                        "end_seconds": (
                                            round(line.end_seconds * ratio, 3) if line.end_seconds is not None else None
                                        ),
                                    }
                                )
                                for line in shot.dialogue
                            ]
                        updates["visual_beats"] = [
                            beat.model_copy(
                                update={
                                    "start_seconds": round(beat.start_seconds * ratio, 3),
                                    "end_seconds": round(beat.end_seconds * ratio, 3),
                                }
                            )
                            for beat in shot.visual_beats
                        ]
                    project.shots[position] = ShotSpec.model_validate(
                        {
                            **shot.model_dump(),
                            **updates,
                            "status": ShotStatus.PLANNED,
                            "selected_take_path": None,
                            "anchor_frame_path": None,
                            "boundary_frame_path": None,
                            "render_started_at": None,
                            "render_completed_at": None,
                            "render_duration_seconds": None,
                        }
                    )
                except ValidationError as exc:
                    detail = [
                        {
                            "type": error["type"],
                            "loc": error["loc"],
                            "msg": error["msg"],
                        }
                        for error in exc.errors(include_url=False)
                    ]
                    raise HTTPException(status_code=422, detail=detail) from exc
                project = FilmProject.model_validate(project.model_dump(mode="python"))
                project.status = "planned"
                project.updated_at = utc_now()
                return services.repository.save_project(project)
        raise HTTPException(status_code=404, detail="shot not found")

    @router.get("/planner-profiles")
    def list_planner_profiles(request: Request) -> dict[str, object]:
        services = _services(request)
        return {
            "profiles": [profile.public() for profile in services.settings.planner_profiles],
            "active": services.active_planner_view(),
        }

    @router.get("/llm-clients")
    def list_llm_clients(request: Request) -> dict[str, object]:
        services = _services(request)
        return {
            "clients": services.list_llm_clients(),
            "active": services.active_planner_view(),
        }

    @router.post("/llm-clients", status_code=201)
    def create_llm_client(request: Request, payload: LLMClientCreate) -> dict[str, object]:
        services = _services(request)
        try:
            return services.create_llm_client(LLMClient(**payload.model_dump()))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.put("/llm-clients/{client_id}")
    def update_llm_client(request: Request, client_id: str, payload: LLMClientUpdate) -> dict[str, object]:
        services = _services(request)
        try:
            return services.update_llm_client(client_id, payload.model_dump(exclude_unset=True))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="llm client not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.delete("/llm-clients/{client_id}")
    def delete_llm_client(request: Request, client_id: str) -> dict[str, object]:
        services = _services(request)
        try:
            return services.delete_llm_client(client_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="llm client not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post("/llm-clients/{client_id}/default")
    def set_default_llm_client(
        request: Request, client_id: str, payload: DefaultLLMClientRequest | None = None,
    ) -> dict[str, str | bool]:
        services = _services(request)
        try:
            return services.set_default_llm_client(client_id, payload.model if payload else "")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/planner-profile/active")
    def get_active_planner(request: Request) -> dict[str, str | bool]:
        return _services(request).active_planner_view()

    @router.put("/planner-profile/active")
    def set_active_planner(request: Request, payload: ActivePlannerRequest) -> dict[str, str | bool]:
        services = _services(request)
        try:
            services.set_active_planner(payload.profile_id, payload.model)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return services.active_planner_view()

    @router.post("/projects/{project_id}/h3-enrichment", response_model=FilmProject)
    async def start_h3_enrichment(
        request: Request, project_id: str, payload: H3BatchPlanningRequest,
    ) -> FilmProject:
        try:
            project = await _planning_manager(request).start_imported_h3(
                project_id, profile_id=payload.profile_id, model=payload.model, retry=payload.retry,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="project not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _project_view(project)

    @router.post("/projects/{project_id}/preproduction", response_model=PreproductionPlan)
    async def create_preproduction_plan(
        request: Request,
        project_id: str,
        payload: PreproductionRequest | None = None,
    ) -> PreproductionPlan:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        if project.batch_planning_run:
            if project.batch_planning_run.status.value != "complete":
                raise HTTPException(status_code=409, detail="H3 分镜批处理尚未完成，请等待或重试失败批次")
        else:
            try:
                planner = services.resolve_planner(
                    profile_id=(payload.profile_id if payload else None),
                    model=(payload.model if payload else None),
                )
                project = await planner.enrich_imported_shots(project)
            except PlannerError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            except (httpx.HTTPError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"H3 timeline enrichment failed: {error}") from error
        project.preproduction_plan = services.preproduction.plan(project, services.repository.list_assets())
        project = services.preproduction_assets.apply_execution_bindings(project)
        return services.repository.save_project(project).preproduction_plan

    @router.get("/projects/{project_id}/preproduction", response_model=PreproductionPlan)
    def get_preproduction_plan(request: Request, project_id: str) -> PreproductionPlan:
        project = _services(request).repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        if not project.preproduction_plan:
            raise HTTPException(status_code=404, detail="preproduction plan not created")
        return project.preproduction_plan

    @router.post("/projects/{project_id}/preproduction/approve", response_model=PreproductionPlan)
    async def approve_preproduction_plan(request: Request, project_id: str) -> PreproductionPlan:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        plan = project.preproduction_plan
        if not plan:
            raise HTTPException(status_code=409, detail="create and review a preproduction plan first")
        if plan.blockers:
            raise HTTPException(status_code=409, detail="resolve preproduction blockers before approval")
        approved = plan.model_copy(update={"status": PreproductionStatus.APPROVED, "approved_at": utc_now()})
        if not approved.generated_image_count:
            project.preproduction_plan = approved.become_ready()
            return services.repository.save_project(project).preproduction_plan
        project.preproduction_plan = approved.model_copy(update={"status": PreproductionStatus.GENERATING_ASSETS})
        project = await services.preproduction_assets.generate_approved_gaps(project)
        return services.repository.save_project(project).preproduction_plan

    @router.post("/projects/{project_id}/compile", response_model=ExecutionPlan)
    def compile_project(request: Request, project_id: str) -> ExecutionPlan:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        plan = services.compiler.compile(project)
        project.status = "compiled"
        services.repository.save_project(project)
        return plan

    @router.post("/projects/{project_id}/render", response_model=RenderJob)
    async def render_project(
        request: Request,
        project_id: str,
        force: bool = Query(False, description="Re-render every shot instead of reusing completed takes"),
        shot_id: str | None = Query(None, description="单镜渲染：只重渲染这一个镜头，其余不动、不组装 final.mp4"),
    ) -> RenderJob:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        if project.preproduction_plan and project.preproduction_plan.status != PreproductionStatus.READY:
            raise HTTPException(
                status_code=409,
                detail="preproduction plan must be approved and ready before rendering",
            )
        if shot_id:
            # D7-M 单镜渲染：只校验目标镜头（H3 backend），提交渲染指定镜头。
            shot = next((item for item in project.shots if item.id == shot_id), None)
            if not shot:
                raise HTTPException(status_code=404, detail="shot not found")
            runtime_task = effective_video_task(
                shot,
                ref2va_configured=services.settings.h3_configured("ref2va"),
                fl2va_configured=services.settings.h3_configured("fl2va"),
                continuation_mode=resolved_continuation_mode(project, shot),
            )
            missing_single: list[str] = []
            if runtime_task == ShotTask.REF2VA and not services.settings.h3_configured("ref2va"):
                missing_single.append("MiniMax-H3 Ref2VA backend")
            elif runtime_task == ShotTask.FL2VA and not services.settings.h3_configured("fl2va"):
                missing_single.append("MiniMax-H3 FL2VA backend")
            if missing_single:
                raise HTTPException(status_code=409, detail="制作前置条件未满足：" + "；".join(missing_single))
            try:
                return _runner(request).submit(project_id, force=force, shot_ids=[shot_id])
            except KeyError as error:
                raise HTTPException(status_code=404, detail="project not found") from error
        missing: list[str] = []
        ordered_shots = sorted(project.shots, key=lambda shot: shot.index)
        pending_shots = (
            ordered_shots
            if force
            else [shot for shot in ordered_shots if RenderManager.reusable_take_path(shot) is None]
        )
        ultra_independent = {shot.id: uses_independent_ultra_fast_anchor(project, shot) for shot in pending_shots}
        continuation_modes = {
            shot.id: (
                resolved_continuation_mode(project, shot)
                if not shot.start_frame_asset_id
                and (
                    shot.continuity_from_shot_id
                    or resolved_continuation_mode(project, shot) == ContinuationMode.ULTRA_FAST
                )
                else None
            )
            for shot in pending_shots
        }
        runtime_tasks = {
            shot.id: effective_video_task(
                shot,
                ref2va_configured=services.settings.h3_configured("ref2va"),
                fl2va_configured=services.settings.h3_configured("fl2va"),
                continuation_mode=continuation_modes[shot.id],
            )
            for shot in pending_shots
        }
        if (
            any(task == ShotTask.FL2VA for task in runtime_tasks.values())
            and not services.settings.h3_configured("fl2va")
        ):
            missing.append("MiniMax-H3 FL2VA backend")
        if (
            any(task == ShotTask.REF2VA for task in runtime_tasks.values())
            and not services.settings.h3_configured("ref2va")
        ):
            missing.append("MiniMax-H3 Ref2VA backend")
        preceding_shot_ids: set[str] = set()
        for position, shot in enumerate(ordered_shots):
            if not force and RenderManager.reusable_take_path(shot) is not None:
                preceding_shot_ids.add(shot.id)
                continue
            runtime_task = runtime_tasks[shot.id]
            assets = [services.repository.get_asset(asset_id) for asset_id in shot.reference_asset_ids]
            assets = [asset for asset in assets if asset]
            is_clip_continuation = bool(
                shot.continuity_from_shot_id and not shot.start_frame_asset_id and runtime_task == ShotTask.REF2VA
            )
            if (
                shot.continuity_from_shot_id
                and not shot.start_frame_asset_id
                and shot.continuity_from_shot_id not in preceding_shot_ids
            ):
                missing.append(f"earlier continuation source for shot {shot.index + 1}")
            if runtime_task == ShotTask.FL2VA:
                image_references = [asset for asset in assets if asset.kind == AssetKind.IMAGE]
                first_shot_needs_t2i = (
                    position == 0
                    and not shot.start_frame_asset_id
                    and not shot.continuity_from_shot_id
                    and bool(shot.anchor_prompt)
                )
                is_boundary_continuation = (
                    continuation_modes[shot.id] == ContinuationMode.ULTRA_FAST
                    and bool(shot.continuity_from_shot_id)
                    and not ultra_independent[shot.id]
                )
                needs_anchor = (
                    first_shot_needs_t2i
                    or ultra_independent[shot.id]
                    or (
                        not is_boundary_continuation
                        and anchor_selected(
                            shot,
                            position,
                            services.settings.image_edit_anchor_mode,
                        )
                    )
                )
                if needs_anchor:
                    uses_image_edit = bool(image_references or shot.continuity_from_shot_id)
                    if uses_image_edit:
                        image_edit_configured = bool(
                            services.settings.image_edit_provider not in {"", "disabled", "none"}
                            and services.settings.image_edit_base_url
                            and services.settings.image_edit_model
                        )
                        if not image_edit_configured:
                            missing.append(
                                f"Image Edit endpoint for shot {shot.index + 1} "
                                "(set STUDIO_IMAGE_EDIT_BASE_URL and STUDIO_IMAGE_EDIT_MODEL)"
                            )
                    else:
                        t2i_configured = bool(
                            services.settings.text_to_image_provider not in {"", "disabled", "none"}
                            and services.settings.text_to_image_base_url
                        )
                        if not t2i_configured:
                            missing.append(
                                f"T2I endpoint for zero-material shot {shot.index + 1} (set STUDIO_T2I_BASE_URL)"
                            )
                elif not shot.start_frame_asset_id and not image_references and not shot.continuity_from_shot_id:
                    missing.append(f"explicit start frame for shot {shot.index + 1}")
            if runtime_task == ShotTask.REF2VA and not is_clip_continuation:
                has_image = any(asset.kind == AssetKind.IMAGE for asset in assets)
                has_media = any(asset.kind in {AssetKind.AUDIO, AssetKind.VIDEO} for asset in assets)
                if not (has_image and has_media):
                    missing.append(f"image plus audio/video references for shot {shot.index + 1}")
            preceding_shot_ids.add(shot.id)
        if missing:
            raise HTTPException(
                status_code=409,
                detail="制作前置条件未满足：" + "；".join(missing),
            )
        try:
            return _runner(request).submit(project_id, force=force)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="project not found") from error

    @router.get("/projects/{project_id}/render-estimate", response_model=ProjectRenderEstimate)
    def project_render_estimate(request: Request, project_id: str) -> ProjectRenderEstimate:
        services = _services(request)
        project = services.repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return services.estimator.estimate_project(project)

    @router.get("/jobs/active", response_model=list[RenderJob])
    def active_jobs(request: Request) -> list[RenderJob]:
        return _services(request).repository.list_active_jobs()

    @router.get("/jobs/{job_id}", response_model=RenderJob)
    def get_job(request: Request, job_id: str) -> RenderJob:
        job = _services(request).repository.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @router.get("/projects/{project_id}/jobs/latest", response_model=RenderJob | None)
    def latest_project_job(request: Request, project_id: str) -> RenderJob | None:
        if not _services(request).repository.get_project(project_id):
            raise HTTPException(status_code=404, detail="project not found")
        return _services(request).repository.get_latest_job(project_id)

    @router.get("/projects/{project_id}/shots/{shot_id}/boundary")
    def shot_boundary(request: Request, project_id: str, shot_id: str) -> FileResponse:
        project = _services(request).repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        shot = next((item for item in project.shots if item.id == shot_id), None)
        if not shot or not shot.boundary_frame_path:
            raise HTTPException(status_code=404, detail="boundary frame not ready")
        path = Path(shot.boundary_frame_path).resolve()
        output_root = (_services(request).settings.output_dir / project_id).resolve()
        if output_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="boundary frame not found")
        return FileResponse(path, media_type="image/png")

    @router.get("/projects/{project_id}/shots/{shot_id}/anchor")
    def shot_anchor(request: Request, project_id: str, shot_id: str) -> FileResponse:
        project = _services(request).repository.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        shot = next((item for item in project.shots if item.id == shot_id), None)
        if not shot or not shot.anchor_frame_path:
            raise HTTPException(status_code=404, detail="anchor frame not ready")
        path = Path(shot.anchor_frame_path).resolve()
        output_root = (_services(request).settings.output_dir / project_id).resolve()
        if output_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="anchor frame not found")
        return FileResponse(path, media_type="image/png")

    @router.get("/jobs/{job_id}/output")
    def job_output(
        request: Request,
        job_id: str,
        download: bool = Query(False),
    ) -> FileResponse:
        job = _services(request).repository.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "complete" or not job.output_path:
            raise HTTPException(status_code=409, detail="render output is not ready")
        path = Path(job.output_path).resolve()
        output_root = (_services(request).settings.output_dir / job.project_id).resolve()
        if output_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="render output is missing")
        filename = f"{job.project_id}.mp4"
        if download:
            return FileResponse(path, media_type="video/mp4", filename=filename)
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    @router.get("/jobs/{job_id}/subtitles")
    def job_subtitles(request: Request, job_id: str) -> FileResponse:
        job = _services(request).repository.get_job(job_id)
        if not job or job.status != "complete" or not job.subtitle_path:
            raise HTTPException(status_code=404, detail="sidecar subtitles are not available")
        path = Path(job.subtitle_path).resolve()
        output_root = (_services(request).settings.output_dir / job.project_id).resolve()
        if output_root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="sidecar subtitles are missing")
        return FileResponse(path, media_type="application/x-subrip", filename=f"{job.project_id}.srt")

    @router.get("/outputs")
    def list_outputs(request: Request, project_id: str = Query(...)) -> list[dict[str, object]]:
        output_dir = _services(request).settings.output_dir / project_id
        if not output_dir.exists():
            return []
        return [{"name": path.name, "size_bytes": path.stat().st_size} for path in sorted(output_dir.glob("*.mp4"))]

    return router
