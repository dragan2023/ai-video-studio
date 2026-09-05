// Nautilus Studio REST 客户端（M9 只读桥接）
// 后端契约：nautilus-studio/src/long_video_studio/domain.py + api.py（router prefix = /api）
// 请求经 vite dev proxy（/api -> http://127.0.0.1:7860）转发，生产走 nginx 同规则。

export type NautilusAssetKind = "image" | "video" | "audio" | "document" | "other";
export type NautilusAssetRole = "character" | "location" | "prop" | "style" | "start_frame" | "audio" | "reference";

export interface NautilusAsset {
    id: string;
    sha256: string;
    original_name: string;
    display_name: string;
    media_type: string;
    kind: NautilusAssetKind;
    size_bytes: number;
    width?: number | null;
    height?: number | null;
    duration_seconds?: number | null;
    caption: string;
    tags: string[];
    roles: NautilusAssetRole[];
    source: "upload" | "path";
    created_at: string;
}

export interface NautilusSubject {
    subject_id: string;
    label: string;
    aliases?: string[];
    visual_identity?: string;
    wardrobe?: string;
    reference_asset_ids?: string[];
    speaker_id?: string | null;
}

export interface NautilusWorldBible {
    logline: string;
    visual_style: string;
    character_notes?: string[];
    location_notes?: string[];
    prop_notes?: string[];
    audio_notes?: string[];
    continuity_rules?: string[];
    subjects: NautilusSubject[];
}

export interface NautilusShot {
    id: string;
    index: number;
    title: string;
    purpose: string;
    source_section?: string;
    duration_seconds: number;
    task: "fl2va" | "ref2va";
    transition_kind: string;
    prompt: string;
    audio_prompt?: string;
    music_prompt?: string;
    dialogue?: unknown[];
    opening_state?: string;
    ending_state?: string;
    continuity_handoff?: string;
    reference_anchors?: string[];
    hook?: string;
    visual_beats?: unknown[];
    negative_prompt?: string;
    subtitle_text?: string | null;
    camera?: string;
    reference_asset_ids: string[];
    start_frame_asset_id?: string | null;
    audio_asset_id?: string | null;
    continuity_from_shot_id?: string | null;
    seed?: number;
    fps?: number;
    inference_steps?: number;
    flow_shift?: number;
    status: "planned" | "ready" | "rendering" | "complete" | "failed";
    selected_take_path?: string | null;
    anchor_frame_path?: string | null;
    boundary_frame_path?: string | null;
}

export interface NautilusBrief {
    title: string;
    prompt: string;
    duration_seconds: number;
    aspect_ratio: string;
    style: string;
    continuation_mode: string;
    quality: string;
}

export interface NautilusProject {
    id: string;
    brief: NautilusBrief;
    world_bible: NautilusWorldBible;
    shots: NautilusShot[];
    status: string;
    created_at: string;
    updated_at: string;
}

const NAUTILUS_API_BASE = "/api";

export class NautilusApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
        super(message);
        this.name = "NautilusApiError";
        this.status = status;
    }
}

async function apiGet<T>(path: string): Promise<T> {
    const response = await fetch(`${NAUTILUS_API_BASE}${path}`, { headers: { Accept: "application/json" } });
    if (!response.ok) {
        let detail: unknown = null;
        try {
            const body = await response.json();
            detail = (body as { detail?: unknown })?.detail;
        } catch {
            // non-JSON error body
        }
        const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `nautilus request failed (${response.status})`;
        throw new NautilusApiError(message, response.status);
    }
    return (await response.json()) as T;
}

export function listProjects(): Promise<NautilusProject[]> {
    return apiGet<NautilusProject[]>("/projects");
}

export function getProject(projectId: string): Promise<NautilusProject> {
    return apiGet<NautilusProject>(`/projects/${projectId}`);
}

export function listAssets(): Promise<NautilusAsset[]> {
    return apiGet<NautilusAsset[]>("/assets");
}

/** 素材二进制内容 URL（图片直接可 <img src>；视频/音频可 <video>/<audio> src）。 */
export function assetUrl(assetId: string, version = 0): string {
    return `${NAUTILUS_API_BASE}/assets/${assetId}/content?version=${version}`;
}


// ── M10：写回 API（updateShot / reorderShots） ────────────────────────────────
// ShotUpdate 字段清单对齐 nautilus api.py ShotUpdate（api.py:112-146）。
export interface ShotUpdatePayload {
    title?: string;
    purpose?: string;
    duration_seconds?: number;
    task?: "fl2va" | "ref2va";
    transition_kind?: string;
    prompt?: string;
    anchor_prompt?: string;
    audio_prompt?: string;
    music_prompt?: string;
    dialogue?: unknown[];
    opening_state?: string;
    ending_state?: string;
    continuity_handoff?: string;
    reference_anchors?: string[];
    hook?: string;
    visual_beats?: unknown[];
    negative_prompt?: string;
    subtitle_text?: string;
    camera?: string;
    reference_asset_ids?: string[];
    start_frame_asset_id?: string;
    audio_asset_id?: string;
    continuity_from_shot_id?: string;
    continuation_mode?: string;
    continuity_in?: unknown;
    continuity_out?: unknown;
    seed?: number;
    fps?: number;
    inference_steps?: number;
    flow_shift?: number;
}

async function apiSend<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await fetch(`${NAUTILUS_API_BASE}${path}`, {
        method,
        headers: { Accept: "application/json", ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
        body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
        let detail: unknown = null;
        try {
            const parsed = await response.json();
            detail = (parsed as { detail?: unknown })?.detail;
        } catch {
            // non-JSON error body
        }
        const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `nautilus request failed (${response.status})`;
        throw new NautilusApiError(message, response.status);
    }
    return (await response.json()) as T;
}

/** 更新单个镜头（PATCH /shots/{id}）；返回更新后的完整 FilmProject。 */
export function updateShot(projectId: string, shotId: string, payload: ShotUpdatePayload): Promise<NautilusProject> {
    return apiSend<NautilusProject>("PATCH", `/projects/${projectId}/shots/${shotId}`, payload);
}

/** 镜头顺序重排（D7：PATCH /shots/order）；shotIds 必须恰好包含全部镜头。 */
export function reorderShots(projectId: string, shotIds: string[]): Promise<NautilusProject> {
    return apiSend<NautilusProject>("PATCH", `/projects/${projectId}/shots/order`, { shot_ids: shotIds });
}



// ── M11：预制片计划 + 渲染状态 API ────────────────────────────────────────────
export interface NautilusPreproductionShotPlan {
    shot_id: string;
    shot_index: number;
    script_evidence?: string;
    transition_kind: string;
    start_frame_source: string;
    source_shot_id?: string | null;
    candidate_asset_ids: string[];
    selected_asset_id?: string | null;
    gap_reason: string;
    confidence: number;
    generation_permitted: boolean;
    parameter_summary?: string;
}

export interface NautilusPreproductionPlan {
    version: number;
    asset_input_fingerprint?: string;
    generated_image_count: number;
    runtime_profile?: Record<string, unknown>;
    blockers: string[];
    warnings: string[];
    approved_at?: string | null;
    status: string;
    shot_plans: NautilusPreproductionShotPlan[];
}

export interface NautilusRenderJob {
    id: string;
    project_id: string;
    shot_ids?: string[] | null;
    status: "queued" | "running" | "complete" | "failed";
    progress: number;
    current_shot_id?: string | null;
    current_service_id?: string | null;
    message: string;
    output_path?: string | null;
    subtitle_path?: string | null;
    error?: string | null;
    retry_count: number;
    max_retries: number;
    estimated_seconds?: number | null;
    created_at: string;
    started_at?: string | null;
    completed_at?: string | null;
}

export function createPreproduction(projectId: string): Promise<NautilusPreproductionPlan> {
    return apiSend<NautilusPreproductionPlan>("POST", `/projects/${projectId}/preproduction`, {});
}

export function getPreproduction(projectId: string): Promise<NautilusPreproductionPlan> {
    return apiGet<NautilusPreproductionPlan>(`/projects/${projectId}/preproduction`);
}

export function approvePreproduction(projectId: string): Promise<NautilusPreproductionPlan> {
    return apiSend<NautilusPreproductionPlan>("POST", `/projects/${projectId}/preproduction/approve`, {});
}

export function startRender(projectId: string, force = false, shotId?: string): Promise<NautilusRenderJob> {
    const params: string[] = [];
    if (force) params.push("force=true");
    if (shotId) params.push(`shot_id=${encodeURIComponent(shotId)}`);
    return apiSend<NautilusRenderJob>("POST", `/projects/${projectId}/render${params.length ? "?" + params.join("&") : ""}`, {});
}

export function getJob(jobId: string): Promise<NautilusRenderJob> {
    return apiGet<NautilusRenderJob>(`/jobs/${jobId}`);
}

export function getActiveJobs(): Promise<NautilusRenderJob[]> {
    return apiGet<NautilusRenderJob[]>("/jobs/active");
}



// ── 画布 → nautilus 素材入库（资产生成工作流桥接）──
// POST /api/assets/upload：multipart 表单 files[] + tags + roles，返回 list[AssetView]。
export interface UploadAssetOptions {
    tags?: string[];
    roles?: NautilusAssetRole[];
}

/** 把画布生成的图片/视频/音频 Blob 上传进 nautilus 素材库（sha256 去重合并 tags/roles）。 */
export async function uploadAssets(files: Blob[], options: UploadAssetOptions = {}): Promise<NautilusAsset[]> {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    // nautilus API 声明为 Form(str)，多标签/多角色必须逗号编码而非重复字段。
    form.append("tags", (options.tags || []).join(","));
    form.append("roles", (options.roles || ["reference"]).join(","));
    const response = await fetch(`${NAUTILUS_API_BASE}/assets/upload`, { method: "POST", body: form });
    if (!response.ok) {
        let detail: unknown = null;
        try {
            const parsed = await response.json();
            detail = (parsed as { detail?: unknown })?.detail;
        } catch {
            // ignore
        }
        const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `assets upload failed (${response.status})`;
        throw new NautilusApiError(message, response.status);
    }
    return (await response.json()) as NautilusAsset[];
}

