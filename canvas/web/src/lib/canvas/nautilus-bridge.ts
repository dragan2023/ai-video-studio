// Nautilus → 画布 映射（M9 只读桥接）
// 把 nautilus FilmProject（镜头/素材/角色）投影为画布 CanvasProject 的 nodes + connections。
// 数据权威在 nautilus SQLite；本项目（CanvasProject）只作可视化投影，业务数据全部冗余在节点 metadata。
// 后续 M10 双向同步基于 metadata.shotId / assetId 反查回写。

import { nanoid } from "nanoid";

import type { CanvasConnection, CanvasNodeData, CanvasNodeMetadata } from "@/types/canvas";
import type { NautilusAsset, NautilusPreproductionPlan, NautilusProject, NautilusShot, NautilusSubject } from "@/services/api/nautilus";
import { assetUrl } from "@/services/api/nautilus";
// 注意：节点注册由页面层显式调用 registerNautilusNodes()（见 pages/canvas/index.tsx），
// 本模块保持纯逻辑以支持 vitest 单测。

export type NautilusShotMetadata = CanvasNodeMetadata & {
    nautilusKind?: "shot";
    projectId?: string;
    shotId?: string;
    shotIndex?: number;
    shotTitle?: string;
    duration?: number;
    task?: string;
    transitionKind?: string;
    dialogue?: unknown[];
    visualBeats?: unknown[];
    audioPrompt?: string;
    musicPrompt?: string;
    purpose?: string;
    camera?: string;
    negativePrompt?: string;
    openingState?: string;
    endingState?: string;
    continuityHandoff?: string;
    hook?: string;
    fps?: number;
    inferenceSteps?: number;
    flowShift?: number;
    seed?: number;
    continuityFromShotId?: string;
    startFrameAssetId?: string;
    shotStatus?: string;
    preproduction?: NautilusShotPreproduction;
};

export type NautilusShotPreproduction = {
    startFrameSource?: string;
    transitionKind?: string;
    confidence?: number;
    gapReason?: string;
    generationPermitted?: boolean;
    selectedAssetId?: string;
};

export type NautilusCharacterMetadata = CanvasNodeMetadata & {
    nautilusKind?: "character";
    subjectId?: string;
    label?: string;
    wardrobe?: string;
};

export type NautilusAssetMetadata = CanvasNodeMetadata & {
    nautilusKind?: "asset";
    assetId?: string;
    roles?: string[];
};

export type ProjectToCanvasResult = {
    title: string;
    nodes: CanvasNodeData[];
    connections: CanvasConnection[];
};

const ASSETS_PER_ROW = 12;
const SHOTS_PER_ROW = 10;
const NODE_GAP_X = 380;
const NODE_GAP_Y = 300;
const ASSET_TOP = 60;
const CHARACTER_TOP = 320;
const SHOT_TOP = 560;

function assetNodeType(kind: string): CanvasNodeData["type"] {
    switch (kind) {
        case "image":
            return "image";
        case "video":
            return "video";
        case "audio":
            return "audio";
        default:
            return "text";
    }
}

export function projectToCanvas(project: NautilusProject, assets: NautilusAsset[], preproduction?: NautilusPreproductionPlan): ProjectToCanvasResult {
    const nodes: CanvasNodeData[] = [];
    const connections: CanvasConnection[] = [];
    const assetNodeOf = new Map<string, string>();
    const planByShot = new Map<string, NonNullable<NautilusPreproductionPlan["shot_plans"]>[number]>(
        (preproduction?.shot_plans || []).map((p) => [p.shot_id, p]),
    );

    // ── 素材节点区（顶部）──
    assets.forEach((asset, i) => {
        const id = `asset-${asset.id}`;
        const type = assetNodeType(asset.kind);
        const meta: NautilusAssetMetadata = {
            nautilusKind: "asset",
            assetId: asset.id,
            roles: asset.roles,
            status: "success",
        };
        if (type === "image") {
            meta.content = assetUrl(asset.id);
            meta.mimeType = asset.media_type;
            meta.naturalWidth = asset.width ?? undefined;
            meta.naturalHeight = asset.height ?? undefined;
        } else if (type === "video" || type === "audio") {
            meta.content = assetUrl(asset.id);
            meta.mimeType = asset.media_type;
            meta.durationMs = asset.duration_seconds ? asset.duration_seconds * 1000 : undefined;
        }
        const col = i % ASSETS_PER_ROW;
        const row = Math.floor(i / ASSETS_PER_ROW);
        nodes.push({
            id,
            type,
            title: asset.display_name || asset.original_name,
            position: { x: 40 + col * NODE_GAP_X, y: ASSET_TOP + row * NODE_GAP_Y },
            width: 340,
            height: 240,
            metadata: meta as CanvasNodeMetadata,
        });
        assetNodeOf.set(asset.id, id);
    });

    // ── 角色节点区 ──
    const subjects: NautilusSubject[] = project.world_bible?.subjects || [];
    subjects.forEach((subject, i) => {
        const id = `character-${subject.subject_id}`;
        const meta: NautilusCharacterMetadata = {
            nautilusKind: "character",
            subjectId: subject.subject_id,
            label: subject.label,
            wardrobe: subject.wardrobe,
        };
        nodes.push({
            id,
            type: "nautilus:character",
            title: subject.label,
            position: { x: 40 + i * 220, y: CHARACTER_TOP },
            width: 200,
            height: 150,
            metadata: meta as CanvasNodeMetadata,
        });
        (subject.reference_asset_ids || []).forEach((aid) => {
            const nid = assetNodeOf.get(aid);
            if (nid) connections.push({ id: nanoid(), fromNodeId: nid, toNodeId: id });
        });
    });

    // ── 分镜节点区（主区域）──
    const shots: NautilusShot[] = [...project.shots].sort((a, b) => a.index - b.index);
    const shotNodeOf = new Map<string, string>();
    shots.forEach((shot, i) => {
        const id = `shot-${shot.id}`;
        const meta: NautilusShotMetadata = {
            nautilusKind: "shot",
            projectId: project.id,
            shotId: shot.id,
            shotIndex: shot.index,
            shotTitle: shot.title,
            status: shot.status === "complete" ? "success" : shot.status === "rendering" ? "loading" : "idle",
            shotStatus: shot.status,
            prompt: shot.prompt,
            duration: shot.duration_seconds,
            task: shot.task,
            transitionKind: shot.transition_kind,
            dialogue: shot.dialogue,
            visualBeats: shot.visual_beats,
            audioPrompt: shot.audio_prompt,
            musicPrompt: shot.music_prompt,
            purpose: shot.purpose,
            camera: shot.camera,
            negativePrompt: shot.negative_prompt,
            openingState: shot.opening_state,
            endingState: shot.ending_state,
            continuityHandoff: shot.continuity_handoff,
            hook: shot.hook,
            fps: shot.fps,
            inferenceSteps: shot.inference_steps,
            flowShift: shot.flow_shift,
            seed: shot.seed,
            continuityFromShotId: shot.continuity_from_shot_id || undefined,
            startFrameAssetId: shot.start_frame_asset_id || undefined,
            references: shot.reference_asset_ids,
        };
        const plan = planByShot.get(shot.id);
        if (plan) {
            meta.preproduction = {
                startFrameSource: plan.start_frame_source,
                transitionKind: plan.transition_kind,
                confidence: plan.confidence,
                gapReason: plan.gap_reason,
                generationPermitted: plan.generation_permitted,
                selectedAssetId: plan.selected_asset_id || undefined,
            };
        }
        const col = i % SHOTS_PER_ROW;
        const row = Math.floor(i / SHOTS_PER_ROW);
        nodes.push({
            id,
            type: "nautilus:shot",
            title: shot.title,
            position: { x: 40 + col * NODE_GAP_X, y: SHOT_TOP + row * NODE_GAP_Y },
            width: 340,
            height: 240,
            metadata: meta as CanvasNodeMetadata,
        });
        shotNodeOf.set(shot.id, id);
    });

    // ── 连线：参考素材→镜头 / 首帧→镜头 / 上一镜→本镜 ──
    shots.forEach((shot) => {
        const shotNodeId = shotNodeOf.get(shot.id);
        if (!shotNodeId) return;
        (shot.reference_asset_ids || []).forEach((aid) => {
            const nid = assetNodeOf.get(aid);
            if (nid) connections.push({ id: nanoid(), fromNodeId: nid, toNodeId: shotNodeId });
        });
        if (shot.start_frame_asset_id) {
            const nid = assetNodeOf.get(shot.start_frame_asset_id);
            if (nid) connections.push({ id: nanoid(), fromNodeId: nid, toNodeId: shotNodeId });
        }
        if (shot.continuity_from_shot_id) {
            const pid = shotNodeOf.get(shot.continuity_from_shot_id);
            if (pid) connections.push({ id: nanoid(), fromNodeId: pid, toNodeId: shotNodeId });
        }
    });

    return { title: project.brief?.title || project.id, nodes, connections };
}

// ── M10 反向映射：画布节点 → nautilus ShotUpdate 载荷 ──
import type { ShotUpdatePayload } from "@/services/api/nautilus";

export function canvasToShotPatch(node: CanvasNodeData): ShotUpdatePayload {
    const meta = (node.metadata || {}) as NautilusShotMetadata;
    const patch: ShotUpdatePayload = {};

    if (node.title && node.title !== meta.shotTitle) patch.title = node.title;
    if (typeof meta.duration === "number") patch.duration_seconds = meta.duration;
    if (meta.task === "fl2va" || meta.task === "ref2va") patch.task = meta.task;
    if (meta.transitionKind) patch.transition_kind = meta.transitionKind;
    if (typeof meta.prompt === "string" && meta.prompt.trim()) patch.prompt = meta.prompt;
    if (meta.camera) patch.camera = meta.camera;
    if (meta.audioPrompt) patch.audio_prompt = meta.audioPrompt;
    if (meta.musicPrompt) patch.music_prompt = meta.musicPrompt;
    if (meta.purpose) patch.purpose = meta.purpose;
    if (meta.negativePrompt) patch.negative_prompt = meta.negativePrompt;
    if (meta.openingState) patch.opening_state = meta.openingState;
    if (meta.endingState) patch.ending_state = meta.endingState;
    if (meta.continuityHandoff) patch.continuity_handoff = meta.continuityHandoff;
    if (meta.hook) patch.hook = meta.hook;
    if (meta.dialogue?.length) patch.dialogue = meta.dialogue;
    if (meta.visualBeats?.length) patch.visual_beats = meta.visualBeats;
    if (meta.fps) patch.fps = meta.fps;
    if (meta.inferenceSteps) patch.inference_steps = meta.inferenceSteps;
    if (meta.flowShift) patch.flow_shift = meta.flowShift;
    if (typeof meta.seed === "number") patch.seed = meta.seed;
    if (Array.isArray(meta.references) && meta.references.length) patch.reference_asset_ids = meta.references.filter(Boolean);
    if (meta.continuityFromShotId) patch.continuity_from_shot_id = meta.continuityFromShotId;
    if (meta.startFrameAssetId) patch.start_frame_asset_id = meta.startFrameAssetId;
    return patch;
}
