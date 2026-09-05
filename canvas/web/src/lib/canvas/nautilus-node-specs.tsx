// Nautilus 自定义节点注册（M9 只读 → M10 可编辑）
// 节点类型：nautilus:shot（分镜，可编辑面板）、nautilus:character（角色，只读）

import { useState } from "react";
import { message } from "antd";
import type { ComponentType } from "react";

import type { CanvasNodeContext, CanvasNodeDefinition } from "@/types/canvas-plugin";
import type { CanvasNodeMetadata } from "@/types/canvas";
import { registerNodeDefinitions } from "@/lib/canvas/node-registry";
import type { NautilusCharacterMetadata, NautilusShotMetadata } from "@/lib/canvas/nautilus-bridge";
import { reorderShotBySwap, saveShotNode, setShotReferences, type Notify } from "@/lib/canvas/nautilus-sync";

const STATUS_BADGE: Record<string, string> = {
    planned: "bg-stone-100 text-stone-600",
    ready: "bg-sky-100 text-sky-700",
    rendering: "bg-amber-100 text-amber-700",
    complete: "bg-emerald-100 text-emerald-700",
    failed: "bg-red-100 text-red-700",
    idle: "bg-stone-100 text-stone-700 dark:text-stone-200",
    success: "bg-emerald-100 text-emerald-700",
    loading: "bg-amber-100 text-amber-700",
    error: "bg-red-100 text-red-700",
};

const notify: Notify = (text, type) => {
    if (type === "error") message.error(text);
    else if (type === "success") message.success(text);
    else message.info(text);
};

const TASK_OPTIONS: { value: "fl2va" | "ref2va"; label: string }[] = [
    { value: "fl2va", label: "FL2VA（文生视频）" },
    { value: "ref2va", label: "REF2VA（视频续写）" },
];
const TRANSITION_OPTIONS: { value: string; label: string }[] = [
    { value: "continuous", label: "连续承接" },
    { value: "camera_move", label: "运镜转场" },
    { value: "match_cut", label: "匹配剪辑" },
    { value: "occlusion_cut", label: "遮挡剪辑" },
    { value: "hard_cut", label: "硬切" },
    { value: "anchor", label: "锚点定场" },
];
const TASK_LABEL: Record<string, string> = { fl2va: "FL2VA（文生视频）", ref2va: "REF2VA（视频续写）" };
const FRAME_SOURCE_LABEL: Record<string, string> = {
    creator_asset: "用户素材",
    previous_boundary: "上一镜末帧",
    system_black: "系统黑场",
    generate_t2i: "需自动补图",
    needs_review: "待人工复核",
};
const FRAME_SOURCE_COLOR: Record<string, string> = {
    creator_asset: "bg-emerald-100 text-emerald-800",
    previous_boundary: "bg-sky-100 text-sky-800",
    system_black: "bg-stone-200 text-stone-700",
    generate_t2i: "bg-amber-100 text-amber-800",
    needs_review: "bg-red-100 text-red-800",
};

const ShotNodeContent: ComponentType<{ ctx: CanvasNodeContext }> = ({ ctx }) => {
    const meta = (ctx.node.metadata || {}) as NautilusShotMetadata;
    const status = meta.shotStatus || meta.status || "planned";
    return (
        <div className="flex h-full w-full flex-col gap-1 overflow-hidden p-2 text-xs">
            <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-semibold">{ctx.node.title}</span>
                <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${STATUS_BADGE[status] || STATUS_BADGE.planned}`}>{status}</span>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-stone-700 dark:text-stone-200">
                {typeof meta.shotIndex === "number" ? <span>#{meta.shotIndex + 1}</span> : null}
                {typeof meta.duration === "number" ? <span>⏱ {meta.duration}s</span> : null}
                {meta.task ? <span>🎬 {TASK_LABEL[meta.task] || meta.task}</span> : null}
                {meta.references?.length ? <span>📎 {meta.references.length} 参考</span> : null}
                {meta.preproduction?.startFrameSource ? (
                    <span className={`rounded px-1 py-px ${FRAME_SOURCE_COLOR[meta.preproduction.startFrameSource] || "bg-stone-100 text-stone-700"}`}>
                        🎭 {FRAME_SOURCE_LABEL[meta.preproduction.startFrameSource] || meta.preproduction.startFrameSource}
                    </span>
                ) : null}
                {typeof meta.preproduction?.confidence === "number" ? (
                    <span className="rounded bg-indigo-100 px-1 py-px text-indigo-800">🎯 {(meta.preproduction.confidence * 100).toFixed(0)}%</span>
                ) : null}
                {meta.preproduction?.gapReason ? <span className="truncate text-red-700 dark:text-red-300">⚠ {meta.preproduction.gapReason}</span> : null}
            </div>
            <p className="line-clamp-3 whitespace-pre-wrap text-stone-700 dark:text-stone-200">{meta.prompt || "（无镜头描述）"}</p>
        </div>
    );
};

const ShotNodePanel: ComponentType<{ ctx: CanvasNodeContext; onClose: () => void }> = ({ ctx, onClose }) => {
    const meta = (ctx.node.metadata || {}) as NautilusShotMetadata;
    const [title, setTitle] = useState(ctx.node.title || "");
    const [purpose, setPurpose] = useState(meta.purpose || "");
    const [duration, setDuration] = useState(String(meta.duration ?? 5));
    const [task, setTask] = useState(meta.task || "fl2va");
    const [transition, setTransition] = useState(meta.transitionKind || "continuous");
    const [prompt, setPrompt] = useState(meta.prompt || "");
    const [camera, setCamera] = useState(meta.camera || "");
    const [audioPrompt, setAudioPrompt] = useState(meta.audioPrompt || "");
    const [negativePrompt, setNegativePrompt] = useState(meta.negativePrompt || "");

    const assetNodes = ctx.getNodes().filter((n) => (n.metadata as { assetId?: string }).assetId);
    const selectedRefs = new Set(meta.references || []);

    const handleSave = async () => {
        const durationNum = Number(duration);
        if (!Number.isFinite(durationNum) || durationNum < 4 || durationNum > 15) {
            message.error("时长需在 4–15 秒之间（H3 约束）");
            return;
        }
        ctx.updateNode({ title: title.trim() || ctx.node.title });
        ctx.updateMetadata({
            prompt,
            duration: durationNum,
            task,
            transitionKind: transition,
            purpose,
            camera,
            audioPrompt,
            negativePrompt,
        } as CanvasNodeMetadata);
        await saveShotNode(ctx, ctx.node.id, notify);
    };

    const handleMove = (dir: -1 | 1) => {
        void reorderShotBySwap(ctx, ctx.node.id, dir, notify);
    };

    const fieldCls = "w-full rounded border border-stone-300 bg-white px-2 py-1 text-sm text-stone-900 outline-none focus:border-indigo-500 dark:border-stone-600 dark:bg-stone-800 dark:text-stone-100";

    return (
        <div className="flex w-[320px] flex-col gap-2 rounded-lg border border-stone-200 bg-white p-3 text-sm shadow-lg dark:border-stone-700 dark:bg-stone-900">
            <div className="flex items-center justify-between">
                <span className="font-medium">{ctx.node.title}</span>
                <span className="flex gap-1">
                    <button type="button" className="rounded border border-stone-300 bg-white px-2 py-0.5 text-stone-800 hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-700 dark:text-stone-100" onClick={() => handleMove(-1)}>↑上移</button>
                    <button type="button" className="rounded border border-stone-300 bg-white px-2 py-0.5 text-stone-800 hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-700 dark:text-stone-100" onClick={() => handleMove(1)}>↓下移</button>
                    <button type="button" className="rounded border border-stone-300 bg-white px-2 py-0.5 text-stone-800 hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-700 dark:text-stone-100" onClick={onClose}>✕</button>
                </span>
            </div>

            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">标题</span>
                <input className={fieldCls} value={title} onChange={(e) => setTitle(e.target.value)} />
            </label>

            <div className="grid grid-cols-3 gap-2">
                <label className="flex flex-col gap-1">
                    <span className="text-stone-700 dark:text-stone-200">时长(s)</span>
                    <input className={fieldCls} type="number" min={4} max={15} value={duration} onChange={(e) => setDuration(e.target.value)} />
                </label>
                <label className="flex flex-col gap-1">
                    <span className="text-stone-700 dark:text-stone-200">任务</span>
                    <select className={fieldCls} value={task} onChange={(e) => setTask(e.target.value)}>
                        {TASK_OPTIONS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                    </select>
                </label>
                <label className="flex flex-col gap-1">
                    <span className="text-stone-700 dark:text-stone-200">转场</span>
                    <select className={fieldCls} value={transition} onChange={(e) => setTransition(e.target.value)}>
                        {TRANSITION_OPTIONS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                    </select>
                </label>
            </div>

            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">镜头描述 prompt</span>
                <textarea className={`${fieldCls} min-h-[72px] resize-y`} rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </label>

            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">运镜 camera</span>
                <input className={fieldCls} value={camera} onChange={(e) => setCamera(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">音频提示 audio_prompt</span>
                <input className={fieldCls} value={audioPrompt} onChange={(e) => setAudioPrompt(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">反向提示 negative_prompt</span>
                <input className={fieldCls} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">用途 purpose</span>
                <input className={fieldCls} value={purpose} onChange={(e) => setPurpose(e.target.value)} />
            </label>

            <div className="flex flex-col gap-1">
                <span className="text-stone-700 dark:text-stone-200">参考素材（连线 = 生成参考）</span>
                <div className="max-h-40 overflow-auto rounded border border-stone-300 bg-stone-50 p-1 dark:border-stone-600 dark:bg-stone-800">
                    {assetNodes.length === 0 ? (
                        <span className="text-stone-600 dark:text-stone-300">（本项目无素材节点）</span>
                    ) : (
                        assetNodes.map((n) => {
                            const assetId = (n.metadata as { assetId?: string }).assetId as string;
                            const checked = selectedRefs.has(assetId);
                            return (
                                <label key={n.id} className="flex items-center gap-2 rounded px-1 py-0.5 text-stone-800 hover:bg-stone-100 dark:text-stone-100 dark:hover:bg-stone-700">
                                    <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={(e) => {
                                            const next = new Set(selectedRefs);
                                            if (e.target.checked) next.add(assetId);
                                            else next.delete(assetId);
                                            setShotReferences(ctx, ctx.node.id, [...next]);
                                        }}
                                    />
                                    <span className="truncate">{n.title || assetId}</span>
                                </label>
                            );
                        })
                    )}
                </div>
            </div>

            <button type="button" className="mt-1 rounded bg-indigo-600 px-3 py-1.5 text-white hover:bg-indigo-700" onClick={() => void handleSave()}>
                保存到 nautilus
            </button>
        </div>
    );
};

const CharacterNodeContent: ComponentType<{ ctx: CanvasNodeContext }> = ({ ctx }) => {
    const meta = (ctx.node.metadata || {}) as NautilusCharacterMetadata;
    return (
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 p-2 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-indigo-100 text-lg">👤</div>
            <div className="text-sm font-semibold">{meta.label || ctx.node.title}</div>
            <div className="max-w-full truncate text-[10px] text-stone-600 dark:text-stone-300">{meta.subjectId || ""}</div>
        </div>
    );
};

export function registerNautilusNodes(): void {
    registerNodeDefinitions(
        [
            {
                type: "nautilus:shot",
                title: "镜头",
                icon: "🎬",
                description: "Nautilus 分镜节点（可编辑，M10）",
                defaultSize: { width: 340, height: 240 },
                defaultMetadata: { status: "idle" },
                minimapColor: "#6366f1",
                Content: ShotNodeContent,
                Panel: ShotNodePanel,
            },
            {
                type: "nautilus:character",
                title: "角色",
                icon: "👤",
                description: "Nautilus 角色节点（只读投影）",
                defaultSize: { width: 200, height: 150 },
                defaultMetadata: {},
                minimapColor: "#f59e0b",
                hidePanel: true,
                Content: CharacterNodeContent,
            },
        ],
        "nautilus",
    );
}

registerNautilusNodes();
