// Nautilus 渲染控制台（M11）：浮动于画布右上角
// 能力：生成预制片计划 / 确认 / 触发顺序渲染 / 进度轮询 → 驱动分镜节点徽章。

import { useEffect, useMemo, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

import type { CanvasNodeData, CanvasNodeMetadata } from "@/types/canvas";
import type { NautilusPreproductionPlan, NautilusRenderJob } from "@/services/api/nautilus";
import { approvePreproduction, createPreproduction, getActiveJobs, getJob, getPreproduction, startRender } from "@/services/api/nautilus";

type Props = {
    nodes: CanvasNodeData[];
    setNodes: Dispatch<SetStateAction<CanvasNodeData[]>>;
};

const PLAN_STATUS_LABEL: Record<string, string> = {
    draft: "草稿",
    awaiting_approval: "待确认",
    approved: "已确认",
    generating_assets: "补齐首帧中",
    ready: "就绪（可渲染）",
    blocked: "被阻断",
};

function applyJobToNodes(nodes: CanvasNodeData[], job: NautilusRenderJob): CanvasNodeData[] {
    const shotStat = new Map<string, string>();
    if (job.status === "complete") {
        for (const n of nodes) {
            const m = n.metadata as { shotId?: string };
            if (m.shotId) shotStat.set(m.shotId, "complete");
        }
    } else if (job.status === "failed") {
        for (const n of nodes) {
            const m = n.metadata as { shotId?: string };
            if (m.shotId) shotStat.set(m.shotId, "failed");
        }
    }
    if (job.current_shot_id && job.status === "running") {
        shotStat.set(job.current_shot_id, "rendering");
    }
    return nodes.map((node) => {
        const m = node.metadata as { shotId?: string; shotStatus?: string; status?: string };
        if (!m.shotId) return node;
        const next = shotStat.get(m.shotId);
        if (!next || next === m.shotStatus) return node;
        const canvasStatus: CanvasNodeMetadata["status"] = next === "complete" ? "success" : next === "rendering" ? "loading" : next === "failed" ? "error" : "idle";
        return {
            ...node,
            metadata: { ...node.metadata, shotStatus: next, status: canvasStatus } as CanvasNodeMetadata,
        };
    });
}

export function NautilusConsole({ nodes, setNodes }: Props) {
    const shotNodes = useMemo(() => nodes.filter((n) => n.type === "nautilus:shot"), [nodes]);
    const singleShotNode = shotNodes.length === 1 ? shotNodes[0] : null;
    const shotMeta = useMemo(() => {
        const shot = shotNodes[0];
        return (shot?.metadata || {}) as { projectId?: string; shotId?: string; shotIndex?: number };
    }, [shotNodes]);
    const projectId = shotMeta.projectId;
    const singleShotId = singleShotNode ? (singleShotNode.metadata as { shotId?: string }).shotId : undefined;
    const singleShotTitle = singleShotNode?.title || "";
    const isSingleShotMode = Boolean(singleShotId);

    const [plan, setPlan] = useState<NautilusPreproductionPlan | null>(null);
    const [job, setJob] = useState<NautilusRenderJob | null>(null);
    const [busy, setBusy] = useState(false);
    const [force, setForce] = useState(false);

    // 初始加载：读取已存在的计划与活跃任务
    useEffect(() => {
        if (!projectId) return;
        let alive = true;
        getPreproduction(projectId)
            .then((p) => alive && setPlan(p))
            .catch(() => undefined);
        getActiveJobs()
            .then((jobs) => {
                if (!alive) return;
                const mine = jobs.find((j) => j.project_id === projectId);
                if (mine) setJob(mine);
            })
            .catch(() => undefined);
        return () => {
            alive = false;
        };
    }, [projectId]);

    // 渲染进度轮询（job queued/running 时 500ms）
    useEffect(() => {
        if (!projectId || !job || (job.status !== "queued" && job.status !== "running")) return;
        const timer = window.setInterval(() => {
            getJob(job.id)
                .then((latest) => {
                    setJob(latest);
                    setNodes((prev) => applyJobToNodes(prev, latest));
                    if (latest.status === "complete" || latest.status === "failed") {
                        window.clearInterval(timer);
                    }
                })
                .catch(() => undefined);
        }, 500);
        return () => window.clearInterval(timer);
    }, [job?.id, job?.status, projectId, setNodes]);

    if (!projectId) return null;

    const generatePlan = async () => {
        setBusy(true);
        try {
            setPlan(await createPreproduction(projectId));
        } catch (e) {
            setPlan((prev) => prev);
            alert("生成预制片失败: " + (e instanceof Error ? e.message : String(e)));
        } finally {
            setBusy(false);
        }
    };

    const approve = async () => {
        setBusy(true);
        try {
            setPlan(await approvePreproduction(projectId));
        } catch (e) {
            alert("确认失败: " + (e instanceof Error ? e.message : String(e)));
        } finally {
            setBusy(false);
        }
    };

    const render = async () => {
        setBusy(true);
        try {
            // 单镜模式必须 force=true：只清目标镜并重生成，runner 已保证不清其余 69 镜/不组装 final.mp4。
            const started = await startRender(projectId, isSingleShotMode ? true : force, singleShotId);
            setJob(started);
            setNodes((prev) => applyJobToNodes(prev, started));
        } catch (e) {
            alert("发起渲染失败: " + (e instanceof Error ? e.message : String(e)));
        } finally {
            setBusy(false);
        }
    };

    const replaceBackToNautilus = () => {
        // 单镜 job 已写回 selected_take_path/status；返回主流程后，普通 render 会复用新镜并重拼 final.mp4。
        window.location.assign(`http://127.0.0.1:7860/?project=${encodeURIComponent(projectId)}${singleShotId ? `&shot=${encodeURIComponent(singleShotId)}` : ""}`);
    };

    const jobActive = job && (job.status === "queued" || job.status === "running");

    const btnCls = "rounded bg-indigo-600 px-3 py-1.5 text-sm text-white hover:bg-indigo-700 disabled:opacity-40";
    const ghostCls = "rounded border border-stone-300 bg-white px-3 py-1.5 text-sm text-stone-800 hover:bg-stone-100 disabled:opacity-40 dark:border-stone-600 dark:bg-stone-700 dark:text-stone-100";

    return (
        <div className="absolute right-3 top-14 z-50 w-80 rounded-lg border border-stone-200 bg-white p-3 text-sm shadow-lg dark:border-stone-700 dark:bg-stone-900">
            <div className="mb-2 flex items-center justify-between">
                <span className="font-semibold text-stone-900 dark:text-stone-50">🎬 Nautilus 渲染控制台</span>
                <span className="text-[10px] text-stone-500 dark:text-stone-400">{projectId.slice(0, 14)}</span>
            </div>

            {/* 预制片计划区 */}
            {!plan ? (
                <div className="flex flex-col gap-2">
                    {isSingleShotMode ? (
                        <>
                            <p className="text-stone-700 dark:text-stone-200">单镜调整模式：第{shotMeta.shotIndex !== undefined ? shotMeta.shotIndex + 1 : ""}镜 · {singleShotTitle}</p>
                            <p className="text-xs text-stone-600 dark:text-stone-300">保存参数后，点击下方按钮只重生成这一镜，不影响其余镜头。</p>
                            <button className={btnCls} disabled={busy} onClick={() => void render()}>{busy ? "处理中…" : "重新生成该镜头"}</button>
                        </>
                    ) : (
                        <>
                            <p className="text-stone-700 dark:text-stone-200">尚未生成预制片计划。</p>
                            <button className={btnCls} disabled={busy} onClick={() => void generatePlan()}>{busy ? "处理中…" : "生成预制片计划"}</button>
                        </>
                    )}
                </div>
            ) : (
                <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                        <span className="text-stone-700 dark:text-stone-200">计划状态</span>
                        <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs text-indigo-800">{PLAN_STATUS_LABEL[plan.status] || plan.status}</span>
                    </div>
                    {plan.blockers?.length ? (
                        <ul className="list-inside list-disc text-xs text-red-700 dark:text-red-300">
                            {plan.blockers.map((b, i) => <li key={i}>{b}</li>)}
                        </ul>
                    ) : null}
                    {plan.status === "awaiting_approval" ? (
                        <button className={btnCls} disabled={busy || (plan.blockers?.length || 0) > 0} onClick={() => void approve()}>
                            {busy ? "处理中…" : "确认计划" + (plan.generated_image_count ? `（需补 ${plan.generated_image_count} 张首帧）` : "")}
                        </button>
                    ) : plan.status === "ready" || plan.status === "approved" ? (
                        <div className="flex flex-col gap-2">
                            {isSingleShotMode ? (
                                <p className="text-xs text-stone-600 dark:text-stone-300">单镜模式会强制重生成当前镜头，但不会清除其他镜头或 final.mp4。</p>
                            ) : (
                                <label className="flex items-center gap-2 text-xs text-stone-700 dark:text-stone-200">
                                    <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                                    强制重渲全部镜头
                                </label>
                            )}
                            <button className={btnCls} disabled={busy || (plan.status !== "ready" && plan.status !== "approved")} onClick={() => void render()}>
                                {busy ? "处理中…" : isSingleShotMode ? "重新生成该镜头" : "开始顺序渲染"}
                            </button>
                        </div>
                    ) : null}
                </div>
            )}

            {/* 渲染进度区 */}
            {jobActive ? (
                <div className="mt-3 border-t border-stone-200 pt-3 dark:border-stone-700">
                    <div className="mb-1 flex justify-between text-xs text-stone-700 dark:text-stone-200">
                        <span>渲染进度</span>
                        <span>{Math.round(job.progress * 100)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded bg-stone-200 dark:bg-stone-700">
                        <div className="h-full bg-indigo-500 transition-all" style={{ width: `${Math.round(job.progress * 100)}%` }} />
                    </div>
                    <p className="mt-1 truncate text-xs text-stone-500 dark:text-stone-400">{job.message || "排队中…"}</p>
                </div>
            ) : job?.status === "complete" ? (
                <div className="mt-3 flex flex-col gap-2 border-t border-stone-200 pt-2 text-sm text-emerald-700 dark:border-stone-700 dark:text-emerald-300">
                    <span>✅ {isSingleShotMode ? "该镜头已重生成" : "渲染完成"}</span>
                    {isSingleShotMode && job.id ? (
                        <video className="max-h-40 w-full rounded bg-black" controls src={`/api/jobs/${job.id}/output`} />
                    ) : null}
                    {isSingleShotMode ? (
                        <button className={ghostCls} onClick={replaceBackToNautilus}>替换回 Nautilus 主流程</button>
                    ) : null}
                </div>
            ) : job?.status === "failed" ? (
                <div className="mt-3 border-t border-stone-200 pt-2 text-xs text-red-700 dark:border-stone-700 dark:text-red-300">❌ 渲染失败：{job.error || job.message}</div>
            ) : null}
        </div>
    );
}
