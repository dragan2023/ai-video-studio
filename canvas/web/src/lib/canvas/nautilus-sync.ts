// Nautilus 画布同步（M10 双向桥接）
// 保存：画布节点 → canvasToShotPatch → PATCH nautilus → 局部刷新节点状态。
// 参考连线：素材勾选 → 增删 CanvasConnection 与 metadata.references（assetId 列表）。
// 顺序重排：交换相邻节点布局 + PATCH /shots/order。

import { reorderShots, updateShot } from "@/services/api/nautilus";
import { canvasToShotPatch } from "@/lib/canvas/nautilus-bridge";
import type { CanvasAgentOp } from "@/lib/canvas/canvas-agent-ops";
import type { CanvasNodeContext } from "@/types/canvas-plugin";
import type { CanvasNodeMetadata } from "@/types/canvas";

export type Notify = (text: string, type?: "success" | "error" | "info") => void;

interface ShotNodeMeta {
    projectId?: string;
    shotId?: string;
    shotIndex?: number;
    references?: string[];
}

function shotMeta(node: { metadata?: Record<string, unknown> }): ShotNodeMeta {
    return (node.metadata || {}) as ShotNodeMeta;
}

/** 保存单个镜头节点：PATCH nautilus；成功后节点标记 success/planned。 */
export async function saveShotNode(ctx: CanvasNodeContext, nodeId: string, notify?: Notify): Promise<boolean> {
    const node = ctx.getNode(nodeId) || ctx.getNodes().find((n) => n.id === nodeId);
    if (!node) return false;
    const meta = shotMeta(node);
    if (!meta.shotId || !meta.projectId) {
        notify?.("该节点缺少 nautilus 引用（shotId/projectId），无法保存", "error");
        return false;
    }
    const patch = canvasToShotPatch(node);
    if (!Object.keys(patch).length) {
        notify?.("没有可保存的改动", "info");
        return true;
    }
    try {
        ctx.updateMetadata({ status: "loading" } as CanvasNodeMetadata);
        await updateShot(meta.projectId, meta.shotId, patch);
        ctx.updateMetadata({ status: "success", shotStatus: "planned" } as CanvasNodeMetadata);
        notify?.("已保存到 nautilus", "success");
        return true;
    } catch (error) {
        const text = error instanceof Error ? error.message : String(error);
        ctx.updateMetadata({ status: "error", errorDetails: text } as CanvasNodeMetadata);
        notify?.("保存失败: " + text, "error");
        return false;
    }
}

/** 同步镜头参考素材：更新 references（assetId 列表）并增删对应连线。 */
export function setShotReferences(ctx: CanvasNodeContext, shotNodeId: string, assetIds: string[]): void {
    const node = ctx.getNode(shotNodeId) || ctx.getNodes().find((n) => n.id === shotNodeId);
    if (!node) return;
    const assetNodeByAssetId = new Map<string, string>();
    for (const n of ctx.getNodes()) {
        const m = n.metadata as { assetId?: string };
        if (m.assetId) assetNodeByAssetId.set(m.assetId, n.id);
    }
    const current = shotMeta(node).references || [];
    const ops: CanvasAgentOp[] = [];

    for (const conn of ctx.getConnections()) {
        if (conn.toNodeId !== shotNodeId) continue;
        const from = ctx.getNode(conn.fromNodeId);
        const fromAssetId = from?.metadata ? (from.metadata as { assetId?: string }).assetId : undefined;
        if (fromAssetId && !assetIds.includes(fromAssetId)) {
            ops.push({ type: "delete_connections", id: conn.id });
        }
    }
    for (const aid of assetIds) {
        const assetNodeId = assetNodeByAssetId.get(aid);
        if (!assetNodeId || assetNodeId === shotNodeId) continue;
        const exists = ctx.getConnections().some((c) => c.fromNodeId === assetNodeId && c.toNodeId === shotNodeId);
        if (!exists) ops.push({ type: "connect_nodes", fromNodeId: assetNodeId, toNodeId: shotNodeId });
    }
    if (ops.length) ctx.applyOps(ops);
    const changed = JSON.stringify(current) !== JSON.stringify(assetIds);
    if (changed) ctx.updateMetadata({ references: assetIds } as CanvasNodeMetadata);
}

/** 重排相邻镜头（direction=-1 上移 / +1 下移）：先 PATCH 后端，成功后交换布局与 shotIndex。 */
export async function reorderShotBySwap(ctx: CanvasNodeContext, nodeId: string, direction: -1 | 1, notify?: Notify): Promise<boolean> {
    const nodes = ctx.getNodes().filter((n) => n.type === "nautilus:shot");
    const sorted = [...nodes].sort((a, b) => (shotMeta(a).shotIndex ?? 0) - (shotMeta(b).shotIndex ?? 0));
    const idx = sorted.findIndex((n) => n.id === nodeId);
    const target = idx + direction;
    if (idx < 0 || target < 0 || target >= sorted.length) {
        notify?.("已在边界，无法移动", "info");
        return false;
    }
    const cur = sorted[idx];
    const tgt = sorted[target];
    const curMeta = shotMeta(cur);
    const tgtMeta = shotMeta(tgt);
    if (!curMeta.projectId || !curMeta.shotId || !tgtMeta.shotId) return false;

    const ids = sorted.map((n) => shotMeta(n).shotId).filter(Boolean) as string[];
    [ids[idx], ids[target]] = [ids[target], ids[idx]];

    try {
        await reorderShots(curMeta.projectId, ids);
    } catch (error) {
        notify?.("重排失败: " + (error instanceof Error ? error.message : String(error)), "error");
        return false;
    }

    const ops: CanvasAgentOp[] = [
        { type: "update_node", id: cur.id, patch: { position: tgt.position } },
        { type: "update_node", id: tgt.id, patch: { position: cur.position } },
        { type: "update_node", id: cur.id, metadata: { shotIndex: tgtMeta.shotIndex ?? 0 } as CanvasNodeMetadata },
        { type: "update_node", id: tgt.id, metadata: { shotIndex: curMeta.shotIndex ?? 0 } as CanvasNodeMetadata },
        { type: "update_node", id: cur.id, metadata: { status: "success", shotStatus: "planned" } as CanvasNodeMetadata },
        { type: "update_node", id: tgt.id, metadata: { status: "success", shotStatus: "planned" } as CanvasNodeMetadata },
    ];
    ctx.applyOps(ops);
    notify?.("已重排并保存顺序", "success");
    return true;
}

