// 单镜头调整工作流的数据层（画布侧）
// 从 nautilus 项目只导入**一个镜头** + 其关联素材，避免 70 镜全量导入的卡顿。

import type { NautilusProject, NautilusShot } from "@/services/api/nautilus";
import { getProject, listAssets } from "@/services/api/nautilus";
import { projectToCanvas, type ProjectToCanvasResult } from "@/lib/canvas/nautilus-bridge";

export interface SingleShotLoad {
    project: NautilusProject;
    shot: NautilusShot;
    canvas: ProjectToCanvasResult;
}

/** 加载某项目中的单个镜头 + 关联素材，映射为画布节点（镜头 1 个居中 + 相关素材）。 */
export async function loadSingleShotToCanvas(projectId: string, shotId: string): Promise<SingleShotLoad> {
    const project = await getProject(projectId);
    const shot = project.shots.find((s) => s.id === shotId);
    if (!shot) throw new Error(`shot ${shotId} not found in project ${projectId}`);

    const assets = await listAssets();
    const wanted = new Set<string>();
    for (const id of [...(shot.reference_asset_ids || []), shot.start_frame_asset_id, shot.audio_asset_id]) {
        if (id) wanted.add(id);
    }
    const relevant = assets.filter((a) => wanted.has(a.id));

    const miniProject: NautilusProject = { ...project, shots: [shot] };
    const canvas = projectToCanvas(miniProject, relevant);
    // 单镜头调整：镜头节点默认置中，参考素材在左侧（重排布局更聚焦）
    const shotNode = canvas.nodes.find((n) => n.type === "nautilus:shot");
    if (shotNode) {
        shotNode.position = { x: 460, y: 220 };
        shotNode.width = 420;
        shotNode.height = 300;
        const assetNodes = canvas.nodes.filter((n) => n.type !== "nautilus:shot" && n.type !== "nautilus:character");
        assetNodes.forEach((n, i) => (n.position = { x: 40, y: 80 + i * 260 }));
        const charNodes = canvas.nodes.filter((n) => n.type === "nautilus:character");
        charNodes.forEach((n, i) => (n.position = { x: 40, y: 80 + assetNodes.length * 260 + i * 160 }));
    }
    return { project, shot, canvas };
}

