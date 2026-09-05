import { useEffect, useState } from "react";
import { App, Input, Modal, Select } from "antd";

import { getMediaBlob } from "@/services/file-storage";
import { getImageBlob } from "@/services/image-storage";
import { uploadAssets, type NautilusAsset, type NautilusAssetRole } from "@/services/api/nautilus";
import { CanvasNodeType, type CanvasNodeData } from "@/types/canvas";

type Props = {
    node: CanvasNodeData | null;
    open: boolean;
    onClose: () => void;
    onSaved: (nodeId: string, asset: NautilusAsset, roles: NautilusAssetRole[]) => void;
};

const ROLE_OPTIONS: { value: NautilusAssetRole; label: string }[] = [
    { value: "character", label: "角色" },
    { value: "location", label: "场景" },
    { value: "prop", label: "道具" },
    { value: "style", label: "风格" },
    { value: "start_frame", label: "首帧" },
    { value: "audio", label: "音频" },
    { value: "reference", label: "参考" },
];

function defaultRoles(node: CanvasNodeData): NautilusAssetRole[] {
    if (node.type === CanvasNodeType.Audio) return ["audio"];
    return ["reference"];
}

function extension(mimeType: string, nodeType: CanvasNodeType) {
    const known: Record<string, string> = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
    };
    return known[mimeType] || (nodeType === CanvasNodeType.Image ? "png" : nodeType === CanvasNodeType.Video ? "mp4" : "mp3");
}

function safeFileName(title: string, fallback: string) {
    const clean = title.replace(/[\\/:*?"<>|]+/g, "-").trim().slice(0, 60);
    return clean || fallback;
}

async function resolveNodeBlob(node: CanvasNodeData): Promise<{ blob: Blob; mimeType: string } | null> {
    const meta = node.metadata;
    if (!meta) return null;
    const primary = meta.images?.find((item) => item.id === meta.primaryImageId) || meta.images?.[0];
    const storageKey = primary?.storageKey || meta.storageKey;
    const content = primary?.content || meta.content;
    const mimeType = primary?.mimeType || meta.mimeType || "";

    let blob: Blob | null = null;
    if (storageKey?.startsWith("image:")) blob = await getImageBlob(storageKey);
    else if (storageKey) blob = await getMediaBlob(storageKey);
    if (!blob && content) blob = await (await fetch(content)).blob();
    if (!blob) return null;
    return { blob, mimeType: blob.type || mimeType };
}

/** 把画布媒体节点上传为 nautilus AssetRecord；本地 storageKey 保留，不影响画布继续预览。 */
export function NautilusAssetSaveDialog({ node, open, onClose, onSaved }: Props) {
    const { message } = App.useApp();
    const [roles, setRoles] = useState<NautilusAssetRole[]>(["reference"]);
    const [tags, setTags] = useState("canvas");
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (!node || !open) return;
        setRoles(defaultRoles(node));
        setTags("canvas");
    }, [node?.id, open]);

    const save = async () => {
        if (!node) return;
        if (!roles.length) {
            message.error("请至少选择一个素材角色");
            return;
        }
        setSaving(true);
        try {
            const resolved = await resolveNodeBlob(node);
            if (!resolved) throw new Error("该节点没有可上传的本地媒体文件");
            const type = resolved.mimeType || "application/octet-stream";
            const file = new File(
                [resolved.blob],
                `${safeFileName(node.title, `canvas-${node.type}-${node.id}`)}.${extension(type, node.type as CanvasNodeType)}`,
                { type },
            );
            const tagList = tags
                .split(/[,，]/)
                .map((item) => item.trim())
                .filter(Boolean);
            const assets = await uploadAssets([file], { roles, tags: tagList });
            const asset = assets[0];
            if (!asset) throw new Error("nautilus 未返回上传素材");
            onSaved(node.id, asset, roles);
            message.success(`已入库：${asset.display_name || asset.original_name}`);
            onClose();
        } catch (error) {
            message.error(`保存到 Nautilus 失败：${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setSaving(false);
        }
    };

    return (
        <Modal
            open={open}
            title="保存到 Nautilus 素材库"
            okText="保存入库"
            cancelText="取消"
            confirmLoading={saving}
            onCancel={onClose}
            onOk={() => void save()}
        >
            <div className="flex flex-col gap-3 text-sm text-stone-800">
                <p className="rounded bg-stone-100 px-3 py-2">节点：{node?.title || "未命名媒体"}</p>
                <label className="flex flex-col gap-1">
                    <span>素材角色</span>
                    <Select
                        mode="multiple"
                        value={roles}
                        options={ROLE_OPTIONS}
                        onChange={(value) => setRoles(value as NautilusAssetRole[])}
                        placeholder="选择素材角色"
                    />
                </label>
                <label className="flex flex-col gap-1">
                    <span>标签（逗号分隔）</span>
                    <Input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="例如：张子兴, 角色设定, 场1" />
                </label>
                <p className="text-xs text-stone-600">上传后会写入 nautilus 共享素材库；画布本地媒体和 storageKey 会保留，不影响继续预览。</p>
            </div>
        </Modal>
    );
}
