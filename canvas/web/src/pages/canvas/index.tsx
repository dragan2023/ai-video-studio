import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { App, Button, List, Modal, Spin } from "antd";
import { Database, Download, FileUp, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";

import { readZip } from "@/lib/zip";
import { setMediaBlob } from "@/services/file-storage";
import { setImageBlob } from "@/services/image-storage";
import { CanvasDeleteProjectsDialog } from "@/components/canvas/canvas-delete-projects-dialog";
import { CanvasProjectCard } from "@/components/canvas/canvas-project-card";
import type { CanvasExportFile } from "@/types/canvas-export";
import { useCanvasStore } from "@/stores/canvas/use-canvas-store";
import { useCanvasUiStore } from "@/stores/canvas/use-canvas-ui-store";
import { listProjects, type NautilusProject } from "@/services/api/nautilus";
import { loadSingleShotToCanvas } from "@/lib/canvas/nautilus-single-shot";
import { registerNautilusNodes } from "@/lib/canvas/nautilus-node-specs";

// 注册 nautilus:shot / nautilus:character 节点（全局单例，幂等）
registerNautilusNodes();
import { exportCanvasProjects } from "@/lib/canvas/canvas-export";

export default function CanvasPage() {
    const { message } = App.useApp();
    const { t } = useTranslation();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const inputRef = useRef<HTMLInputElement>(null);
    const autoOpenRef = useRef(false);
    const hydrated = useCanvasStore((state) => state.hydrated);
    const projects = useCanvasStore((state) => state.projects);
    const createProject = useCanvasStore((state) => state.createProject);
    const importProject = useCanvasStore((state) => state.importProject);
    const selectedIds = useCanvasUiStore((state) => state.selectedProjectIds);
    const setDeleteIds = useCanvasUiStore((state) => state.setDeleteProjectIds);

    const mode = searchParams.get("mode");
    const agentMode = mode === "new" || mode === "recent" || mode === "choose";
    const agentQuery = agentMode ? `?${searchParams.toString()}` : "";
    const enterProject = (id: string) => {
        navigate(`/canvas/${id}${agentQuery}`);
    };
    const createAndEnter = () => enterProject(createProject(t("canvas.defaultTitle", { count: projects.length + 1 })));
    const importCanvas = async (file?: File) => {
        if (!file) return;
        try {
            const zip = await readZip(file);
            const projectFile = zip.get("projects.json");
            if (!projectFile) throw new Error("missing projects.json");
            const data = JSON.parse(await projectFile.text()) as CanvasExportFile;
            await Promise.all(
                data.projects.flatMap((project) =>
                    project.files.map(async (item) => {
                        const blob = zip.get(item.path);
                        if (!blob) return;
                        const typedBlob = blob.type ? blob : blob.slice(0, blob.size, item.mimeType);
                        await (item.storageKey.startsWith("image:") ? setImageBlob(item.storageKey, typedBlob) : setMediaBlob(item.storageKey, typedBlob));
                    }),
                ),
            );
            data.projects.forEach((item) => importProject(item.project));
            message.success(t("canvas.imported", { count: data.projects.length }));
        } catch {
            message.error(t("canvas.importFailed"));
        } finally {
            if (inputRef.current) inputRef.current.value = "";
        }
    };

    // ── Nautilus 项目导入入口（M9 只读桥接）──
    const [nautilusOpen, setNautilusOpen] = useState(false);
    const [nautilusProjects, setNautilusProjects] = useState<NautilusProject[] | null>(null);
    const [nautilusSelectedProject, setNautilusSelectedProject] = useState<NautilusProject | null>(null);
    const [nautilusBusy, setNautilusBusy] = useState(false);
    const nautilusDeepLinkRef = useRef(false);

    const openNautilusPicker = async () => {
        setNautilusSelectedProject(null);
        setNautilusOpen(true);
        setNautilusBusy(true);
        try {
            setNautilusProjects(await listProjects());
        } catch (error) {
            message.error(error instanceof Error ? error.message : t("canvas.importFailed"));
        } finally {
            setNautilusBusy(false);
        }
    };

    const closeNautilusPicker = () => {
        setNautilusOpen(false);
        setNautilusSelectedProject(null);
    };

    /** 12b：只导入一个镜头 + 关联参考素材，绝不把 70 镜全量塞进画布。 */
    const loadNautilusShot = async (projectId: string, shotId: string) => {
        setNautilusBusy(true);
        try {
            const loaded = await loadSingleShotToCanvas(projectId, shotId);
            const canvasId = importProject({
                title: `${loaded.project.brief.title} · 第${loaded.shot.index + 1}镜 ${loaded.shot.title}（单镜调整）`,
                nodes: loaded.canvas.nodes,
                connections: loaded.canvas.connections,
            });
            closeNautilusPicker();
            navigate(`/canvas/${canvasId}`);
        } catch (error) {
            message.error(error instanceof Error ? error.message : t("canvas.importFailed"));
        } finally {
            setNautilusBusy(false);
        }
    };

    useEffect(() => {
        if (!hydrated || autoOpenRef.current || (mode !== "new" && mode !== "recent")) return;
        autoOpenRef.current = true;
        enterProject(mode === "new" ? createProject(t("canvas.defaultTitle", { count: projects.length + 1 })) : projects[0]?.id || createProject(t("canvas.defaultTitle", { count: projects.length + 1 })));
    }, [createProject, hydrated, mode, projects, t]);

    // 12c 深度链接：Nautilus 镜头行可跳 /canvas?nautilus-project=<pid>&shot=<sid>，直接进入单镜工作台。
    const linkedProjectId = searchParams.get("nautilus-project");
    const linkedShotId = searchParams.get("shot");
    useEffect(() => {
        if (!hydrated || !linkedProjectId || !linkedShotId || nautilusDeepLinkRef.current) return;
        nautilusDeepLinkRef.current = true;
        void loadNautilusShot(linkedProjectId, linkedShotId);
    }, [hydrated, linkedProjectId, linkedShotId]);

    if (hydrated && (mode === "new" || mode === "recent")) return <main className="flex h-full items-center justify-center bg-background text-sm text-stone-500">{t("canvas.opening")}</main>;

    return (
        <main className="h-full overflow-auto bg-background text-stone-950 dark:text-stone-100">
            <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 py-10">
                <header className="flex flex-wrap items-end justify-between gap-4 border-b border-stone-200 pb-6 dark:border-stone-800">
                    <div>
                        <p className="text-xs text-stone-500">{t("canvas.library")}</p>
                        <h1 className="mt-3 text-3xl font-semibold">{t("canvas.title")}</h1>
                    </div>
                    <div className="flex items-center gap-2">
                        {selectedIds.length ? (
                            <>
                                <Button disabled={!hydrated} icon={<Download className="size-4" />} onClick={() => void exportCanvasProjects(projects.filter((project) => selectedIds.includes(project.id)), `${t("canvas.title")}-${selectedIds.length}`)}>
                                    {t("canvas.exportSelected")}
                                </Button>
                                <Button disabled={!hydrated} onClick={() => setDeleteIds(selectedIds)}>
                                    {t("canvas.deleteSelected")}
                                </Button>
                            </>
                        ) : null}
                        {projects.length ? (
                            <Button disabled={!hydrated} onClick={() => setDeleteIds(projects.map((project) => project.id))}>
                                {t("canvas.deleteAll")}
                            </Button>
                        ) : null}
                        <Button disabled={!hydrated} icon={<FileUp className="size-4" />} onClick={() => inputRef.current?.click()}>
                            {t("canvas.import")}
                        </Button>
                        <Button disabled={!hydrated} icon={<Database className="size-4" />} onClick={() => void openNautilusPicker()}>
                            Nautilus 项目
                        </Button>
                        <Button disabled={!hydrated} type="primary" icon={<Plus className="size-4" />} onClick={createAndEnter}>
                            {t("canvas.create")}
                        </Button>
                    </div>
                </header>

                {!hydrated ? (
                    <section className="flex min-h-[360px] items-center justify-center border-y border-stone-200 text-sm text-stone-500 dark:border-stone-800">{t("canvas.loading")}</section>
                ) : projects.length ? (
                    <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
                        {projects.map((project) => (
                            <CanvasProjectCard key={project.id} project={project} />
                        ))}
                    </div>
                ) : (
                    <section className="flex min-h-[360px] flex-col items-center justify-center border-y border-stone-200 text-center dark:border-stone-800">
                        <h2 className="text-xl font-medium">{t("canvas.empty")}</h2>
                        <p className="mt-3 text-sm text-stone-500">{t("canvas.emptyDescription")}</p>
                        <Button type="primary" className="mt-6" icon={<Plus className="size-4" />} onClick={createAndEnter}>
                            {t("canvas.create")}
                        </Button>
                    </section>
                )}
            </div>

            <input ref={inputRef} type="file" accept="application/zip,.zip" className="hidden" onChange={(event) => void importCanvas(event.target.files?.[0])} />
            <CanvasDeleteProjectsDialog />

            <Modal
                title={nautilusSelectedProject ? `选择镜头 · ${nautilusSelectedProject.brief?.title || nautilusSelectedProject.id}` : "选择 Nautilus 项目"}
                open={nautilusOpen}
                onCancel={closeNautilusPicker}
                footer={null}
                width={640}
            >
                {nautilusBusy ? (
                    <div className="flex items-center justify-center py-10">
                        <Spin />
                    </div>
                ) : nautilusSelectedProject ? (
                    <div className="flex flex-col gap-3">
                        <Button onClick={() => setNautilusSelectedProject(null)}>← 返回项目列表</Button>
                        <p className="text-sm text-stone-700">单镜调整模式：只导入所选镜头和关联参考素材，不导入 70 镜全量画布。</p>
                        <List
                            size="small"
                            dataSource={[...(nautilusSelectedProject.shots || [])].sort((a, b) => a.index - b.index)}
                            locale={{ emptyText: "该项目没有镜头" }}
                            renderItem={(shot) => (
                                <List.Item className="cursor-pointer rounded px-2 hover:bg-stone-100" onClick={() => void loadNautilusShot(nautilusSelectedProject.id, shot.id)}>
                                    <List.Item.Meta
                                        title={`第${shot.index + 1}镜 · ${shot.title}`}
                                        description={`${shot.duration_seconds}s · ${shot.task} · ${shot.status}`}
                                    />
                                </List.Item>
                            )}
                        />
                    </div>
                ) : (
                    <List
                        dataSource={nautilusProjects || []}
                        locale={{ emptyText: "无 Nautilus 项目（请先启动 nautilus 服务）" }}
                        renderItem={(project) => (
                            <List.Item className="cursor-pointer rounded px-2 hover:bg-stone-100" onClick={() => setNautilusSelectedProject(project)}>
                                <List.Item.Meta
                                    title={project.brief?.title || project.id}
                                    description={`🔢 ${project.shots?.length || 0} 镜 · ${project.status} · 点击后选择单个镜头`}
                                />
                            </List.Item>
                        )}
                    />
                )}
            </Modal>
        </main>
    );
}
