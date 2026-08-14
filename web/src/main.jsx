import { AnimatePresence, motion } from "framer-motion";
import {
  Aperture,
  ArrowUpRight,
  Check,
  CircleAlert,
  Clapperboard,
  CloudUpload,
  Film,
  Gauge,
  ImagePlus,
  Library,
  Menu,
  Pencil,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const api = async (path, options = {}) => {
  const response = await fetch(path, options);
  const body = await response.text();
  let data;
  try {
    data = body ? JSON.parse(body) : null;
  } catch {
    data = body;
  }
  if (!response.ok) {
    const detail = data?.detail;
    const message =
      (typeof detail === "string" ? detail : detail?.message) ||
      data?.message ||
      `Request failed (${response.status})`;
    throw Object.assign(new Error(message), {
      status: response.status,
      projectId:
        typeof detail === "object" && detail ? detail.project_id : null,
    });
  }
  return data;
};

const roleLabel = {
  reference: "参考",
  character: "角色",
  location: "场景",
  prop: "道具",
  style: "画风",
  start_frame: "首帧",
  audio: "声音",
};

const stylePresets = [
  {
    id: "cinematic",
    label: "电影写实",
    copy: "自然光 · 真实运动",
    instructions: "电影级自然光，真实生活质感，连续运动，克制的镜头语言。",
    color: "amber",
  },
  {
    id: "documentary",
    label: "纪录片",
    copy: "手持感 · 生活质感",
    instructions: "轻微手持跟拍，真实环境声，保留偶然性和人物呼吸感。",
    color: "mint",
  },
  {
    id: "music_video",
    label: "音乐短片",
    copy: "节奏感 · 强烈构图",
    instructions: "节奏鲜明，构图大胆，色彩和动作随情绪推进，但保持主体连续。",
    color: "violet",
  },
  {
    id: "commercial",
    label: "品牌广告",
    copy: "精致光线 · 高级质感",
    instructions: "精致布光，干净背景，主体清晰，动作有设计感，适合品牌叙事。",
    color: "rose",
  },
  {
    id: "noir",
    label: "黑色电影",
    copy: "低调光 · 悬疑氛围",
    instructions:
      "低调光和高反差，局部光源，沉静克制，营造悬疑但不牺牲可读性。",
    color: "blue",
  },
  {
    id: "animation",
    label: "手绘动画",
    copy: "笔触感 · 想象力",
    instructions: "统一的手绘笔触和角色造型，动作流畅，色彩有童话般的层次。",
    color: "peach",
  },
  {
    id: "retro",
    label: "复古胶片",
    copy: "颗粒感 · 怀旧色调",
    instructions:
      "复古胶片颗粒，柔和高光，略微褪色的暖色调，像被保存下来的记忆。",
    color: "amber",
  },
  {
    id: "surreal",
    label: "超现实",
    copy: "梦境感 · 非日常",
    instructions:
      "现实空间中加入克制的梦境元素，保持人物身份、空间和动作的连续性。",
    color: "violet",
  },
];

function loadCustomStyles() {
  try {
    const value = JSON.parse(
      window.localStorage.getItem("nautilus.customStyles") || "[]",
    );
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

const splitLines = (value) =>
  String(value || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
const joinLines = (value) =>
  Array.isArray(value) ? value.join("\n") : String(value || "");

function formatDuration(seconds) {
  if (
    seconds === null ||
    seconds === undefined ||
    Number.isNaN(Number(seconds))
  )
    return "—";
  const rounded = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(rounded / 60);
  const rest = rounded % 60;
  return minutes ? `${minutes}m ${String(rest).padStart(2, "0")}s` : `${rest}s`;
}

function estimateProjectSeconds(value, scale = 1) {
  if (!value?.shots?.length) return 0;
  let total = 0;
  value.shots.forEach((shot, index) => {
    const isContinuation = Boolean(
      shot.continuity_from_shot_id && !shot.start_frame_asset_id,
    );
    const continuationMode =
      shot.continuation_mode || value.brief.continuation_mode || "quality";
    const referenceSeconds = !isContinuation
      ? 431.1
      : continuationMode === "quality"
        ? 931.1
        : 635.0;
    total +=
      referenceSeconds *
        (shot.inference_steps / 50) *
        (shot.duration_seconds / 15) +
      8;
    total += 2;
    if (index > 0 && !shot.continuity_from_shot_id) total += 20;
  });
  total += Math.max(5, value.brief.duration_seconds * 0.15);
  return Math.round(total * scale);
}

function runtimeShotTask(project, shot) {
  if (shot?.start_frame_asset_id) return "fl2va";
  if (shot?.continuity_from_shot_id) {
    const mode =
      shot.continuation_mode || project?.brief?.continuation_mode || "quality";
    return mode === "quality" ? "ref2va" : "fl2va";
  }
  return shot?.task || "fl2va";
}

function runtimeShotLabel(project, shot) {
  if (shot.start_frame_asset_id) return "FL2VA · 显式首帧";
  const task = runtimeShotTask(project, shot);
  if (shot.continuity_from_shot_id) {
    return task === "ref2va" ? "REF2VA · 高质量续写" : "FL2VA · 快速续写";
  }
  return String(task).toUpperCase();
}

function transitionLabel(shot) {
  return (
    {
      continuous: "连续动作",
      camera_move: "同轴镜头运动",
      match_cut: "匹配剪辑",
      occlusion_cut: "遮挡转场",
      hard_cut: "硬切",
      anchor: "首帧锚点",
    }[shot?.transition_kind] || "连续动作"
  );
}

function frameUrl(projectId, shot, kind) {
  const path =
    kind === "anchor" ? shot.anchor_frame_path : shot.boundary_frame_path;
  const version = encodeURIComponent(path || `${projectId}-${shot.id}`);
  return `/api/projects/${projectId}/shots/${shot.id}/${kind}?v=${version}`;
}

function assetRoles(asset) {
  return (asset.roles?.length ? asset.roles : ["reference"]).map(
    (role) => roleLabel[role] || role,
  );
}

function assetLabel(asset) {
  return asset.display_name || asset.caption || asset.original_name;
}

function AssetCard({ asset, selected, onToggle, onEdit }) {
  const image =
    asset.kind === "image" ? `/api/assets/${asset.id}/content` : null;
  return (
    <motion.article
      className={`asset-card ${selected ? "selected" : ""}`}
      whileHover={{ y: -4 }}
    >
      <motion.button
        className="asset-select"
        whileTap={{ scale: 0.98 }}
        onClick={() => onToggle(asset.id)}
      >
        <div className="asset-visual">
          {image ? (
            <img src={image} alt={assetLabel(asset)} />
          ) : (
            <div className="asset-file">
              <Film size={25} />
            </div>
          )}
          <span className="asset-check">
            {selected ? <Check size={13} /> : <Plus size={13} />}
          </span>
        </div>
        <div className="asset-copy">
          <strong>{assetLabel(asset)}</strong>
          <span>{assetRoles(asset).join(" · ")}</span>
          {asset.tags?.length ? (
            <small>{asset.tags.slice(0, 2).join("  ·  ")}</small>
          ) : null}
        </div>
      </motion.button>
      <button
        className="asset-edit-button"
        type="button"
        onClick={() => onEdit(asset)}
        aria-label={`编辑 ${assetLabel(asset)}`}
      >
        <Pencil size={12} />
        编辑
      </button>
    </motion.article>
  );
}

function StatusPill({ health, job }) {
  const running = job?.status === "running" || job?.status === "queued";
  return (
    <div className={`status-pill ${running ? "working" : "ready"}`}>
      <span className="status-dot" />
      {running
        ? `${Math.round((job.progress || 0) * 100)}% 渲染中`
        : health
          ? "引擎在线"
          : "检查连接"}
    </div>
  );
}

function PlannerDebugConsole({ events, open, onToggle, onClear }) {
  return (
    <section className={`planner-debug ${open ? "is-open" : ""}`}>
      <div className="planner-debug-bar">
        <button type="button" className="text-button" onClick={onToggle}>
          {open ? "收起 Debug Console" : "打开 Debug Console"} · {events.length} events
        </button>
        {events.length > 0 && (
          <button type="button" className="text-button muted" onClick={onClear}>
            清空
          </button>
        )}
      </div>
      {open && (
        <div className="planner-debug-body">
          <p className="planner-debug-hint">
            这里显示浏览器点击、planner 各阶段请求/响应和异常。payload 已做长度限制，不包含认证 header。
          </p>
          {events.length === 0 ? (
            <div className="empty-state">还没有 planner 事件。点击“开始构思”后这里会实时更新。</div>
          ) : (
            [...events].reverse().map((event) => (
              <details className={`planner-debug-event status-${event.status}`} key={event.id}>
                <summary>
                  <span>{event.stage}</span>
                  <span>{event.status}</span>
                  <span>{event.created_at ? new Date(event.created_at).toLocaleTimeString() : ""}</span>
                  <strong>{event.message || event.error || ""}</strong>
                </summary>
                <div className="planner-debug-content">
                  {event.error && <pre className="debug-error">{event.error}</pre>}
                  {event.request_payload && (
                    <details>
                      <summary>raw input</summary>
                      <pre>{event.request_payload}</pre>
                    </details>
                  )}
                  {event.response_payload && (
                    <details>
                      <summary>raw output</summary>
                      <pre>{event.response_payload}</pre>
                    </details>
                  )}
                </div>
              </details>
            ))
          )}
        </div>
      )}
    </section>
  );
}

function App() {
  const [assets, setAssets] = useState([]);
  const [projects, setProjects] = useState([]);
  const [project, setProject] = useState(null);
  const [health, setHealth] = useState(null);
  const [renderEstimateScale, setRenderEstimateScale] = useState(1);
  const [job, setJob] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [title, setTitle] = useState("未命名影片");
  const [duration, setDuration] = useState(60);
  const [aspect, setAspect] = useState("16:9");
  const [quality, setQuality] = useState("final");
  const [continuationMode, setContinuationMode] = useState("quality");
  const [style, setStyle] = useState("cinematic");
  const [styleName, setStyleName] = useState(stylePresets[0].label);
  const [styleInstructions, setStyleInstructions] = useState(
    stylePresets[0].instructions,
  );
  const [registryStyles, setRegistryStyles] = useState(stylePresets);
  const [customStyles, setCustomStyles] = useState(loadCustomStyles);
  const [selected, setSelected] = useState(new Set());
  const [uploadRole, setUploadRole] = useState("reference");
  const [uploadTags, setUploadTags] = useState("");
  const [activeTab, setActiveTab] = useState("brief");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [planningError, setPlanningError] = useState("");
  const [projectLoading, setProjectLoading] = useState(false);
  const [plannerTrace, setPlannerTrace] = useState([]);
  const [clientTrace, setClientTrace] = useState([]);
  const [debugOpen, setDebugOpen] = useState(false);
  const [clockNow, setClockNow] = useState(Date.now());
  const [editingAsset, setEditingAsset] = useState(null);
  const [assetDraft, setAssetDraft] = useState({
    display_name: "",
    caption: "",
    tags: "",
    roles: ["reference"],
  });
  const [assetSaving, setAssetSaving] = useState(false);
  const [assetDeleting, setAssetDeleting] = useState(false);
  const [styleDialog, setStyleDialog] = useState(false);
  const [styleDraft, setStyleDraft] = useState({ name: "", instructions: "" });
  const [editingShot, setEditingShot] = useState(null);
  const [shotDraft, setShotDraft] = useState(null);
  const [projectDialog, setProjectDialog] = useState(false);
  const [projectDraft, setProjectDraft] = useState(null);
  const [dialogSaving, setDialogSaving] = useState(false);
  const fileInput = useRef(null);
  const projectRequest = useRef(0);
  const traceEpoch = useRef(0);
  const traceHiddenBefore = useRef(0);

  const addClientTrace = useCallback((status, message, payload = null) => {
    setClientTrace((events) => [
      ...events,
      {
        id: `client-${Date.now()}-${events.length}`,
        created_at: new Date().toISOString(),
        stage: "browser",
        status: "client",
        message: `${status}: ${message}`,
        request_payload: payload ? JSON.stringify(payload, null, 2) : null,
      },
    ].slice(-40));
  }, []);

  const refreshPlannerTrace = useCallback(async (id) => {
    if (!id) return;
    const requestEpoch = traceEpoch.current;
    try {
      const value = await api(`/api/projects/${id}/planner-trace`);
      if (requestEpoch !== traceEpoch.current) return;
      const hiddenBefore = traceHiddenBefore.current;
      const visible = (Array.isArray(value) ? value : []).filter((event) => {
        if (!hiddenBefore || !event.created_at) return true;
        const createdAt = Date.parse(event.created_at);
        return Number.isNaN(createdAt) || createdAt >= hiddenBefore;
      });
      setPlannerTrace(visible);
    } catch {
      if (requestEpoch === traceEpoch.current) setPlannerTrace([]);
    }
  }, []);

  const clearDebugTrace = useCallback(() => {
    // Invalidate in-flight polls and hide the server events that existed at
    // the moment of the click. Otherwise the next poll immediately restores
    // the same trace and makes the button appear ineffective.
    traceEpoch.current += 1;
    traceHiddenBefore.current = Date.now();
    setPlannerTrace([]);
    setClientTrace([]);
    setNotice("Debug Console 已清空");
  }, []);
  const styleRegistryRef = useRef(stylePresets);

  const loadStylePresets = useCallback(async () => {
    try {
      const remote = await api("/api/style-presets");
      if (!Array.isArray(remote) || !remote.length) return;
      const next = remote.map((item) => ({
        id: item.id,
        label: item.label,
        copy: item.copy,
        instructions: item.ui_instructions || item.instructions || "",
        color: item.color,
      }));
      styleRegistryRef.current = next;
      setRegistryStyles(next);
    } catch {
      // Source-install fallback keeps the bundled short presets usable.
    }
  }, []);

  const loadAssets = useCallback(
    async () => setAssets((await api("/api/assets")) || []),
    [],
  );
  const loadProjects = useCallback(async () => {
    const value = (await api("/api/projects")) || [];
    setProjects(value);
    return value;
  }, []);
  const loadHealth = useCallback(async () => {
    try {
      const value = await api("/api/health");
      setHealth(value.fl2va_healthy && value.ref2va_healthy);
      setRenderEstimateScale(Number(value.render_estimate_scale || 1));
    } catch {
      setHealth(false);
    }
  }, []);

  const availableStyles = useMemo(
    () => [...registryStyles, ...customStyles],
    [registryStyles, customStyles],
  );

  const loadProject = useCallback(
    async (id) => {
      if (!id) return;
      const requestId = ++projectRequest.current;
      const traceRequestEpoch = ++traceEpoch.current;
      traceHiddenBefore.current = 0;
      setProjectLoading(true);
      // Clear the previous project before fetching so its video/job cannot be
      // mistaken for the project the creator just selected.
      setProject(null);
      setJob(null);
      setSelected(new Set());
      const [value, latest, trace] = await Promise.all([
        api(`/api/projects/${id}`),
        api(`/api/projects/${id}/jobs/latest`),
        api(`/api/projects/${id}/planner-trace`).catch(() => []),
      ]).finally(() => {
        if (requestId === projectRequest.current) setProjectLoading(false);
      });
      if (requestId !== projectRequest.current) return;
      setProject(value);
      setPlanningError(
        value.status === "failed" && !value.shots?.length
          ? "上次构思未完成。项目草稿已保留，可以直接重新构思。"
          : "",
      );
      setPrompt(value.brief.prompt);
      setTitle(value.brief.title);
      setDuration(value.brief.duration_seconds);
      setAspect(value.brief.aspect_ratio);
      setQuality(value.brief.quality || "final");
      setContinuationMode(value.brief.continuation_mode || "quality");
      setStyle(value.brief.style_preset || "cinematic");
      const loadedStyle = [...styleRegistryRef.current, ...customStyles].find(
        (item) => item.id === (value.brief.style_preset || "cinematic"),
      );
      setStyleName(value.brief.style || loadedStyle?.label || "电影写实");
      setStyleInstructions(
        value.brief.style_instructions || loadedStyle?.instructions || "",
      );
      setSelected(new Set(value.brief.reference_asset_ids || []));
      setJob(latest);
      if (traceRequestEpoch === traceEpoch.current) {
        setPlannerTrace(Array.isArray(trace) ? trace : []);
      }
      setClientTrace([]);
    },
    [customStyles],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [, availableProjects] = await Promise.all([
          loadAssets(),
          loadProjects(),
          loadStylePresets(),
          loadHealth(),
        ]);
        if (!cancelled && availableProjects.length)
          await loadProject(availableProjects[0].id);
      } catch (error) {
        if (!cancelled) setNotice(error.message);
      }
    })();
    return () => {
      cancelled = true;
      projectRequest.current += 1;
    };
  }, [loadAssets, loadProjects, loadStylePresets, loadHealth, loadProject]);

  useEffect(() => {
    const projectId = project?.id;
    const jobId = job?.id;
    const jobStatus = job?.status;
    if (!projectId || !jobId || !["running", "queued"].includes(jobStatus))
      return undefined;
    let cancelled = false;
    const refresh = async () => {
      try {
        const [next, nextProject] = await Promise.all([
          api(`/api/jobs/${jobId}`),
          api(`/api/projects/${projectId}`),
        ]);
        if (cancelled) return;
        setJob(next);
        setProject(nextProject);
        if (next.status === "complete") setNotice("成片已完成，可以开始预览");
      } catch (error) {
        if (!cancelled) setNotice(error.message);
      }
    };
    void refresh();
    const timer = setInterval(refresh, 3500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.id, job?.status, project?.id]);

  useEffect(() => {
    if (!busy || !project?.id) return undefined;
    const timer = window.setInterval(() => refreshPlannerTrace(project.id), 1000);
    return () => window.clearInterval(timer);
  }, [busy, project?.id, refreshPlannerTrace]);

  useEffect(() => {
    if (!job || !["running", "queued"].includes(job.status)) return undefined;
    const timer = setInterval(() => setClockNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [job?.status]);

  const selectedAssets = useMemo(
    () => assets.filter((asset) => selected.has(asset.id)),
    [assets, selected],
  );
  const toggleAsset = (id) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const openAssetEditor = (asset) => {
    setEditingAsset(asset);
    setAssetDraft({
      display_name:
        asset.display_name || asset.caption || asset.original_name || "",
      caption: asset.caption || "",
      tags: (asset.tags || []).join(", "),
      roles: asset.roles?.length ? [...asset.roles] : ["reference"],
    });
  };

  const toggleDraftRole = (role) =>
    setAssetDraft((current) => ({
      ...current,
      roles: current.roles.includes(role)
        ? current.roles.filter((value) => value !== role)
        : [...current.roles, role],
    }));

  const saveAsset = async (event) => {
    event.preventDefault();
    if (!editingAsset) return;
    setAssetSaving(true);
    try {
      await api(`/api/assets/${editingAsset.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: assetDraft.display_name.trim(),
          caption: assetDraft.caption.trim(),
          tags: [
            ...new Set(
              assetDraft.tags
                .split(",")
                .map((item) => item.trim().toLowerCase())
                .filter(Boolean),
            ),
          ],
          roles: assetDraft.roles.length ? assetDraft.roles : ["reference"],
        }),
      });
      await loadAssets();
      setEditingAsset(null);
      setNotice("素材信息已更新，Agent 会使用新的角色和标签");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAssetSaving(false);
    }
  };

  const deleteAsset = async () => {
    if (!editingAsset) return;
    if (!window.confirm(`确认删除素材“${assetLabel(editingAsset)}”？`)) return;
    setAssetDeleting(true);
    try {
      await api(`/api/assets/${editingAsset.id}`, { method: "DELETE" });
      setSelected((current) => {
        const next = new Set(current);
        next.delete(editingAsset.id);
        return next;
      });
      await loadAssets();
      setEditingAsset(null);
      setNotice("素材已删除");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setAssetDeleting(false);
    }
  };

  const newProject = () => {
    projectRequest.current += 1;
    setProject(null);
    setJob(null);
    setPrompt("");
    setTitle("未命名影片");
    setDuration(60);
    setAspect("16:9");
    setQuality("final");
    setContinuationMode("quality");
    setStyle("cinematic");
    setStyleName(stylePresets[0].label);
    setStyleInstructions(stylePresets[0].instructions);
    setSelected(new Set());
    setActiveTab("brief");
    setPlanningError("");
    traceEpoch.current += 1;
    traceHiddenBefore.current = 0;
    setPlannerTrace([]);
    setClientTrace([]);
  };

  const navigateWorkspace = (sectionId) => {
    setActiveTab(sectionId);
    const target =
      document.getElementById(sectionId) ||
      document.getElementById("empty-workspace");
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (sectionId === "storyboard" && !project)
      setNotice("先生成故事板，再查看分镜");
    if (sectionId === "render" && !job)
      setNotice(
        project
          ? "点击开始制作，渲染区会显示实时进度"
          : "先生成故事板，再开始制作",
      );
  };

  const makeBrief = () => ({
    title,
    prompt,
    duration_seconds: Number(duration),
    aspect_ratio: aspect,
    style: styleName || "cinematic realism",
    style_preset: style,
    style_instructions: styleInstructions,
    quality,
    continuation_mode: continuationMode,
    language: "zh-CN",
    audience: "general",
    reference_asset_ids: [...selected],
    subtitle_mode: "none",
  });

  const selectStyle = (item) => {
    setStyle(item.id);
    setStyleName(item.label);
    setStyleInstructions(item.instructions || "");
  };

  const openStyleEditor = () => {
    setStyleDraft({ name: styleName, instructions: styleInstructions });
    setStyleDialog(true);
  };

  const applyStyleDraft = () => {
    setStyleName(styleDraft.name.trim() || "自定义视觉气质");
    setStyleInstructions(styleDraft.instructions.trim());
    setStyleDialog(false);
  };

  const saveCustomStyle = () => {
    const name = styleDraft.name.trim() || "自定义视觉气质";
    const id = `custom-${Date.now()}`;
    const item = {
      id,
      label: name,
      copy: "我的导演模板",
      instructions: styleDraft.instructions.trim(),
      color: "peach",
    };
    const next = [...customStyles, item];
    setCustomStyles(next);
    window.localStorage.setItem("nautilus.customStyles", JSON.stringify(next));
    setStyle(id);
    setStyleName(name);
    setStyleInstructions(item.instructions);
    setStyleDialog(false);
    setNotice("自定义视觉气质已保存");
  };

  const openProjectEditor = () => {
    if (!project) return;
    setProjectDraft({
      brief: {
        title: project.brief.title || "",
        prompt: project.brief.prompt || "",
        style: project.brief.style || styleName,
        style_preset: project.brief.style_preset || style,
        style_instructions:
          project.brief.style_instructions || styleInstructions,
        duration_seconds: project.brief.duration_seconds,
        aspect_ratio: project.brief.aspect_ratio,
        quality: project.brief.quality,
        subtitle_mode: project.brief.subtitle_mode || "none",
      },
      world_bible: {
        logline: project.world_bible?.logline || "",
        visual_style: project.world_bible?.visual_style || "",
        character_notes: joinLines(project.world_bible?.character_notes),
        location_notes: joinLines(project.world_bible?.location_notes),
        prop_notes: joinLines(project.world_bible?.prop_notes),
        audio_notes: joinLines(project.world_bible?.audio_notes),
        continuity_rules: joinLines(project.world_bible?.continuity_rules),
      },
    });
    setProjectDialog(true);
  };

  const updateProjectDraft = (section, key, value) =>
    setProjectDraft((current) => ({
      ...current,
      [section]: { ...current[section], [key]: value },
    }));

  const saveProjectDialog = async (event) => {
    event.preventDefault();
    if (!project || !projectDraft) return;
    setDialogSaving(true);
    try {
      const payload = {
        brief: {
          ...projectDraft.brief,
          duration_seconds: Number(projectDraft.brief.duration_seconds),
        },
        world_bible: {
          ...projectDraft.world_bible,
          character_notes: splitLines(projectDraft.world_bible.character_notes),
          location_notes: splitLines(projectDraft.world_bible.location_notes),
          prop_notes: splitLines(projectDraft.world_bible.prop_notes),
          audio_notes: splitLines(projectDraft.world_bible.audio_notes),
          continuity_rules: splitLines(
            projectDraft.world_bible.continuity_rules,
          ),
        },
      };
      const value = await api(`/api/projects/${project.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setProject(value);
      setJob(null);
      setTitle(value.brief.title);
      setPrompt(value.brief.prompt);
      setStyle(value.brief.style_preset || style);
      setStyleName(value.brief.style || styleName);
      setStyleInstructions(value.brief.style_instructions || styleInstructions);
      setDuration(value.brief.duration_seconds);
      setAspect(value.brief.aspect_ratio);
      setQuality(value.brief.quality || "final");
      setContinuationMode(value.brief.continuation_mode || "quality");
      await loadProjects();
      setProjectDialog(false);
      setNotice("整体设定已保存，旧镜头输出已标记为待重新制作");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setDialogSaving(false);
    }
  };

  const changeContinuationMode = async (mode) => {
    setContinuationMode(mode);
    if (!project) return;
    try {
      const value = await api(`/api/projects/${project.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brief: { continuation_mode: mode } }),
      });
      setProject(value);
      setJob(null);
      await loadProjects();
      setNotice(
        mode === "fast"
          ? "已切换为快速续写：后续镜头参考上一镜最后 5 秒"
          : "已切换为高质量续写：后续镜头参考完整上一镜",
      );
    } catch (error) {
      setNotice(`续写模式保存失败：${error.message}`);
    }
  };

  const openShotEditor = (shot) => {
    setEditingShot(shot);
    setShotDraft({
      ...shot,
      reference_asset_ids: [...(shot.reference_asset_ids || [])],
      duration_seconds: shot.duration_seconds,
      anchor_prompt: shot.anchor_prompt || "",
      audio_prompt: shot.audio_prompt || "",
      music_prompt: shot.music_prompt || "",
      dialogue: (shot.dialogue || []).map((line) => ({ ...line })),
      opening_state: shot.opening_state || "",
      ending_state: shot.ending_state || "",
      continuity_handoff: shot.continuity_handoff || "",
      transition_kind: shot.transition_kind || "continuous",
      reference_anchors: [...(shot.reference_anchors || [])],
      hook: shot.hook || "",
      visual_beats: (shot.visual_beats || []).map((beat) => ({ ...beat })),
      subtitle_text: shot.subtitle_text || "",
    });
  };

  const updateShotDraft = (key, value) =>
    setShotDraft((current) => ({ ...current, [key]: value }));

  const addDialogueLine = () =>
    setShotDraft((current) => ({
      ...current,
      dialogue: [
        ...(current.dialogue || []),
        {
          speaker: "",
          text: "",
          language: "Chinese",
          delivery: "自然",
          mode: "on_screen",
          start_seconds: null,
          end_seconds: null,
        },
      ],
    }));

  const updateDialogueLine = (index, key, value) =>
    setShotDraft((current) => ({
      ...current,
      dialogue: current.dialogue.map((line, lineIndex) =>
        lineIndex === index ? { ...line, [key]: value } : line,
      ),
    }));

  const removeDialogueLine = (index) =>
    setShotDraft((current) => ({
      ...current,
      dialogue: current.dialogue.filter((_, lineIndex) => lineIndex !== index),
    }));

  const updateVisualBeat = (index, key, value) =>
    setShotDraft((current) => ({
      ...current,
      visual_beats: current.visual_beats.map((beat, beatIndex) =>
        beatIndex === index ? { ...beat, [key]: value } : beat,
      ),
    }));

  const updateShotTask = (value) =>
    setShotDraft((current) => {
      if (!current?.continuity_from_shot_id) {
        return { ...current, task: value };
      }
      return {
        ...current,
        task: "fl2va",
        continuation_mode: value === "ref2va" ? "quality" : "fast",
      };
    });

  const toggleShotReference = (assetId) =>
    setShotDraft((current) => ({
      ...current,
      reference_asset_ids: current.reference_asset_ids.includes(assetId)
        ? current.reference_asset_ids.filter((id) => id !== assetId)
        : [...current.reference_asset_ids, assetId],
    }));

  const saveShotDialog = async (event) => {
    event.preventDefault();
    if (!project || !editingShot || !shotDraft) return;
    setDialogSaving(true);
    try {
      const value = await api(
        `/api/projects/${project.id}/shots/${editingShot.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: shotDraft.title,
            purpose: shotDraft.purpose,
            duration_seconds: Number(shotDraft.duration_seconds),
            task: shotDraft.task,
            continuation_mode: shotDraft.continuation_mode || null,
            anchor_prompt: shotDraft.anchor_prompt,
            prompt: shotDraft.prompt,
            audio_prompt: shotDraft.audio_prompt,
            music_prompt: shotDraft.music_prompt,
            dialogue: (shotDraft.dialogue || []).map((line) => ({
              speaker: line.speaker.trim(),
              text: line.text.trim(),
              language: (line.language || "Chinese").trim(),
              delivery: (line.delivery || "自然").trim(),
              mode: line.mode || "on_screen",
              start_seconds:
                line.start_seconds === "" || line.start_seconds == null
                  ? null
                  : Number(line.start_seconds),
              end_seconds:
                line.end_seconds === "" || line.end_seconds == null
                  ? null
                  : Number(line.end_seconds),
            })),
            opening_state: shotDraft.opening_state,
            ending_state: shotDraft.ending_state,
            continuity_handoff: shotDraft.continuity_handoff,
            transition_kind: shotDraft.transition_kind || "continuous",
            reference_anchors: shotDraft.reference_anchors
              .map((value) => value.trim())
              .filter(Boolean),
            hook: shotDraft.hook,
            visual_beats: (shotDraft.visual_beats || []).map((beat) => ({
              start_seconds: Number(beat.start_seconds),
              end_seconds: Number(beat.end_seconds),
              visual_action: beat.visual_action.trim(),
              state_change: beat.state_change.trim(),
              camera: beat.camera.trim(),
              sound: beat.sound.trim(),
              performance: (beat.performance || "").trim(),
              spatial_anchor: (beat.spatial_anchor || "").trim(),
              handoff: (beat.handoff || "").trim(),
            })),
            negative_prompt: shotDraft.negative_prompt,
            subtitle_text: shotDraft.subtitle_text || null,
            camera: shotDraft.camera,
            reference_asset_ids: shotDraft.reference_asset_ids,
            start_frame_asset_id: shotDraft.start_frame_asset_id || null,
            audio_asset_id: shotDraft.audio_asset_id || null,
            continuity_from_shot_id: shotDraft.continuity_from_shot_id || null,
            seed: Number(shotDraft.seed),
            fps: Number(shotDraft.fps),
            inference_steps: Number(shotDraft.inference_steps),
            flow_shift: Number(shotDraft.flow_shift),
          }),
        },
      );
      setProject(value);
      setJob(null);
      setEditingShot(null);
      setNotice("镜头已保存，生成新的成片前会重新渲染这一镜");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setDialogSaving(false);
    }
  };

  const plan = async () => {
    addClientTrace("click", "开始构思 clicked", { promptLength: prompt.trim().length });
    if (projectLoading) {
      addClientTrace("skipped", "project is still loading");
      setDebugOpen(true);
      return setNotice("项目仍在加载，请稍候再开始构思");
    }
    if (prompt.trim().length < 3) {
      addClientTrace("skipped", "prompt is shorter than 3 characters");
      setDebugOpen(true);
      return setNotice("先写一句你想拍的故事");
    }
    const requestId = ++projectRequest.current;
    setPlanningError("");
    traceEpoch.current += 1;
    traceHiddenBefore.current = Date.now();
    setPlannerTrace([]);
    setBusy(true);
    const brief = makeBrief();
    addClientTrace("request", "POST /api/projects/plan", brief);
    try {
      const value = await api("/api/projects/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(brief),
      });
      addClientTrace("response", "planner returned a storyboard", {
        projectId: value?.id,
        shotCount: value?.shots?.length || 0,
      });
      if (requestId !== projectRequest.current) return;
      await loadProjects();
      if (requestId !== projectRequest.current) return;
      setProject(value);
      setJob(null);
      setActiveTab("storyboard");
      setNotice("故事板已生成，可以逐镜检查");
      window.requestAnimationFrame(() =>
        document
          .getElementById("storyboard")
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
    } catch (error) {
      addClientTrace("failed", error.message || String(error), {
        projectId: error.projectId || null,
      });
      setDebugOpen(true);
      if (requestId !== projectRequest.current) return;
      if (error.projectId) {
        await loadProjects();
        await loadProject(error.projectId);
      }
      setPlanningError(error.message);
      setNotice(error.message);
    } finally {
      setBusy(false);
      if (project?.id) refreshPlannerTrace(project.id);
    }
  };

  const render = async () => {
    if (!project?.id) return setNotice("先生成故事板，再渲染成片");
    const projectId = project.id;
    setBusy(true);
    try {
      const value = await api(`/api/projects/${projectId}/render`, {
        method: "POST",
      });
      setJob(value);
      setActiveTab("render");
      setNotice("渲染已开始，期间可以继续编辑项目");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (event) => {
    const files = [...event.target.files];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append("roles", uploadRole);
    form.append("tags", uploadTags);
    try {
      await api("/api/assets/upload", { method: "POST", body: form });
      await loadAssets();
      setNotice(
        `${files.length} 份素材已加入素材库（${roleLabel[uploadRole] || uploadRole}）`,
      );
    } catch (error) {
      setNotice(error.message);
    } finally {
      event.target.value = "";
    }
  };

  const estimatedSeconds = useMemo(
    () => estimateProjectSeconds(project, renderEstimateScale),
    [project, renderEstimateScale],
  );
  const elapsedSeconds = job?.created_at
    ? Math.max(0, (clockNow - Date.parse(job.created_at)) / 1000)
    : 0;
  const estimatedProgress = estimatedSeconds
    ? Math.min(0.99, elapsedSeconds / estimatedSeconds)
    : 0;
  const progress =
    job?.status === "complete"
      ? 100
      : Math.round(
          Math.max((job?.progress || 0) * 100, estimatedProgress * 100),
        );
  const remainingSeconds = Math.max(0, estimatedSeconds - elapsedSeconds);
  const currentShot = project?.shots?.find(
    (shot) => shot.id === job?.current_shot_id,
  );
  const jobActive = job && ["running", "queued"].includes(job.status);
  const debugEvents = [...plannerTrace, ...clientTrace].sort(
    (left, right) => Date.parse(left.created_at || 0) - Date.parse(right.created_at || 0),
  );
  return (
    <div className="nautilus-app">
      <aside className="side-rail">
        <div className="nav-logo">
          <span>N</span>
          <div>
            <b>NAUTILUS</b>
            <small>AI FILM WORKSHOP</small>
          </div>
        </div>
        <div className="rail-section-label">WORKSPACE</div>
        {[
          { id: "brief", icon: Sparkles, label: "灵感简报" },
          { id: "library", icon: Library, label: "素材库" },
          { id: "storyboard", icon: Clapperboard, label: "故事板" },
          { id: "render", icon: Aperture, label: "渲染中心" },
        ].map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            className={`rail-link ${activeTab === id ? "active" : ""}`}
            onClick={() => navigateWorkspace(id)}
          >
            <Icon size={17} />
            <span>{label}</span>
            {id === "render" && job?.status === "running" ? <i /> : null}
          </button>
        ))}
        <div className="rail-spacer" />
        <div className="engine-card">
          <div className="engine-icon">
            <WandSparkles size={16} />
          </div>
          <div>
            <strong>海螺引擎</strong>
            <span>{health ? "H3 · 在线" : "等待连接"}</span>
          </div>
          <span className={`engine-dot ${health ? "on" : ""}`} />
        </div>
      </aside>

      <main className="main-canvas">
        <header className="workspace-header">
          <div className="mobile-menu">
            <Menu size={19} />
          </div>
          <div className="crumb project-picker">
            <span>PROJECT</span>
            <select
              value={project?.id || ""}
              onChange={(event) => {
                if (!event.target.value) return newProject();
                loadProject(event.target.value).catch((error) =>
                  setNotice(error.message),
                );
              }}
              aria-label="选择项目"
            >
              <option value="">NEW FILM</option>
              {projects.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.brief.title}
                </option>
              ))}
            </select>
          </div>
          <div className="header-actions">
            <StatusPill health={health} job={job} />
            <button className="header-new" onClick={newProject}>
              <Plus size={15} /> 新项目
            </button>
          </div>
        </header>

        <section className="hero-card">
          <div className="hero-orbit orbit-one" />
          <div className="hero-orbit orbit-two" />
          <div className="hero-content">
            <div className="overline">
              <Sparkles size={13} /> DIRECTOR'S BRIEF
            </div>
            <h1>
              把一个念头，<em>拍成一部片</em>
            </h1>
            <p>
              一句话、几张参考素材。Nautilus
              会把你的灵感拆成故事、镜头和一条可观看的时间线。
            </p>
          </div>
          <div className="hero-stat">
            <span>CREATIVE ENGINE</span>
            <strong>H3 / OMNI</strong>
            <small>连续镜头渲染</small>
          </div>
        </section>

        <div className="studio-grid">
          <section className="composer glass-panel" id="brief">
            <div className="section-heading">
              <div>
                <span className="overline">01 · STORY SEED</span>
                <h2>你的故事</h2>
              </div>
              <span className="counter">{prompt.length}/2000</span>
            </div>
            <input
              className="title-input"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="给这部片一个名字"
            />
            <textarea
              className="story-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="例如：一位女孩在雨夜的旧车站等待一封迟到的信……"
              maxLength={2000}
            />
            <div className="composer-footer">
              <div className="hint">
                <Sparkles size={14} /> Agent 会自动生成完整分镜
              </div>
              <button className="glow-button" onClick={plan} disabled={busy || projectLoading}>
                <span>{projectLoading ? "正在加载项目…" : busy ? "正在构思…" : "开始构思"}</span>
                <ArrowUpRight size={16} />
              </button>
            </div>
            {planningError ||
            (project?.status === "failed" && !project.shots?.length) ? (
              <div className="planning-error" role="alert">
                <CircleAlert size={17} />
                <div>
                  <b>这次构思没有完成</b>
                  <span>
                    {planningError ||
                      "项目草稿已保留。检查故事内容后，可以直接重新构思。"}
                  </span>
                </div>
                <button
                  className="outline-button compact"
                  type="button"
                  onClick={plan}
                  disabled={busy}
                >
                  重新构思
                </button>
              </div>
            ) : null}
            <PlannerDebugConsole
              events={debugEvents}
              open={debugOpen}
              onToggle={() => setDebugOpen((value) => !value)}
              onClear={clearDebugTrace}
            />
          </section>

          <section className="controls glass-panel">
            <div className="section-heading">
              <div>
                <span className="overline">02 · DIRECTION</span>
                <h2>导演选择</h2>
              </div>
              <Gauge size={18} className="muted-icon" />
            </div>
            <label className="field-label">视觉气质</label>
            <div className="preset-list">
              {availableStyles.map((item) => (
                <button
                  key={item.id}
                  className={`preset ${style === item.id ? "selected" : ""} ${item.color}`}
                  onClick={() => selectStyle(item)}
                >
                  <span className="preset-swatch" />
                  <div>
                    <b>{item.label}</b>
                    <small>{item.copy}</small>
                  </div>
                  {style === item.id ? <Check size={15} /> : null}
                </button>
              ))}
              <button className="preset-add" onClick={openStyleEditor}>
                <Plus size={14} />
                <span>
                  <b>新增视觉气质</b>
                  <small>写下你的导演规则</small>
                </span>
              </button>
            </div>
            <div className="direction-summary">
              <div>
                <span>当前视觉气质</span>
                <strong>{styleName}</strong>
              </div>
              <button
                className="outline-button compact"
                onClick={openStyleEditor}
              >
                <Settings2 size={13} /> 编辑说明
              </button>
            </div>
            <div className="control-row">
              <label>
                <span>时长</span>
                <select
                  value={duration}
                  onChange={(event) => setDuration(event.target.value)}
                >
                  <option value="14">14 秒（安全上限）</option>
                  <option value="30">30 秒</option>
                  <option value="60">1 分钟</option>
                  <option value="120">2 分钟</option>
                </select>
              </label>
              <label>
                <span>画幅</span>
                <select
                  value={aspect}
                  onChange={(event) => setAspect(event.target.value)}
                >
                  <option>16:9</option>
                  <option>9:16</option>
                  <option>1:1</option>
                </select>
              </label>
            </div>
            <div className="quality-toggle">
              <span>渲染质量</span>
              <div>
                {["draft", "final"].map((item) => (
                  <button
                    key={item}
                    className={quality === item ? "active" : ""}
                    onClick={() => setQuality(item)}
                  >
                    {item === "draft" ? "草稿 · 12 steps" : "最终 · 50 steps"}
                  </button>
                ))}
              </div>
              <small className="quality-hint">
                {quality === "draft"
                  ? "12 steps · 更快确认节奏"
                  : "50 steps · 通常更稳，但耗时更长"}
              </small>
            </div>
            <div className="quality-toggle">
              <span>镜头续写</span>
              <div>
                {["fast", "quality"].map((item) => (
                  <button
                    key={item}
                    className={continuationMode === item ? "active" : ""}
                    onClick={() => changeContinuationMode(item)}
                  >
                    {item === "fast" ? "快速续写" : "高质量续写"}
                  </button>
                ))}
              </div>
              <small className="quality-hint">
                {continuationMode === "fast"
                  ? "参考上一镜最后 5 秒 · 默认更快"
                  : "参考完整上一镜 · 上下文最完整"}
              </small>
            </div>
          </section>
        </div>

        <section className="library-section" id="library">
          <div className="section-heading wide">
            <div>
              <span className="overline">03 · MATERIAL LIBRARY</span>
              <h2>
                素材库 <small>{assets.length} assets</small>
              </h2>
            </div>
            <div className="heading-actions">
              <span>{selectedAssets.length} 已编入片单</span>
              <input
                ref={fileInput}
                type="file"
                multiple
                accept="image/*,video/*,audio/*"
                onChange={upload}
                hidden
              />
              <button
                className="outline-button"
                onClick={() => fileInput.current?.click()}
              >
                <CloudUpload size={15} /> 导入素材
              </button>
            </div>
          </div>
          <div className="upload-meta">
            <label>
              素材用途
              <select
                value={uploadRole}
                onChange={(event) => setUploadRole(event.target.value)}
                aria-label="上传素材用途"
              >
                {Object.entries(roleLabel).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              标签
              <input
                value={uploadTags}
                onChange={(event) => setUploadTags(event.target.value)}
                placeholder="人物、客厅、夜景"
                aria-label="上传素材标签"
              />
            </label>
          </div>
          <div className="asset-ribbon">
            {assets.slice(0, 8).map((asset) => (
              <AssetCard
                key={asset.id}
                asset={asset}
                selected={selected.has(asset.id)}
                onToggle={toggleAsset}
                onEdit={openAssetEditor}
              />
            ))}
            {!assets.length ? (
              <div className="empty-library">
                <ImagePlus size={20} />
                <span>导入第一份素材，开始建立你的世界</span>
              </div>
            ) : null}
          </div>
        </section>

        {project ? (
          <section className="project-section" id="storyboard">
            <div className="section-heading wide storyboard-heading">
              <div className="storyboard-heading-copy">
                <span className="overline">04 · STORYBOARD</span>
                <h2>{project.world_bible?.logline || "故事板"}</h2>
              </div>
              <span className="storyboard-meta">
                {project.shots?.length || 0} shots ·{" "}
                {formatDuration(project.brief.duration_seconds)}
              </span>
            </div>
            <div className="storyboard-actions-row">
              <button className="outline-button" onClick={openProjectEditor}>
                <Pencil size={13} /> 编辑整体设定
              </button>
              <button
                className="outline-button"
                onClick={() => navigateWorkspace("render")}
              >
                <Play size={14} /> 查看渲染
              </button>
              <button
                className="glow-button storyboard-render-button"
                onClick={render}
                disabled={busy || jobActive || !project.shots?.length}
              >
                <Play size={14} />{" "}
                {jobActive
                  ? "制作中…"
                  : job?.status === "complete"
                    ? "再次制作"
                    : "开始制作"}
              </button>
            </div>
            <div className="story-strip">
              {(project.shots || []).map((shot, index) => (
                <motion.article
                  key={shot.id}
                  className="shot-tile"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <div className="shot-image">
                    {shot.anchor_frame_path ? (
                      <img
                        src={frameUrl(project.id, shot, "anchor")}
                        alt="anchor"
                      />
                    ) : shot.boundary_frame_path ? (
                      <img
                        src={frameUrl(project.id, shot, "boundary")}
                        alt="boundary"
                      />
                    ) : (
                      <span>{String(index + 1).padStart(2, "0")}</span>
                    )}
                    <small>{formatDuration(shot.duration_seconds)}</small>
                    <button
                      className="shot-edit-button"
                      type="button"
                      onClick={() => openShotEditor(shot)}
                    >
                      <Pencil size={11} /> 编辑镜头
                    </button>
                  </div>
                  <div className="shot-meta">
                    <span>SHOT {String(index + 1).padStart(2, "0")}</span>
                    <b>{shot.title}</b>
                    <p>{shot.purpose}</p>
                    <small>
                      {runtimeShotLabel(project, shot)} · {shot.inference_steps}{" "}
                      steps · {transitionLabel(shot)} · {shot.camera}
                    </small>
                  </div>
                </motion.article>
              ))}
            </div>
          </section>
        ) : (
          <section className="empty-workspace" id="empty-workspace">
            <div className="empty-icon">
              <Film size={25} />
            </div>
            <span className="overline">YOUR CANVAS IS READY</span>
            <h2>先写下一个故事</h2>
            <p>
              {activeTab === "render"
                ? "生成故事板并点击开始制作，渲染进度会在这里实时显示。"
                : activeTab === "storyboard"
                  ? "点击开始构思，AI Agent 会先生成可编辑的分镜故事板。"
                  : "故事板、素材锚点和渲染预览会在这里展开。"}
            </p>
          </section>
        )}

        {project &&
          (job ? (
            <section className="render-section" id="render">
              <div className="render-head">
                <div>
                  <span className="overline">05 · FINAL CUT</span>
                  <h2>渲染中心</h2>
                </div>
                <div className={`render-state ${job.status}`}>
                  <span />
                  {job.status === "complete"
                    ? "已完成"
                    : job.status === "failed"
                      ? "失败"
                      : "渲染中"}
                </div>
              </div>
              <div className="render-body">
                <div className="render-progress">
                  <div className="progress-numbers">
                    <strong>{progress}%</strong>
                    <span>{job.message}</span>
                  </div>
                  <div className="progress-line">
                    <motion.i animate={{ width: `${progress}%` }} />
                  </div>
                  <div className="render-metrics">
                    <div>
                      <small>预计总时长</small>
                      <b>{formatDuration(estimatedSeconds)}</b>
                    </div>
                    <div>
                      <small>已用时间</small>
                      <b>{formatDuration(elapsedSeconds)}</b>
                    </div>
                    <div>
                      <small>{jobActive ? "预计剩余" : "最终用时"}</small>
                      <b>
                        {formatDuration(
                          jobActive ? remainingSeconds : elapsedSeconds,
                        )}
                      </b>
                    </div>
                  </div>
                  <small>
                    {currentShot ? `当前：${currentShot.title}` : "等待下一步"}
                  </small>
                </div>
                {job.status === "complete" ? (
                  <div className="result-wrap">
                    <video
                      className="result-video"
                      controls
                      src={`/api/jobs/${job.id}/output`}
                    />
                    <a
                      className="download-link"
                      href={`/api/jobs/${job.id}/output?download=true`}
                    >
                      下载成片
                    </a>
                  </div>
                ) : null}
                {job.status === "failed" ? (
                  <div className="error-box">{job.error}</div>
                ) : null}
              </div>
            </section>
          ) : (
            <section className="render-section empty-render" id="render">
              <div className="render-head">
                <div>
                  <span className="overline">05 · FINAL CUT</span>
                  <h2>渲染中心</h2>
                </div>
                <div className="render-state">
                  <span />
                  尚未开始
                </div>
              </div>
              <div className="empty-render-body">
                <Aperture size={22} />
                <p>
                  当前项目还没有制作任务。确认故事板后，点击开始制作即可看到实时进度、倒计时和每一镜的锚点更新。
                </p>
                <button
                  className="glow-button"
                  onClick={render}
                  disabled={busy}
                >
                  <Play size={14} /> 开始制作
                </button>
              </div>
            </section>
          ))}
        {!project ? (
          <section className="empty-workspace" id="render">
            <div className="empty-icon">
              <Aperture size={25} />
            </div>
            <span className="overline">RENDER WORKSPACE</span>
            <h2>还没有可制作的故事板</h2>
            <p>先在灵感简报中生成故事板，再回到这里开始制作。</p>
          </section>
        ) : null}
        <footer className="footer-note">
          <span>NAUTILUS STUDIO · CREATOR-FIRST AI FILM WORKSHOP</span>
          <span>
            海螺引擎 <i>●</i>
          </span>
        </footer>
      </main>
      <AnimatePresence>
        {styleDialog ? (
          <motion.div
            className="director-dialog-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={() => setStyleDialog(false)}
          >
            <motion.div
              className="director-dialog"
              initial={{ opacity: 0, y: 14, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              onMouseDown={(event) => event.stopPropagation()}
            >
              <div className="dialog-head">
                <div>
                  <span className="overline">DIRECTOR'S DIRECTION</span>
                  <h2>视觉气质编辑器</h2>
                  <p>
                    模板只是起点。把光线、镜头、节奏和禁用项写成自己的导演规则。
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setStyleDialog(false)}
                  aria-label="关闭视觉气质编辑器"
                >
                  <X size={17} />
                </button>
              </div>
              <label className="dialog-field">
                <span>气质名称</span>
                <input
                  value={styleDraft.name}
                  onChange={(event) =>
                    setStyleDraft((current) => ({
                      ...current,
                      name: event.target.value,
                    }))
                  }
                  placeholder="例如：东方悬疑胶片"
                />
              </label>
              <label className="dialog-field">
                <span>视觉与导演说明</span>
                <textarea
                  rows="7"
                  value={styleDraft.instructions}
                  onChange={(event) =>
                    setStyleDraft((current) => ({
                      ...current,
                      instructions: event.target.value,
                    }))
                  }
                  placeholder="描述光线、色彩、镜头、动作节奏、声音和需要避免的内容…"
                />
              </label>
              <div className="dialog-actions">
                <button
                  className="outline-button"
                  type="button"
                  onClick={applyStyleDraft}
                >
                  仅应用到当前项目
                </button>
                <button
                  className="glow-button"
                  type="button"
                  onClick={saveCustomStyle}
                >
                  <Plus size={14} /> 保存为自定义模板
                </button>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {projectDialog && projectDraft ? (
          <motion.div
            className="director-dialog-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={() => setProjectDialog(false)}
          >
            <motion.form
              className="director-dialog wide-dialog"
              initial={{ opacity: 0, y: 14, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              onSubmit={saveProjectDialog}
              onMouseDown={(event) => event.stopPropagation()}
            >
              <div className="dialog-head">
                <div>
                  <span className="overline">PROJECT BIBLE</span>
                  <h2>编辑整体设定</h2>
                  <p>这是 Agent 和每个镜头共同使用的创作底稿。</p>
                </div>
                <button
                  type="button"
                  onClick={() => setProjectDialog(false)}
                  aria-label="关闭项目设定"
                >
                  <X size={17} />
                </button>
              </div>
              <div className="dialog-columns">
                <div>
                  <label className="dialog-field">
                    <span>片名</span>
                    <input
                      value={projectDraft.brief.title}
                      onChange={(event) =>
                        updateProjectDraft("brief", "title", event.target.value)
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>故事简介 / 原始念头</span>
                    <textarea
                      rows="4"
                      value={projectDraft.brief.prompt}
                      onChange={(event) =>
                        updateProjectDraft(
                          "brief",
                          "prompt",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>视觉气质说明</span>
                    <textarea
                      rows="4"
                      value={projectDraft.brief.style_instructions}
                      onChange={(event) =>
                        updateProjectDraft(
                          "brief",
                          "style_instructions",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>镜头续写策略</span>
                    <select
                      value={projectDraft.brief.continuation_mode || "quality"}
                      onChange={(event) =>
                        updateProjectDraft(
                          "brief",
                          "continuation_mode",
                          event.target.value,
                        )
                      }
                    >
                      <option value="fast">快速续写（上一镜最后 5 秒）</option>
                      <option value="quality">高质量续写（完整上一镜）</option>
                    </select>
                    <small className="field-hint">
                      仅影响 clip1 及后续的镜头承接；首镜和显式首帧不受影响。
                    </small>
                  </label>
                </div>
                <div>
                  <label className="dialog-field">
                    <span>Logline / 一句话介绍</span>
                    <textarea
                      rows="3"
                      value={projectDraft.world_bible.logline}
                      onChange={(event) =>
                        updateProjectDraft(
                          "world_bible",
                          "logline",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>人物设定（每行一个）</span>
                    <textarea
                      rows="3"
                      value={projectDraft.world_bible.character_notes}
                      onChange={(event) =>
                        updateProjectDraft(
                          "world_bible",
                          "character_notes",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>场景 / 道具 / 声音（每行一个）</span>
                    <textarea
                      rows="4"
                      value={`${projectDraft.world_bible.location_notes}\n${projectDraft.world_bible.prop_notes}\n${projectDraft.world_bible.audio_notes}`.trim()}
                      onChange={(event) => {
                        const lines = splitLines(event.target.value);
                        updateProjectDraft(
                          "world_bible",
                          "location_notes",
                          lines.slice(0, 2).join("\n"),
                        );
                        updateProjectDraft(
                          "world_bible",
                          "prop_notes",
                          lines.slice(2, 4).join("\n"),
                        );
                        updateProjectDraft(
                          "world_bible",
                          "audio_notes",
                          lines.slice(4).join("\n"),
                        );
                      }}
                    />
                  </label>
                  <label className="dialog-field">
                    <span>连续性规则（每行一个）</span>
                    <textarea
                      rows="3"
                      value={projectDraft.world_bible.continuity_rules}
                      onChange={(event) =>
                        updateProjectDraft(
                          "world_bible",
                          "continuity_rules",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                </div>
              </div>
              <div className="dialog-actions">
                <button
                  className="outline-button"
                  type="button"
                  onClick={() => setProjectDialog(false)}
                >
                  取消
                </button>
                <button
                  className="glow-button"
                  type="submit"
                  disabled={dialogSaving}
                >
                  {dialogSaving ? "保存中…" : "保存整体设定"}
                </button>
              </div>
            </motion.form>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {editingShot && shotDraft ? (
          <motion.div
            className="director-dialog-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={() => setEditingShot(null)}
          >
            <motion.form
              className="director-dialog wide-dialog shot-dialog"
              initial={{ opacity: 0, y: 14, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              onSubmit={saveShotDialog}
              onMouseDown={(event) => event.stopPropagation()}
            >
              <div className="dialog-head">
                <div>
                  <span className="overline">
                    SHOT {String((editingShot.index || 0) + 1).padStart(2, "0")}
                  </span>
                  <h2>编辑镜头</h2>
                  <p>完整修改会让该镜重新进入待制作状态。</p>
                </div>
                <button
                  type="button"
                  onClick={() => setEditingShot(null)}
                  aria-label="关闭镜头编辑"
                >
                  <X size={17} />
                </button>
              </div>
              <div className="dialog-columns">
                <div>
                  <label className="dialog-field">
                    <span>镜头标题</span>
                    <input
                      value={shotDraft.title}
                      onChange={(event) =>
                        updateShotDraft("title", event.target.value)
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>镜头目的</span>
                    <textarea
                      rows="3"
                      value={shotDraft.purpose}
                      onChange={(event) =>
                        updateShotDraft("purpose", event.target.value)
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>首帧构图 Prompt · 仅零秒静态画面</span>
                    <textarea
                      rows="7"
                      maxLength={1000}
                      value={shotDraft.anchor_prompt}
                      onChange={(event) =>
                        updateShotDraft("anchor_prompt", event.target.value)
                      }
                      placeholder="人物、场景、服装、位置关系、表情、构图、光线；不要写动作过程、镜头运动、对白或音效。"
                    />
                    <small>
                      {shotDraft.start_frame_asset_id
                        ? "已指定首帧素材，本 Prompt 会保留但不会调用 Image Edit。"
                        : `由 Planner 生成；启用 Image Edit 后会作为首帧合成指令。参考素材不会自动变成首帧；需完整绑定参考图，最多 1000 字符（当前 ${shotDraft.anchor_prompt.length}/1000），并以 Qwen 模板后的文本 tokens ≤ 1000 为最终门禁。`}
                    </small>
                  </label>
                  <label className="dialog-field">
                    <span>视频画面与动作 Prompt · 非对白</span>
                    <textarea
                      rows="8"
                      value={shotDraft.prompt}
                      onChange={(event) =>
                        updateShotDraft("prompt", event.target.value)
                      }
                      placeholder="描述主体、环境、动作发展、表演、构图和镜头运动；不要把台词写在这里。"
                    />
                    <small>
                      这部分会被明确标记为非语音指令，角色不会朗读。
                    </small>
                  </label>
                  <label className="dialog-field">
                    <span>负面提示词</span>
                    <textarea
                      rows="3"
                      value={shotDraft.negative_prompt || ""}
                      onChange={(event) =>
                        updateShotDraft("negative_prompt", event.target.value)
                      }
                    />
                  </label>
                </div>
                <div>
                  <div className="dialog-field-row">
                    <label className="dialog-field">
                      <span>任务</span>
                      <select
                        value={runtimeShotTask(project, shotDraft)}
                        onChange={(event) => updateShotTask(event.target.value)}
                      >
                        <option value="fl2va">FL2VA · 首帧/末帧生视频</option>
                        <option value="ref2va">Ref2VA · 参考视频续写</option>
                      </select>
                    </label>
                    <label className="dialog-field">
                      <span>时长（秒）</span>
                      <input
                        type="number"
                        min="4"
                        max="14"
                        step="0.5"
                        value={shotDraft.duration_seconds}
                        onChange={(event) =>
                          updateShotDraft(
                            "duration_seconds",
                            event.target.value,
                          )
                        }
                      />
                    </label>
                  </div>
                  <label className="dialog-field">
                    <span>摄影机</span>
                    <input
                      value={shotDraft.camera || ""}
                      onChange={(event) =>
                        updateShotDraft("camera", event.target.value)
                      }
                    />
                  </label>
                  <label className="dialog-field">
                    <span>边界策略</span>
                    <select
                      value={shotDraft.transition_kind || "continuous"}
                      onChange={(event) =>
                        updateShotDraft("transition_kind", event.target.value)
                      }
                    >
                      <option value="continuous">连续动作</option>
                      <option value="camera_move">同轴镜头运动</option>
                      <option value="match_cut">匹配剪辑</option>
                      <option value="occlusion_cut">遮挡转场</option>
                      <option value="hard_cut">硬切 / 新场景</option>
                      <option value="anchor">新首帧锚点</option>
                    </select>
                  </label>
                  <section className="shot-prompt-section h3-storyboard-editor">
                    <div className="shot-prompt-section-head">
                      <div>
                        <strong>H3 分镜时间线</strong>
                        <small>
                          Agent 生成的可观测状态、连续性与逐拍动作会直接编译进
                          H3 Prompt。
                        </small>
                      </div>
                    </div>
                    <label className="dialog-field">
                      <span>开场状态 · 第一帧可见事实</span>
                      <textarea
                        required
                        rows="3"
                        value={shotDraft.opening_state}
                        onChange={(event) =>
                          updateShotDraft("opening_state", event.target.value)
                        }
                      />
                    </label>
                    <label className="dialog-field">
                      <span>收尾状态 · 最后一帧可见事实</span>
                      <textarea
                        required
                        rows="3"
                        value={shotDraft.ending_state}
                        onChange={(event) =>
                          updateShotDraft("ending_state", event.target.value)
                        }
                      />
                    </label>
                    <label className="dialog-field">
                      <span>跨镜连续性 Handoff</span>
                      <textarea
                        required
                        rows="3"
                        value={shotDraft.continuity_handoff}
                        onChange={(event) =>
                          updateShotDraft(
                            "continuity_handoff",
                            event.target.value,
                          )
                        }
                        placeholder="身份、服装、道具、空间、运动方向、光线、机位与环境声。"
                      />
                    </label>
                    <label className="dialog-field">
                      <span>本镜 Hook · 单一注意力焦点</span>
                      <textarea
                        required
                        rows="2"
                        value={shotDraft.hook}
                        onChange={(event) =>
                          updateShotDraft("hook", event.target.value)
                        }
                      />
                    </label>
                    <label className="dialog-field">
                      <span>语义参考锚点 · 每行一个</span>
                      <textarea
                        required
                        rows="3"
                        value={(shotDraft.reference_anchors || []).join("\n")}
                        onChange={(event) =>
                          updateShotDraft(
                            "reference_anchors",
                            event.target.value.split("\n"),
                          )
                        }
                        placeholder="Character identity: …\nScene geography: …\nProp identity: …"
                      />
                    </label>
                    <div className="h3-beat-list">
                      {(shotDraft.visual_beats || []).map((beat, index) => (
                        <div
                          className="dialogue-line-card h3-beat-card"
                          key={`beat-${index}`}
                        >
                          <div className="dialogue-line-title">
                            <span>
                              BEAT {String(index + 1).padStart(2, "0")}
                            </span>
                            <small>
                              {Number(beat.start_seconds).toFixed(1)}s–
                              {Number(beat.end_seconds).toFixed(1)}s
                            </small>
                          </div>
                          <div className="dialog-field-row">
                            <label className="dialog-field">
                              <span>开始秒数</span>
                              <input
                                required
                                type="number"
                                min="0"
                                step="0.1"
                                value={beat.start_seconds}
                                onChange={(event) =>
                                  updateVisualBeat(
                                    index,
                                    "start_seconds",
                                    event.target.value,
                                  )
                                }
                              />
                            </label>
                            <label className="dialog-field">
                              <span>结束秒数</span>
                              <input
                                required
                                type="number"
                                min="0.1"
                                max={shotDraft.duration_seconds}
                                step="0.1"
                                value={beat.end_seconds}
                                onChange={(event) =>
                                  updateVisualBeat(
                                    index,
                                    "end_seconds",
                                    event.target.value,
                                  )
                                }
                              />
                            </label>
                          </div>
                          <label className="dialog-field">
                            <span>唯一主动作</span>
                            <textarea
                              required
                              rows="2"
                              value={beat.visual_action}
                              onChange={(event) =>
                                updateVisualBeat(
                                  index,
                                  "visual_action",
                                  event.target.value,
                                )
                              }
                            />
                          </label>
                          <label className="dialog-field">
                            <span>状态变化</span>
                            <textarea
                              rows="2"
                              value={beat.state_change}
                              onChange={(event) =>
                                updateVisualBeat(
                                  index,
                                  "state_change",
                                  event.target.value,
                                )
                              }
                            />
                          </label>
                          <label className="dialog-field">
                            <span>镜头运动 · 类型 / 幅度 / 速度</span>
                            <input
                              value={beat.camera}
                              onChange={(event) =>
                                updateVisualBeat(
                                  index,
                                  "camera",
                                  event.target.value,
                                )
                              }
                            />
                          </label>
                          <label className="dialog-field">
                            <span>与动作同步的声音</span>
                            <input
                              value={beat.sound}
                              onChange={(event) =>
                                updateVisualBeat(
                                  index,
                                  "sound",
                                  event.target.value,
                                )
                              }
                            />
                          </label>
                          <label className="dialog-field">
                            <span>表演与表情</span>
                            <textarea
                              rows="2"
                              value={beat.performance || ""}
                              onChange={(event) =>
                                updateVisualBeat(index, "performance", event.target.value)
                              }
                              placeholder="重心、眼神、面部微表情与动作阶段。"
                            />
                          </label>
                          <label className="dialog-field">
                            <span>屏幕空间锚点</span>
                            <input
                              value={beat.spatial_anchor || ""}
                              onChange={(event) =>
                                updateVisualBeat(index, "spatial_anchor", event.target.value)
                              }
                              placeholder="左/右位置、固定地标、前中后景。"
                            />
                          </label>
                          <label className="dialog-field">
                            <span>Beat handoff</span>
                            <input
                              value={beat.handoff || ""}
                              onChange={(event) =>
                                updateVisualBeat(index, "handoff", event.target.value)
                              }
                              placeholder="下一拍继承的姿态或道具状态。"
                            />
                          </label>
                        </div>
                      ))}
                    </div>
                  </section>
                  <section className="shot-prompt-section">
                    <div className="shot-prompt-section-head">
                      <div>
                        <strong>声音设计</strong>
                        <small>
                          环境声、画内音效与观众听到的配乐分开描述。
                        </small>
                      </div>
                    </div>
                    <label className="dialog-field">
                      <span>环境声与动作音效 · 角色可听</span>
                      <textarea
                        rows="4"
                        value={shotDraft.audio_prompt}
                        onChange={(event) =>
                          updateShotDraft("audio_prompt", event.target.value)
                        }
                        placeholder="环境底噪、脚步、衣物、风声、物体碰撞等；不要写对白。"
                      />
                    </label>
                    <label className="dialog-field">
                      <span>非画内配乐 · 仅观众可听</span>
                      <textarea
                        rows="3"
                        value={shotDraft.music_prompt}
                        onChange={(event) =>
                          updateShotDraft("music_prompt", event.target.value)
                        }
                        placeholder="留空表示无配乐。"
                      />
                    </label>
                  </section>
                  <section className="shot-prompt-section dialogue-editor">
                    <div className="shot-prompt-section-head">
                      <div>
                        <strong>角色对白</strong>
                        <small>
                          只有这里的原文会被角色说出来；留空即强制无对白、无旁白。
                        </small>
                      </div>
                      <button type="button" onClick={addDialogueLine}>
                        <Plus size={13} /> 添加对白
                      </button>
                    </div>
                    {(shotDraft.dialogue || []).length ? (
                      shotDraft.dialogue.map((line, index) => (
                        <div
                          className="dialogue-line-card"
                          key={`dialogue-${index}`}
                        >
                          <div className="dialogue-line-title">
                            <span>
                              对白 {String(index + 1).padStart(2, "0")}
                            </span>
                            <button
                              type="button"
                              aria-label={`删除对白 ${index + 1}`}
                              onClick={() => removeDialogueLine(index)}
                            >
                              <Trash2 size={13} />
                            </button>
                          </div>
                          <div className="dialog-field-row">
                            <label className="dialog-field">
                              <span>说话人</span>
                              <input
                                required
                                value={line.speaker}
                                onChange={(event) =>
                                  updateDialogueLine(
                                    index,
                                    "speaker",
                                    event.target.value,
                                  )
                                }
                                placeholder="角色名"
                              />
                            </label>
                            <label className="dialog-field">
                              <span>语言</span>
                              <input
                                required
                                value={line.language}
                                onChange={(event) =>
                                  updateDialogueLine(
                                    index,
                                    "language",
                                    event.target.value,
                                  )
                                }
                                placeholder="Chinese"
                              />
                            </label>
                          </div>
                          <label className="dialog-field">
                            <span>准确原文 · 不要写角色名或动作</span>
                            <textarea
                              required
                              rows="3"
                              value={line.text}
                              onChange={(event) =>
                                updateDialogueLine(
                                  index,
                                  "text",
                                  event.target.value,
                                )
                              }
                              placeholder="只填写需要被说出的原句。"
                            />
                          </label>
                          <label className="dialog-field">
                            <span>说话方式</span>
                            <input
                              required
                              value={line.delivery}
                              onChange={(event) =>
                                updateDialogueLine(
                                  index,
                                  "delivery",
                                  event.target.value,
                                )
                              }
                              placeholder="克制、兴奋、低声警惕……"
                            />
                          </label>
                          <label className="dialog-field">
                            <span>声音来源</span>
                            <select
                              value={line.mode || "on_screen"}
                              onChange={(event) =>
                                updateDialogueLine(
                                  index,
                                  "mode",
                                  event.target.value,
                                )
                              }
                            >
                              <option value="on_screen">
                                画内角色说话 · 对口型
                              </option>
                              <option value="off_screen">
                                画外角色说话 · 画内人物闭嘴
                              </option>
                              <option value="voice_over">
                                旁白 / 内心声 · 画内人物闭嘴
                              </option>
                            </select>
                          </label>
                          <div className="dialog-field-row">
                            <label className="dialog-field">
                              <span>开始秒数（可选）</span>
                              <input
                                type="number"
                                min="0"
                                step="0.1"
                                value={line.start_seconds ?? ""}
                                onChange={(event) =>
                                  updateDialogueLine(
                                    index,
                                    "start_seconds",
                                    event.target.value,
                                  )
                                }
                              />
                            </label>
                            <label className="dialog-field">
                              <span>结束秒数（可选）</span>
                              <input
                                type="number"
                                min="0"
                                max={shotDraft.duration_seconds}
                                step="0.1"
                                value={line.end_seconds ?? ""}
                                onChange={(event) =>
                                  updateDialogueLine(
                                    index,
                                    "end_seconds",
                                    event.target.value,
                                  )
                                }
                              />
                            </label>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="dialogue-empty">
                        本镜头无对白，H3 将被明确要求保持无语音。
                      </div>
                    )}
                  </section>
                  <div className="dialog-field-row">
                    <label className="dialog-field">
                      <span>Steps</span>
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={shotDraft.inference_steps}
                        onChange={(event) =>
                          updateShotDraft("inference_steps", event.target.value)
                        }
                      />
                    </label>
                    <label className="dialog-field">
                      <span>Seed</span>
                      <input
                        type="number"
                        value={shotDraft.seed}
                        onChange={(event) =>
                          updateShotDraft("seed", event.target.value)
                        }
                      />
                    </label>
                  </div>
                  <label className="dialog-field">
                    <span>承接上一镜</span>
                    <select
                      value={shotDraft.continuity_from_shot_id || ""}
                      onChange={(event) =>
                        updateShotDraft(
                          "continuity_from_shot_id",
                          event.target.value || null,
                        )
                      }
                    >
                      <option value="">不承接 / 独立起镜</option>
                      {(project.shots || [])
                        .filter((shot) => shot.index < editingShot.index)
                        .map((shot) => (
                          <option key={shot.id} value={shot.id}>
                            {String(shot.index + 1).padStart(2, "0")} ·{" "}
                            {shot.title}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label className="dialog-field">
                    <span>续写方式（留空继承项目设定）</span>
                    <select
                      value={shotDraft.continuation_mode || ""}
                      onChange={(event) =>
                        updateShotDraft(
                          "continuation_mode",
                          event.target.value || null,
                        )
                      }
                    >
                      <option value="">继承项目设定</option>
                      <option value="fast">快速续写 · 最后 5 秒</option>
                      <option value="quality">高质量续写 · 完整上一镜</option>
                    </select>
                  </label>
                  <label className="dialog-field">
                    <span>显式首帧（优先于自动合成）</span>
                    <select
                      value={shotDraft.start_frame_asset_id || ""}
                      onChange={(event) =>
                        updateShotDraft(
                          "start_frame_asset_id",
                          event.target.value || null,
                        )
                      }
                    >
                      <option value="">未指定，由素材与上下文自动生成</option>
                      {assets
                        .filter((asset) => asset.kind === "image")
                        .map((asset) => (
                          <option key={asset.id} value={asset.id}>
                            {assetLabel(asset)}
                          </option>
                        ))}
                    </select>
                  </label>
                  <div className="shot-reference-picker">
                    <span>本镜参考素材</span>
                    <div>
                      {assets.map((asset) => (
                        <button
                          type="button"
                          key={asset.id}
                          className={
                            shotDraft.reference_asset_ids.includes(asset.id)
                              ? "selected"
                              : ""
                          }
                          onClick={() => toggleShotReference(asset.id)}
                        >
                          {shotDraft.reference_asset_ids.includes(asset.id) ? (
                            <Check size={12} />
                          ) : (
                            <Plus size={12} />
                          )}
                          {assetLabel(asset)}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              <div className="dialog-actions">
                <button
                  className="outline-button"
                  type="button"
                  onClick={() => setEditingShot(null)}
                >
                  取消
                </button>
                <button
                  className="glow-button"
                  type="submit"
                  disabled={dialogSaving}
                >
                  {dialogSaving ? "保存中…" : "保存镜头"}
                </button>
              </div>
            </motion.form>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {editingAsset ? (
          <motion.div
            className="asset-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onMouseDown={() => setEditingAsset(null)}
          >
            <motion.form
              className="asset-modal"
              initial={{ opacity: 0, scale: 0.97, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.98 }}
              onSubmit={saveAsset}
              onMouseDown={(event) => event.stopPropagation()}
            >
              <div className="asset-modal-head">
                <div>
                  <span className="overline">MATERIAL METADATA</span>
                  <h2>编辑素材</h2>
                  <small>{editingAsset.original_name}</small>
                </div>
                <button
                  type="button"
                  onClick={() => setEditingAsset(null)}
                  aria-label="关闭素材编辑"
                >
                  <X size={17} />
                </button>
              </div>
              <label className="asset-modal-field">
                <span>素材名称 / 角色名</span>
                <input
                  value={assetDraft.display_name}
                  onChange={(event) =>
                    setAssetDraft((current) => ({
                      ...current,
                      display_name: event.target.value,
                    }))
                  }
                  placeholder="例如：白鹿、孟子义、太和殿"
                />
              </label>
              <label className="asset-modal-field">
                <span>描述（可选）</span>
                <input
                  value={assetDraft.caption}
                  onChange={(event) =>
                    setAssetDraft((current) => ({
                      ...current,
                      caption: event.target.value,
                    }))
                  }
                  placeholder="例如：女主角正面定妆照，红色宫装"
                />
              </label>
              <label className="asset-modal-field">
                <span>标签</span>
                <input
                  value={assetDraft.tags}
                  onChange={(event) =>
                    setAssetDraft((current) => ({
                      ...current,
                      tags: event.target.value,
                    }))
                  }
                  placeholder="用逗号分隔，例如：女主、红衣、夜景"
                />
              </label>
              <fieldset className="asset-role-field">
                <legend>素材角色（可多选）</legend>
                <div>
                  {Object.entries(roleLabel).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      className={
                        assetDraft.roles.includes(value) ? "selected" : ""
                      }
                      onClick={() => toggleDraftRole(value)}
                    >
                      {assetDraft.roles.includes(value) ? (
                        <Check size={13} />
                      ) : (
                        <Plus size={13} />
                      )}
                      {label}
                    </button>
                  ))}
                </div>
              </fieldset>
              <div className="asset-modal-actions">
                <button
                  className="outline-button asset-delete-button"
                  type="button"
                  onClick={deleteAsset}
                  disabled={assetDeleting || assetSaving}
                >
                  <Trash2 size={14} />
                  {assetDeleting ? "删除中…" : "删除素材"}
                </button>
                <button
                  className="outline-button"
                  type="button"
                  onClick={() => setEditingAsset(null)}
                  disabled={assetSaving || assetDeleting}
                >
                  取消
                </button>
                <button
                  className="glow-button"
                  type="submit"
                  disabled={assetSaving || assetDeleting}
                >
                  {assetSaving ? "保存中…" : "保存素材"}
                </button>
              </div>
            </motion.form>
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {notice ? (
          <motion.div
            className="toast"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            onAnimationComplete={() => setTimeout(() => setNotice(""), 3600)}
          >
            <Sparkles size={15} />
            {notice}
            <button onClick={() => setNotice("")}>
              <X size={14} />
            </button>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
