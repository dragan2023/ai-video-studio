// Suno 音乐生成脚本模板（客户端预置，供画布 model-plugin 执行）
// baseUrl / apiKey 由画布「配置 → 渠道」注入（用户后填，不在代码硬编码）。
// 协议按常见 Suno API（POST /v1/music 提交、GET /v1/songs/{id} 轮询）适配；
// 若你的 Suno 端点不同，可在画布「模型脚本编辑器」里直接改这个脚本。

export const SUNO_MUSIC_SCRIPT = `// Suno 音乐生成（提交任务 → 轮询 → 返回音频 URL）
// 可用变量：prompt / model / baseUrl / apiKey / http / request / poll / sleep / signal
const submit = await request({
  method: "post",
  url: \`\${baseUrl}/v1/music\`,
  headers: { "Content-Type": "application/json", Authorization: \`Bearer \${apiKey}\` },
  data: { prompt, model: model || "chirp-v3" },
});
const direct = submit?.data?.[0] || submit?.data || submit;
const directUrl = direct?.audio_url || direct?.music_url || direct?.song_url || direct?.url;
if (directUrl) return { url: directUrl };
const taskId = submit?.id || submit?.task_id || direct?.id;
if (!taskId) throw new Error("suno: 无法识别音频 URL 或任务 ID（请按你的 Suno API 调整脚本）");
const result = await poll(
  async () => {
    const res = await request({
      method: "get",
      url: \`\${baseUrl}/v1/songs/\${taskId}\`,
      headers: { Authorization: \`Bearer \${apiKey}\` },
    });
    const item = res?.data?.[0] || res?.data || res;
    if (item?.audio_url || item?.music_url || item?.song_url || item?.url) {
      return { url: item.audio_url || item.music_url || item.song_url || item.url };
    }
    if (item?.status === "error" || item?.status === "failed") throw new Error("suno: " + (item?.error || "生成失败"));
    return false;
  },
  (v) => v,
  { intervalMs: 5000, timeoutMs: 180000 },
);
return result;
`;

