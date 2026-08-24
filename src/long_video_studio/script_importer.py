"""厚版分镜脚本解析器（通用型）。

解析用户提供的"厚版"Markdown 分镜脚本，输出结构化镜头数据与解析报告。
脚本格式约定（已对《极乐城》场1厚版 70 镜实测）：

    ### 1-01 · 黑屏心跳（5s｜文戏）

    @图片1：R-01,@图片2：S-03        （可选资产对照行）

    生成一段5秒、...故事背景：...     （厚版完整提示词正文）
    0—1.60秒：...                    （时间分段）
    剪辑与动作：...
    视觉风格：...
    声音设计：...

本模块只做确定性解析，不调用 LLM，不写死任何具体剧目内容。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 镜头标题：### 1-01 · 黑屏心跳（5s｜文戏）
_SHOT_HEADER_RE = re.compile(
    r"^###\s+(?P<no>\d+-\d+)\s*·\s*(?P<title>.+?)\s*"
    r"[(（]\s*(?P<dur>\d+(?:\.\d+)?)\s*s\s*[｜|]\s*(?P<kind>文戏|武戏)(?:·[^）)]*)?\s*[)）]"
)

# 资产对照行：@图片1：R-01,@图片2：S-03 或 @图片1：S-02与P-07
_ASSET_LINE_RE = re.compile(r"^@图片(?P<idx>\d+)\s*：\s*(?P<codes>[^@\n]+)")

# 编号分隔符（逗号/顿号/与/和/加号/空白）
_CODE_SPLIT_RE = re.compile(r"[,，、与和+\s]+")

# 时间分段：0—1.60秒：... 或 2.20—4.10秒：...
_BEAT_RE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*[—–-]\s*(?P<end>\d+(?:\.\d+)?)\s*秒\s*[：:]?\s*(?P<action>.+)"
)

# 中文双引号/直角引号台词（单引号包裹避免与 ASCII 引号冲突）
_QUOTE_CN_RE = re.compile(r'[“「]([^”」]{2,})[”」]')
_QUOTE_EN_RE = re.compile(r'"([^"]{2,})"')

# 台词/口型标记（明确标记才计数；"无对白"不在此列，避免误报）
_DIALOGUE_FLAG_RE = re.compile(r"台词|口型同步|念白|旁白|口型")

# 说话动词（不消费引号）；去掉"道/叫"等高频误报字，负向后顾排除"说明/解说/传说/小说/叙说/询问/答案"等
_SPEECH_VERB_RE = re.compile(
    r"(?<![明解传小叙诉询疑提学纪思怀告解访悬])(说|问|答|念|唱|喊|呢喃|低语)\s*[：:]?\s*"
)

# 段落标题（声音设计单独抽取供音频参考）
_SECTION_RE = re.compile(r"^(剪辑与动作|视觉风格|声音设计|剪辑|动作|声音)\s*[：:]")

# 故事背景摘要
_STORY_BG_RE = re.compile(r"故事背景[：:](.+?)(?:$|\n)")


@dataclass
class RawDialogue:
    """解析出的台词原文（speaker 尽力推断，允许占位）。"""

    speaker: str
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass
class RawBeat:
    """解析出的时间分段。"""

    start_seconds: float
    end_seconds: float
    action: str


@dataclass
class RawShot:
    shot_no: str
    index: int
    title: str
    duration_seconds: float
    shot_type: str  # 文戏 / 武戏
    refs: dict[int, list[str]] = field(default_factory=dict)
    prompt: str = ""
    audio_prompt: str = ""
    camera: str = ""
    story_background: str = ""
    beats: list[RawBeat] = field(default_factory=list)
    dialogue: list[RawDialogue] = field(default_factory=list)
    dialogue_flagged: bool = False
    source_section: str = ""

    @property
    def has_refs(self) -> bool:
        return bool(self.refs)

    @property
    def ordered_ref_codes(self) -> list[str]:
        """按 @图片N 顺序展开的资产编号列表（去重保序）。"""
        codes: list[str] = []
        seen: set[str] = set()
        for idx in sorted(self.refs):
            for code in self.refs[idx]:
                if code not in seen:
                    seen.add(code)
                    codes.append(code)
        return codes


@dataclass
class ParseIssue:
    level: str  # info / warning / error
    shot_no: str
    message: str


@dataclass
class ScriptParseResult:
    shots: list[RawShot] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)
    total_lines: int = 0
    no_ref_shots: list[str] = field(default_factory=list)
    dialogue_shots: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "warning")

    def summary(self) -> str:
        return (
            f"解析完成：{len(self.shots)} 镜，"
            f"错误 {self.error_count}，警告 {self.warning_count}，"
            f"无图镜 {len(self.no_ref_shots)}，台词镜 {len(self.dialogue_shots)}"
        )


def parse_shot_script(text: str) -> ScriptParseResult:
    """解析厚版分镜脚本。"""
    lines = text.splitlines()
    result = ScriptParseResult(total_lines=len(lines))

    # 第一遍：找到所有镜头标题行的位置
    boundaries: list[int] = []
    for line_no, line in enumerate(lines):
        if _SHOT_HEADER_RE.match(line.strip()):
            boundaries.append(line_no)

    if not boundaries:
        result.issues.append(ParseIssue("error", "-", "未识别到任何镜头标题（### 镜号 · 标题（Ns｜类型））"))
        return result

    for position, start_line in enumerate(boundaries):
        end_line = boundaries[position + 1] if position + 1 < len(boundaries) else len(lines)
        block = lines[start_line:end_line]
        shot = _parse_shot_block(block, position)
        if shot is None:
            continue
        result.shots.append(shot)
        if not shot.has_refs:
            result.no_ref_shots.append(shot.shot_no)
        if shot.dialogue_flagged:
            result.dialogue_shots.append(shot.shot_no)

    return result


def _parse_shot_block(block: list[str], position: int) -> RawShot | None:
    header_line = block[0].strip()
    match = _SHOT_HEADER_RE.match(header_line)
    if not match:
        return None

    shot_no = match.group("no")
    title = match.group("title").strip()
    duration = float(match.group("dur"))
    shot_type = match.group("kind")

    # 收集正文（去掉标题行）
    body: list[str] = []
    refs: dict[int, list[str]] = {}
    for raw_line in block[1:]:
        line = raw_line.strip()
        if not line:
            continue
        asset_match = _ASSET_LINE_RE.match(line)
        if asset_match:
            idx = int(asset_match.group("idx"))
            codes = [code.strip() for code in _CODE_SPLIT_RE.split(asset_match.group("codes").strip()) if code.strip()]
            refs.setdefault(idx, []).extend(codes)
            continue
        body.append(raw_line)

    prompt = "\n".join(line.strip() for line in body if line.strip()).strip()

    audio_prompt = _extract_section(body, "声音设计")
    camera = _extract_section(body, "剪辑与动作") or _extract_section(body, "剪辑") or ""
    story_background = _extract_story_background(body)

    beats = _extract_beats(body)
    dialogue, dialogue_flagged = _extract_dialogue(body)

    shot = RawShot(
        shot_no=shot_no,
        index=position,
        title=title,
        duration_seconds=duration,
        shot_type=shot_type,
        refs=refs,
        prompt=prompt,
        audio_prompt=audio_prompt,
        camera=camera,
        story_background=story_background,
        beats=beats,
        dialogue=dialogue,
        dialogue_flagged=dialogue_flagged,
        source_section=f"{block[0].strip()}\n{prompt}",
    )
    return shot


def _extract_section(body: list[str], name: str) -> str:
    """抽取段落标题下的内容（到下一个段落标题或结尾）。"""
    capturing = False
    collected: list[str] = []
    for raw_line in body:
        line = raw_line.strip()
        section_match = _SECTION_RE.match(line)
        if section_match:
            capturing = section_match.group(1) == name
            if capturing:
                # 标题行本身可能带内容
                content = _SECTION_RE.sub("", line).strip()
                if content:
                    collected.append(content)
            continue
        if capturing and line:
            collected.append(line)
    return "\n".join(collected).strip()


def _extract_story_background(body: list[str]) -> str:
    for raw_line in body:
        line = raw_line.strip()
        match = _STORY_BG_RE.search(line)
        if match:
            value = match.group(1).strip()
            # 截断到句号，避免吞掉后续内容
            for end_char in ("。", "；"):
                cut = value.find(end_char)
                if cut > 0:
                    value = value[: cut + 1]
                    break
            return value
    return ""


def _extract_beats(body: list[str]) -> list[RawBeat]:
    beats: list[RawBeat] = []
    for raw_line in body:
        line = raw_line.strip()
        match = _BEAT_RE.search(line)
        if match:
            try:
                beats.append(
                    RawBeat(
                        start_seconds=float(match.group("start")),
                        end_seconds=float(match.group("end")),
                        action=match.group("action").strip(),
                    )
                )
            except ValueError:
                continue
    return beats


def _extract_dialogue(body: list[str]) -> tuple[list[RawDialogue], bool]:
    """只提取"说话动词 + 引号"的口型台词，忽略字幕/强调性引号。"""
    dialogue: list[RawDialogue] = []
    flagged = False

    for raw_line in body:
        line = raw_line.strip()
        for verb_match in _SPEECH_VERB_RE.finditer(line):
            tail = line[verb_match.end():]
            quote = _QUOTE_CN_RE.search(tail) or _QUOTE_EN_RE.search(tail)
            if not quote:
                continue
            text = quote.group(1).strip()
            speaker = _infer_speaker(line[: verb_match.start()])
            dialogue.append(RawDialogue(speaker=speaker, text=text))

    # 去重 + 子串合并（"加入极乐。"与"加入极乐。我们替您..."取长句）
    dialogue.sort(key=lambda item: len(item.text), reverse=True)
    unique: list[RawDialogue] = []
    for item in dialogue:
        if any(item.text in kept.text for kept in unique):
            continue
        unique.append(item)
    unique.reverse()
    # 只有真的提取到台词原文才算台词镜（"无明确对白的剪影对话"不算）
    return unique, bool(unique)


def _infer_speaker(prefix: str) -> str:
    """从说话动词前的上下文尽力推断说话人。"""
    prefix = prefix.strip()
    # 去掉常见表演/状态修饰词
    prefix = re.sub(
        r"(开口|低声|大声|轻声|轻轻|缓缓|冷冷|平静|沙哑|甜美|电子|温柔|慢慢|语气|继续|随后|突然|缓缓地|轻声地)$",
        "",
        prefix,
    ).strip()
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9·]{1,8})$", prefix)
    if match:
        candidate = match.group(1).strip()
        if candidate and not re.fullmatch(r"(的|地|得|了|着|在|把|被|与|和|，|,|。|：|:)", candidate):
            return candidate
    return "speaker"


def parse_shot_script_file(path: str | Path) -> ScriptParseResult:
    """从文件解析（UTF-8）。"""
    return parse_shot_script(Path(path).read_text(encoding="utf-8"))
