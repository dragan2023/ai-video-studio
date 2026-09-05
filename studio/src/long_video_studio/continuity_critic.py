"""厚版直通连续性审查（LLM）。

对已解析的相邻分镜做一致性审查，输出问题清单供人工审核。
仅在配置了 planner LLM（base_url + api_key + model）时可用；未配置则跳过。
"""

from __future__ import annotations

import json
import logging

import httpx

from long_video_studio.config import Settings
from long_video_studio.domain import FilmProject, ShotSpec, WorldBible

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是动画电影分镜连续性审查员。对比相邻两个分镜，找出跨镜头的一致性风险。"
    "只报告确有问题或高风险的点，不要改写镜头内容。输出严格 JSON。"
)

REVIEW_DIMENSIONS = (
    "identity(角色身份/面部/发型/体型)",
    "wardrobe(服装/道具)",
    "scene(场景/地标)",
    "lighting(光线方向/色调)",
    "motion(运动方向/运镜)",
    "action_replay(动作重复)",
    "dialogue(对白/口型/音色)",
)


class ContinuityCritic:
    def __init__(self, settings: Settings):
        self.base_url = (settings.planner_base_url or "").rstrip("/")
        self.api_key = settings.planner_api_key
        self.model = settings.planner_model

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    async def review_pair(self, shot_a: ShotSpec, shot_b: ShotSpec) -> list[dict]:
        """审查一对相邻镜头，返回问题列表。"""
        if not self.available:
            return []

        user_prompt = self._pair_prompt(shot_a, shot_b)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as error:
            logger.warning("continuity critic request failed: %s", error)
            return []

        content = payload["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        return data.get("issues", [])

    async def review_project(self, project: FilmProject, *, limit: int | None = None) -> dict[str, list[dict]]:
        """审查项目内相邻镜头对（默认前 N 镜）。"""
        shots = sorted(project.shots, key=lambda shot: shot.index)
        if limit is not None:
            shots = shots[:limit]
        issues: dict[str, list[dict]] = {}
        for index in range(len(shots) - 1):
            pair_issues = await self.review_pair(shots[index], shots[index + 1])
            if pair_issues:
                issues[f"{shots[index].index + 1:02d}→{shots[index + 1].index + 1:02d}"] = pair_issues
        return issues

    @staticmethod
    def _pair_prompt(shot_a: ShotSpec, shot_b: ShotSpec) -> str:
        return (
            f"前镜（{shot_a.title}，{shot_a.duration_seconds:g}s）：\n"
            f"{_clip(shot_a.prompt, 900)}\n\n"
            f"后镜（{shot_b.title}，{shot_b.duration_seconds:g}s）：\n"
            f"{_clip(shot_b.prompt, 900)}\n\n"
            f"请从这些维度审查衔接风险：{', '.join(REVIEW_DIMENSIONS)}。\n"
            '返回 JSON：{"issues":[{"type":"<维度>","detail":"<一句话问题>","severity":"high|medium|low"}]}。'
            '没有问题时返回 {"issues":[]}。'
        )


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
