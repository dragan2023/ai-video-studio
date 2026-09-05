"""Visible, deterministic preproduction decisions derived from imported shot evidence."""

from __future__ import annotations

from dataclasses import dataclass

from long_video_studio.domain import (
    AssetKind,
    AssetRecord,
    FilmProject,
    PreproductionPlan,
    PreproductionShotPlan,
    PreproductionStatus,
    StartFrameSource,
    TransitionKind,
)


@dataclass(frozen=True)
class TransitionDecision:
    kind: TransitionKind
    evidence: str
    confidence: float
    independent: bool


class PreproductionPlanner:
    """Create reviewable first-frame decisions without making provider calls."""

    _BLACK_TERMS = ("黑场", "黑屏", "字幕镜")
    _HARD_TERMS = ("硬切", "切至", "换场", "空间切换", "闪回", "时间跳转")
    _MATCH_TERMS = ("匹配剪辑", "声音桥", "亮度匹配", "遮罩转场", "遮镜")
    _CONTINUITY_TERMS = ("承接", "动作余势", "视线", "轴线", "顺势", "行进方向")

    def plan(self, project: FilmProject, assets: list[AssetRecord]) -> PreproductionPlan:
        image_asset_ids = {asset.id for asset in assets if asset.kind == AssetKind.IMAGE}
        rows: list[PreproductionShotPlan] = []
        blockers: list[str] = []
        generated_count = 0
        ordered = sorted(project.shots, key=lambda item: item.index)
        for position, shot in enumerate(ordered):
            decision = self._transition(shot.source_section)
            previous = ordered[position - 1] if position else None
            source, selected_asset_id, source_shot_id, gap_reason, permitted = self._frame_source(
                shot=shot,
                previous=previous,
                decision=decision,
                image_asset_ids=image_asset_ids,
            )
            if source == StartFrameSource.GENERATE_T2I:
                generated_count += 1
            if source == StartFrameSource.NEEDS_REVIEW:
                blockers.append(f"镜头 {shot.index + 1} 缺少可用首帧：{gap_reason}")
            rows.append(
                PreproductionShotPlan(
                    shot_id=shot.id,
                    shot_index=shot.index,
                    script_evidence=decision.evidence,
                    transition_kind=decision.kind,
                    start_frame_source=source,
                    source_shot_id=source_shot_id,
                    candidate_asset_ids=[asset_id for asset_id in shot.reference_asset_ids if asset_id in image_asset_ids],
                    selected_asset_id=selected_asset_id,
                    gap_reason=gap_reason,
                    confidence=decision.confidence,
                    generation_permitted=permitted,
                    parameter_summary=self._summary(shot),
                )
            )
        return PreproductionPlan(
            asset_input_fingerprint="|".join(sorted(image_asset_ids)),
            generated_image_count=generated_count,
            blockers=blockers,
            status=PreproductionStatus.BLOCKED if blockers else PreproductionStatus.AWAITING_APPROVAL,
            shot_plans=rows,
        )

    def _transition(self, source_section: str) -> TransitionDecision:
        text = source_section or ""
        if self._contains(text, self._BLACK_TERMS):
            return TransitionDecision(TransitionKind.HARD_CUT, text, 1.0, True)
        if self._contains(text, self._MATCH_TERMS):
            kind = TransitionKind.OCCLUSION_CUT if self._contains(text, ("遮罩", "遮镜")) else TransitionKind.MATCH_CUT
            return TransitionDecision(kind, text, 0.95, True)
        if self._contains(text, self._HARD_TERMS):
            return TransitionDecision(TransitionKind.HARD_CUT, text, 0.95, True)
        if self._contains(text, self._CONTINUITY_TERMS):
            return TransitionDecision(TransitionKind.CONTINUOUS, text, 0.8, False)
        return TransitionDecision(TransitionKind.ANCHOR, text, 0.35, True)

    def _frame_source(self, *, shot, previous, decision, image_asset_ids):
        if shot.start_frame_asset_id:
            return StartFrameSource.CREATOR_ASSET, shot.start_frame_asset_id, None, "creator selected opening frame", False
        if self._contains(shot.source_section, self._BLACK_TERMS):
            return StartFrameSource.SYSTEM_BLACK, None, None, "black/subtitle shot", False
        candidates = [asset_id for asset_id in shot.reference_asset_ids if asset_id in image_asset_ids]
        if not decision.independent and previous is not None:
            return StartFrameSource.PREVIOUS_BOUNDARY, None, previous.id, "explicit visual continuity", False
        if candidates:
            return StartFrameSource.CREATOR_ASSET, candidates[0], None, "independent shot uses creator reference", False
        if decision.confidence >= 0.8:
            return StartFrameSource.GENERATE_T2I, None, None, "independent shot has no image reference", True
        return StartFrameSource.NEEDS_REVIEW, None, None, "transition evidence is insufficient", False

    @staticmethod
    def _contains(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _summary(shot) -> str:
        return f"{shot.duration_seconds:g}s | {shot.task.value} | {shot.camera} | {shot.prompt}"
