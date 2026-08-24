from long_video_studio.domain import FilmProject, ProjectBrief, ShotSpec, StartFrameSource, TransitionKind, WorldBible
from long_video_studio.preproduction import PreproductionPlanner


def project(*shots):
    return FilmProject(brief=ProjectBrief(prompt="test film"), world_bible=WorldBible(logline="test", visual_style="cinematic"), shots=list(shots))


def shot(index, source, refs=None):
    return ShotSpec(index=index, title=f"Shot {index}", purpose="test", duration_seconds=5, prompt="cinematic test", source_section=source, reference_asset_ids=refs or [])


def test_black_shot_uses_system_black_without_generation():
    plan = PreproductionPlanner().plan(project(shot(0, "剪辑与动作：本镜黑场字幕镜。")), [])
    assert plan.shots[0].start_frame_source == StartFrameSource.SYSTEM_BLACK
    assert plan.generated_image_count == 0


def test_match_cut_is_independent_and_can_generate_missing_reference():
    plan = PreproductionPlanner().plan(project(shot(0, "剪辑与动作：与上一镜形成匹配剪辑，硬切到水面。")), [])
    assert plan.shots[0].transition_kind == TransitionKind.MATCH_CUT
    assert plan.shots[0].start_frame_source == StartFrameSource.GENERATE_T2I


def test_explicit_action_continuity_uses_previous_boundary():
    plan = PreproductionPlanner().plan(project(shot(0, "剪辑与动作：黑场字幕镜。"), shot(1, "剪辑与动作：承接上一镜动作余势，保持空间轴线。")), [])
    assert plan.shots[1].start_frame_source == StartFrameSource.PREVIOUS_BOUNDARY
    assert plan.shots[1].source_shot_id == plan.shots[0].shot_id


def test_ambiguous_independent_shot_requires_review():
    plan = PreproductionPlanner().plan(project(shot(0, "一段没有剪辑说明的普通画面。")), [])
    assert plan.shots[0].start_frame_source == StartFrameSource.NEEDS_REVIEW
    assert plan.blockers
