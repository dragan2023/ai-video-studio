from long_video_studio.domain import BatchPlanningRun, BatchPlanningStatus, FilmProject, ProjectBrief, WorldBible


def test_batch_planning_run_is_optional_for_legacy_projects():
    project = FilmProject(brief=ProjectBrief(prompt="A legacy project."), world_bible=WorldBible(logline="Legacy", visual_style="cinematic"), shots=[])
    assert project.batch_planning_run is None


def test_batch_planning_run_persists_progress_without_credentials():
    run = BatchPlanningRun(status=BatchPlanningStatus.RUNNING, profile_id="qwen", model="qwen-plus", completed_shot_ids=["shot_1"])
    assert run.completed_count == 1
    assert "key" not in run.model_dump_json().lower()
