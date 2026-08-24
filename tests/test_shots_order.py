from __future__ import annotations

from fastapi.testclient import TestClient

from long_video_studio.app import create_app

_THREE_SHOT_SCRIPT = """### 1-01 · 开场（4s｜文戏）

生成一段4秒、16:9、原生立体声的黑色开场。

0—4秒：黑色背景。

剪辑与动作：单镜头。

视觉风格：纯黑。

声音设计：心跳，无对白。

### 1-02 · 中段（4s｜文戏）

生成一段4秒、16:9、原生立体声的城市中景。

0—4秒：城市灯光。

剪辑与动作：单镜头。

视觉风格：赛博。

声音设计：城市轰鸣，无对白。

### 1-03 · 结尾（4s｜文戏）

生成一段4秒、16:9、原生立体声的城市夜景。

0—4秒：城市夜景。

剪辑与动作：单镜头。

视觉风格：赛博霓虹。

声音设计：环境声，无对白。
"""


def _build_three_shot_project(client: TestClient) -> tuple[str, list[str]]:
    response = client.post(
        "/api/projects/import-thick-script",
        data={"title": "重排测试", "asset_root": ""},
        files={"script": ("test.md", _THREE_SHOT_SCRIPT.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200, response.text
    project = response.json()["project"]
    ordered = sorted(project["shots"], key=lambda shot: shot["index"])
    return project["id"], [shot["id"] for shot in ordered]


def test_reorder_shots_roundtrip(settings):
    with TestClient(create_app(settings)) as client:
        project_id, shot_ids = _build_three_shot_project(client)
        reversed_ids = list(reversed(shot_ids))

        response = client.patch(f"/api/projects/{project_id}/shots/order", json={"shot_ids": reversed_ids})
        assert response.status_code == 200, response.text
        updated = response.json()

        assert [shot["id"] for shot in updated["shots"]] == reversed_ids
        assert [shot["index"] for shot in updated["shots"]] == [0, 1, 2]
        # timeline 由 FilmProject validator 按新顺序重建
        assert [clip["shot_id"] for clip in updated["timeline"]] == reversed_ids
        # 与 update_shot 一致：重排重置渲染状态
        assert all(shot["status"] == "planned" for shot in updated["shots"])


def test_reorder_shots_persisted(settings):
    with TestClient(create_app(settings)) as client:
        project_id, shot_ids = _build_three_shot_project(client)
        assert client.patch(f"/api/projects/{project_id}/shots/order", json={"shot_ids": list(reversed(shot_ids))}).status_code == 200

        reloaded = client.get(f"/api/projects/{project_id}").json()
        assert [shot["id"] for shot in reloaded["shots"]] == list(reversed(shot_ids))


def test_reorder_shots_validation(settings):
    with TestClient(create_app(settings)) as client:
        project_id, shot_ids = _build_three_shot_project(client)

        # 缺失镜头 → 422
        r = client.patch(f"/api/projects/{project_id}/shots/order", json={"shot_ids": shot_ids[:-1]})
        assert r.status_code == 422

        # 重复镜头 → 422
        r = client.patch(f"/api/projects/{project_id}/shots/order", json={"shot_ids": [shot_ids[0], shot_ids[0], shot_ids[1]]})
        assert r.status_code == 422

        # 空列表 → 422（min_length=1）
        r = client.patch(f"/api/projects/{project_id}/shots/order", json={"shot_ids": []})
        assert r.status_code == 422

        # 项目不存在 → 404
        r = client.patch("/api/projects/project_missing/shots/order", json={"shot_ids": shot_ids})
        assert r.status_code == 404
