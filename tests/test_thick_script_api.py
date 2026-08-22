from __future__ import annotations

from fastapi.testclient import TestClient

from long_video_studio.app import create_app


def test_import_thick_script_endpoint(settings):
    script = """### 1-01 · 黑屏（4s｜文戏）

生成一段4秒、16:9、原生立体声的黑屏字幕短片。

0—4秒：黑色背景中央浮现金色字幕。

剪辑与动作：保持黑场稳定。

视觉风格：纯黑与金色文字。

声音设计：低频心跳，无对白。
"""
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/projects/import-thick-script",
            data={"title": "测试厚版", "asset_root": ""},
            files={"script": ("test.md", script.encode("utf-8"), "text/markdown")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["project"]["brief"]["title"] == "测试厚版"
    assert len(payload["project"]["shots"]) == 1
    assert payload["project"]["shots"][0]["subtitle_text"] == "__black_frame__"
    assert payload["missing_codes"] == []
