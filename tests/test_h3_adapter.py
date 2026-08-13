from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from long_video_studio.adapters.h3 import H3Client
from long_video_studio.domain import (
    DialogueLine,
    ProjectBrief,
    ShotSpec,
    ShotTask,
    StoryboardBeat,
    WorldBible,
)
from long_video_studio.h3_prompt import H3Reference, render_h3_prompt


def test_h3_adapter_rejects_legacy_fifteen_second_shot(tmp_path: Path):
    image = tmp_path / "start.png"
    image.write_bytes(b"image")
    shot = ShotSpec(
        index=0,
        title="Legacy",
        purpose="Exercise the transport guard",
        duration_seconds=15,
        prompt="A continuous shot.",
    )
    client = H3Client("http://h3.example:8091")

    with pytest.raises(ValueError, match="safety ceiling is 14 seconds"):
        asyncio.run(client.generate_fl2va(shot, image, tmp_path / "shot.mp4"))


def test_h3_adapter_exposes_actionable_endpoint_error(tmp_path: Path):
    image = tmp_path / "start.png"
    image.write_bytes(b"image")
    shot = ShotSpec(
        index=0,
        title="Opening",
        purpose="Exercise endpoint diagnostics",
        duration_seconds=4,
        prompt="A continuous shot.",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    client = H3Client("http://ref2va.example:8092", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="H3 fl2va endpoint is unavailable") as error:
        asyncio.run(client.generate_fl2va(shot, image, tmp_path / "shot.mp4"))
    assert "start the selected service" in str(error.value)


def test_fl2va_adapter_uses_current_video_api(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "start.png"
    image.write_bytes(b"not-a-real-image-but-valid-for-multipart")
    output = tmp_path / "shot.mp4"
    shot = ShotSpec(
        index=0,
        title="Opening",
        purpose="Open the film",
        duration_seconds=10,
        prompt="A cinematic opening.",
        inference_steps=12,
    )
    client = H3Client("http://h3.example:8091", transport=httpx.MockTransport(handler))
    result = asyncio.run(
        client.generate_fl2va(
            shot,
            image,
            output,
            width=1280,
            height=704,
        )
    )
    assert result.read_bytes() == b"\x00\x00mp4"
    assert b'name="input_reference"' in request_body
    assert b'name="image"' not in request_body
    assert b'name="extra_params"' in request_body
    assert b'"task": "fl2va"' in request_body
    assert b'name="width"' in request_body
    assert b'name="height"' in request_body
    assert b"\r\n\r\n1280\r\n" in request_body
    assert b"\r\n\r\n704\r\n" in request_body
    assert b"CONTINUITY FROM THE PREVIOUS SHOT" not in request_body
    assert b"red umbrella" not in request_body
    assert b"burned-in subtitles" in request_body
    assert client.endpoint == "http://h3.example:8091/v1/videos/sync"


def test_ref2va_video_adapter_uses_plural_reference_field(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "identity.png"
    image.write_bytes(b"image")
    video = tmp_path / "motion.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "shot.mp4"
    shot = ShotSpec(
        index=0,
        title="Reference",
        purpose="Follow a reference",
        duration_seconds=4,
        prompt="Follow the reference motion.",
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_ref2va(shot, image, video, output, width=704, height=1280))

    assert b'name="input_references"' in request_body
    assert request_body.count(b'name="input_references"') == 2
    assert request_body.index(b'filename="identity.png"') < request_body.index(b'filename="motion.mp4"')
    assert b'name="image"' not in request_body
    assert b'name="video"' not in request_body
    assert b'"task": "ref2va"' in request_body
    assert b'name="width"' in request_body
    assert b'name="height"' in request_body
    assert b"\r\n\r\n704\r\n" in request_body
    assert b"\r\n\r\n1280\r\n" in request_body


def test_ref2va_audio_adapter_uses_image_and_data_url(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "identity.png"
    image.write_bytes(b"image")
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"audio")
    output = tmp_path / "shot.mp4"
    shot = ShotSpec(
        index=0,
        title="Reference",
        purpose="Follow audio",
        duration_seconds=4,
        prompt="Lip sync to the voice.",
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_ref2va(shot, image, audio, output))

    assert b'name="input_reference"' in request_body
    assert b'name="audio_reference"' in request_body
    assert b"data:audio/mpeg;base64,YXVkaW8=" in request_body


def test_h3_prompt_keeps_visual_direction_out_of_spoken_dialogue(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "start.png"
    output = tmp_path / "out.mp4"
    image.write_bytes(b"image")
    shot = ShotSpec(
        index=0,
        title="Doorway",
        purpose="Build tension",
        prompt="白鹿皱眉，缓慢走向门口。",
        audio_prompt="脚步声和安静的室内环境声。",
        dialogue=[
            DialogueLine(
                speaker="白鹿",
                text="你终于来了。",
                language="Chinese",
                delivery="克制而警惕",
            )
        ],
        duration_seconds=4,
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_fl2va(shot, image, output))

    assert "白鹿皱眉，缓慢走向门口。".encode() in request_body
    assert "<d>[Chinese] 你终于来了。</d>".encode() in request_body
    assert "<d>[Chinese] 白鹿皱眉".encode() not in request_body
    assert b"integrated_multimodal_description:" in request_body
    assert b"overall_soundscape:" in request_body
    assert b"non_diegetic_music:" in request_body


def test_h3_prompt_explicitly_disables_speech_without_dialogue(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "start.png"
    output = tmp_path / "out.mp4"
    image.write_bytes(b"image")
    shot = ShotSpec(
        index=0,
        title="Silent walk",
        purpose="Show movement",
        prompt="The actor frowns and walks toward the door.",
        duration_seconds=4,
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_fl2va(shot, image, output))

    assert b"No spoken dialogue, no narration, and no voice-over" in request_body
    assert b"<d>" not in request_body


def test_h3_official_section_order_for_fl2va_and_ref2va():
    fl2va = render_h3_prompt(
        ShotSpec(
            index=0,
            title="Opening",
            purpose="Open",
            prompt="The actor turns toward camera.",
            audio_prompt="Room tone and cloth movement.",
            music_prompt="A restrained string pulse.",
            duration_seconds=4,
        )
    )
    ref2va = render_h3_prompt(
        ShotSpec(
            index=1,
            title="Continuation",
            purpose="Continue",
            task=ShotTask.REF2VA,
            prompt="The actor walks through the doorway.",
            dialogue=[DialogueLine(speaker="Actor", text="Wait.", language="English")],
            duration_seconds=4,
        ),
        (
            H3Reference("picture", "Picture 1", "Identity reference image"),
            H3Reference("video", "Video 1", "Temporal reference video"),
        ),
    )

    assert fl2va.index("integrated_multimodal_description:") < fl2va.index("overall_soundscape:")
    assert fl2va.index("overall_soundscape:") < fl2va.index("non_diegetic_music:")
    assert fl2va.startswith(
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert fl2va.rstrip().endswith("A restrained string pulse.")
    headings = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    positions = [ref2va.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "(S1) Actor" in ref2va
    assert "<Picture 1>" in ref2va and "Identity reference image" in ref2va
    assert "<Video 1>" in ref2va and "Temporal reference video" in ref2va
    assert "<d>[English] Wait.</d>" in ref2va
    assert "<d>[English] The actor walks" not in ref2va


def test_fl2va_context_keeps_project_speaker_ids_and_dialogue_in_beat():
    shot = ShotSpec(
        index=1,
        title="Reply",
        purpose="Answer the challenge",
        duration_seconds=4,
        prompt="Meng Ziyi turns toward Bai Lu without changing position.",
        opening_state="Meng Ziyi faces the window while Bai Lu waits by the table.",
        ending_state="Meng Ziyi closes her lips and holds Bai Lu's gaze.",
        visual_beats=[
            StoryboardBeat(
                start_seconds=0,
                end_seconds=2,
                visual_action="Meng Ziyi turns toward Bai Lu",
                state_change="Their eyelines meet",
                camera="the camera pushes in with small amplitude at slow speed",
                sound="a soft fabric movement follows the turn",
            ),
            StoryboardBeat(
                start_seconds=2,
                end_seconds=4,
                visual_action="Meng Ziyi speaks and then holds still",
                state_change="Her lips close after the reply",
                camera="the camera brakes into a static medium close-up",
                sound="quiet room tone continues beneath the voice",
            ),
        ],
        dialogue=[
            DialogueLine(
                speaker="Meng Ziyi",
                text="我明白。",
                language="zh-CN",
                delivery="restrained and certain",
                start_seconds=2.2,
                end_seconds=3.2,
            )
        ],
    )
    prompt = render_h3_prompt(
        shot,
        brief=ProjectBrief(prompt="A quiet palace reply"),
        speaker_ids={"Bai Lu": "S1", "Meng Ziyi": "S2"},
    )
    parts = prompt.split("\n\n")
    assert len(parts) == 4
    assert parts[0].endswith("<Picture 1> (from [Shot 1]) is fully referenced.")
    assert parts[1].startswith("integrated_multimodal_description: [Shot 1] ")
    assert parts[2].startswith("overall_soundscape: ")
    assert parts[3] == "non_diegetic_music: N/A"
    assert "(S2) Meng Ziyi" in parts[1]
    assert "<d>[zh-CN] 我明白。</d>" in parts[1]
    assert parts[1].index("From 2.000s to 4.000s") < parts[1].index("<d>[zh-CN] 我明白。</d>")
    assert "<d>" not in parts[2]
    assert "Picture 2" not in prompt


def test_ref2va_call_uses_ref_prompt_when_runtime_promotes_fl2va_shot(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "reference.png"
    video = tmp_path / "reference.mp4"
    output = tmp_path / "out.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    shot = ShotSpec(
        index=1,
        title="Runtime continuation",
        purpose="Continue",
        task=ShotTask.FL2VA,
        prompt="The actor walks onward.",
        duration_seconds=4,
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_ref2va(shot, image, video, output))

    assert b"subject_definitions:" in request_body
    assert b"retention_analysis:" in request_body
    assert b"<Picture 1>" in request_body
    assert b"<Video 1>" in request_body


def test_ref2va_adapter_compiles_planner_h3_context_into_request(tmp_path: Path):
    request_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = request.read()
        return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"\x00\x00mp4")

    image = tmp_path / "reference.png"
    video = tmp_path / "reference.mp4"
    output = tmp_path / "out.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    previous = ShotSpec(
        index=0,
        title="Opening",
        purpose="Establish the hall",
        prompt="The actor enters the hall.",
        duration_seconds=4,
        ending_state="The actor stops beside the table.",
    )
    shot = ShotSpec(
        index=1,
        title="Continuation",
        purpose="Reveal the letter",
        task=ShotTask.FL2VA,
        prompt="The actor lifts the sealed letter into the morning light.",
        duration_seconds=4,
        opening_state="The actor stands beside the table.",
        ending_state="The sealed letter is visible at chest height.",
        continuity_handoff="Keep identity, wardrobe, table geography, motion direction, and room tone stable.",
        reference_anchors=["Character identity: the same actor", "Prop identity: sealed letter"],
        hook="The wax seal catches the light.",
        visual_beats=[
            StoryboardBeat(
                start_seconds=0,
                end_seconds=2,
                visual_action="The actor reaches for the sealed letter",
                state_change="The letter leaves the table",
                camera="Small-amplitude slow Push In",
                sound="Paper slides softly across wood",
            ),
            StoryboardBeat(
                start_seconds=2,
                end_seconds=4,
                visual_action="The actor raises the sealed letter into the light",
                state_change="The wax seal becomes clearly visible",
                camera="The Push In brakes into a Static Shot",
                sound="Fabric settles over continuous room tone",
            ),
        ],
    )
    client = H3Client("http://h3.example:8092", transport=httpx.MockTransport(handler))

    asyncio.run(
        client.generate_ref2va(
            shot,
            image,
            video,
            output,
            brief=ProjectBrief(prompt="A palace mystery", style_preset="cinematic"),
            world_bible=WorldBible(
                logline="An actor reveals a sealed letter.",
                visual_style="restrained palace realism",
                character_notes=["the same actor in a dark blue robe"],
                location_notes=["a stone palace hall"],
            ),
            previous_shot=previous,
            speaker_ids={},
        )
    )

    assert b"[video continuation + reference generation]" in request_body
    assert b"From 0.000s to 2.000s" in request_body
    assert b"Paper slides softly across wood" in request_body
    assert b"final moment preserves this continuity" in request_body
    assert b"Keep identity, wardrobe" in request_body
    assert b"The primary hook is" in request_body
    assert b"Previous shot ending state:" in request_body


def test_async_video_job_is_polled_and_downloaded(tmp_path: Path):
    seen: list[tuple[str, str]] = []
    expected_prompt = ""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "submission-id", "status": "queued"})
        if request.url.path == "/v1/videos/submission-id":
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/v1/videos":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "video-job-1",
                            "prompt": expected_prompt,
                            "created_at": 1,
                        }
                    ]
                },
            )
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"\x00\x00async-video", headers={"content-type": "video/mp4"})
        return httpx.Response(200, json={"id": "video-job-1", "status": "completed"})

    image = tmp_path / "reference.png"
    video = tmp_path / "reference.mp4"
    output = tmp_path / "async.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    shot = ShotSpec(
        index=0,
        title="Continue",
        purpose="Advance the scene",
        prompt="Continue after the reference.",
        duration_seconds=4,
        task=ShotTask.REF2VA,
    )
    expected_prompt = render_h3_prompt(
        shot,
        (
            H3Reference(
                "picture",
                "Picture 1",
                "Use reference.png as the still-image identity and appearance reference for the subjects requested "
                "by the storyboard.",
                role="identity",
                relationship="fully_preserved",
            ),
            H3Reference(
                "video",
                "Video 1",
                "Use reference.mp4 as the video reference and preserve its relevant continuity, motion, scene, "
                "style, and synchronized sound characteristics.",
                role="continuation",
                relationship="fully_preserved",
            ),
        ),
    )
    client = H3Client("http://h3.test:8092", transport=httpx.MockTransport(handler))

    asyncio.run(client.generate_ref2va(shot, image, video, output, async_job=True))

    assert output.read_bytes() == b"\x00\x00async-video"
    assert seen == [
        ("POST", "/v1/videos"),
        ("GET", "/v1/videos/submission-id"),
        ("GET", "/v1/videos"),
        ("GET", "/v1/videos/video-job-1"),
        ("GET", "/v1/videos/video-job-1/content"),
    ]


def test_async_failed_job_preserves_server_error_payload(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job-id", "status": "queued"})
        return httpx.Response(
            500,
            json={
                "id": "job-id",
                "status": "failed",
                "error": {"code": 500, "message": "MUSA out of memory"},
            },
        )

    image = tmp_path / "reference.png"
    video = tmp_path / "reference.mp4"
    output = tmp_path / "failed.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    shot = ShotSpec(
        index=0,
        title="Continue",
        purpose="Advance the scene",
        duration_seconds=4,
        prompt="Continue after the reference.",
        task=ShotTask.REF2VA,
    )
    client = H3Client("http://h3.test:8092", transport=httpx.MockTransport(handler))

    try:
        asyncio.run(client.generate_ref2va(shot, image, video, output, async_job=True))
    except RuntimeError as error:
        assert "MUSA out of memory" in str(error)
    else:  # pragma: no cover - the assertion is the test's failure branch
        raise AssertionError("failed async job did not raise")
