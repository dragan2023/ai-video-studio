"""MiniMax H3 temporal limits shared by planning and serving adapters."""

from __future__ import annotations

H3_OUTPUT_FPS = 24
H3_MIN_SHOT_SECONDS = 4.0
H3_MAX_REFERENCE_SECONDS = 15.0

# MiniMax H3 accepts output-duration requests through 15.0 seconds. It rounds
# the request to 24 FPS, then aligns upward to ``17n + 5``; the accepted 15s
# request therefore produces 362 frames / about 15.083 seconds. Ref2VA's input
# contract remains <=15 seconds, so continuation references must be trimmed at
# that adapter boundary instead of shortening every FL2VA generation.
H3_MAX_SHOT_SECONDS = 15.0
H3_MAX_OUTPUT_FRAMES = 362
H3_MAX_ALIGNED_OUTPUT_SECONDS = H3_MAX_OUTPUT_FRAMES / H3_OUTPUT_FPS
# Leave one video frame of container margin when a 362-frame FL2VA output is
# reused as a Ref2VA video reference. The end of the tail remains aligned with
# the previous shot's final moment; only the earliest ~0.13s is discarded.
H3_REF2VA_MAX_INPUT_SECONDS = 15.0
H3_REF2VA_TRIM_SECONDS = 14.95


def h3_aligned_frame_count(requested_frames: int) -> int:
    """Mirror H3's upward ``17n + 5`` frame alignment."""

    if requested_frames <= 0:
        return 1
    current = int(requested_frames)
    while current % 17 != 5:
        current += 1
    return current


def h3_aligned_duration_seconds(requested_seconds: float) -> float:
    """Return the encoded H3 duration implied by a duration request."""

    requested_frames = int(round(float(requested_seconds) * H3_OUTPUT_FPS))
    return h3_aligned_frame_count(requested_frames) / H3_OUTPUT_FPS
