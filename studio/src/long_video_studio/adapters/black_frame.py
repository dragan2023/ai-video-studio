"""Dependency-free black PNG anchor for no-reference shots."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

BLACK_FRAME_MARKER = "__black_frame__"


def write_black_png(path: str | Path, width: int, height: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
        payload = b"\x89PNG\r\n\x1a\n"
        payload += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        payload += _chunk(b"IDAT", zlib.compress(raw, level=9))
        payload += _chunk(b"IEND", b"")
        path.write_bytes(payload)
    return path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
