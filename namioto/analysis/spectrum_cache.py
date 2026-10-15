# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free, disposable spectrum results shared by projects using identical audio.

Keys include the audio content, effective analysis parameters and CACHE_VERSION;
bump that version whenever analysis or decoding changes invalidate old results.
Files live under the app's derived-data directory and may be deleted at any time.
Reads never unpickle; corrupt or inaccessible entries are cache misses. Writes
are atomic and best-effort, since persistence must not prevent opening audio.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import numpy as np

from namioto.analysis.spectrum import NOTE_COUNT
from namioto.utils import data_dir, write_bytes

if TYPE_CHECKING:
    from namioto.analysis.spectrum import NoteSpectrum

CACHE_VERSION = 1


def cache_key(path: str | Path, options: dict) -> str | None:
    try:
        with Path(path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None
    payload = json.dumps([CACHE_VERSION, digest, options], sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(key: str) -> Path:
    return data_dir("spectrum-cache", f"{key}.npz")


def load(key: str) -> dict | None:
    try:
        with np.load(cache_path(key), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.pop("key") != key:
                return None
            table = archive["table"]
        if (
            table.dtype != np.float32
            or table.ndim != 2
            or table.shape[0] == 0
            or table.shape[1] != NOTE_COUNT
            or not np.isfinite(table).all()
            or (table < 0).any()
            or set(metadata) != {"frame_ms", "sample_rate", "fft_points", "hop", "a4", "sigma", "backend"}
            or metadata["backend"] not in ("wavetone", "notedigger")
        ):
            return None
        for name in ("frame_ms", "sample_rate", "hop", "a4"):
            if not math.isfinite(metadata[name]) or metadata[name] <= 0:
                return None
        for name in ("sigma", "fft_points"):
            if not math.isfinite(metadata[name]) or metadata[name] < 0:
                return None
        return {"table": table, **metadata}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile):
        return None


def save(key: str, spectrum: NoteSpectrum) -> None:
    metadata = {field.name: getattr(spectrum, field.name) for field in fields(spectrum) if field.name != "table"}
    metadata["key"] = key
    try:
        buffer = io.BytesIO()
        np.savez(buffer, table=spectrum.table, metadata=json.dumps(metadata, allow_nan=False))
        write_bytes(cache_path(key), buffer.getvalue())
    except OSError:
        pass
