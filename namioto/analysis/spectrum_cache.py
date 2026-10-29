# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free, disposable spectrum results shared by projects using identical audio.

Keys include the audio content, effective analysis parameters and CACHE_VERSION;
bump that version whenever analysis or decoding changes invalidate old results.
Files live under the app's derived-data directory and may be deleted at any time.
Reads never unpickle; corrupt or inaccessible entries are cache misses. Writes
are atomic and best-effort, since persistence must not prevent opening audio.

A result is either a whole `NoteSpectrum` (`load`/`save`) or a bare table that
belongs beside one, such as the overtone estimate (`load_table`/`save_table`).
The two share the directory but never a key: a derived table carries the stage
that produced it in its options, so it invalidates with its own parameters and
not with the spectrum's.
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

from namioto.analysis.choices import BACKENDS
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
            or metadata["backend"] not in BACKENDS
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


def _valid_table(table: np.ndarray) -> bool:
    return (
        table.dtype == np.float32
        and table.ndim == 2
        and table.shape[0] > 0
        and table.shape[1] == NOTE_COUNT
        and bool(np.isfinite(table).all())
        and not bool((table < 0).any())
    )


def load_table(key: str) -> np.ndarray | None:
    """A derived table stored beside a spectrum, or a miss when it is not there or not usable."""
    try:
        with np.load(cache_path(key), allow_pickle=False) as archive:
            table = archive["table"]
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None
    return table if _valid_table(table) else None


def save_table(key: str, table: np.ndarray) -> None:
    """Store one derived table under its own key; best-effort, like the spectrum itself."""
    try:
        buffer = io.BytesIO()
        np.savez(buffer, table=np.ascontiguousarray(table, dtype=np.float32))
        write_bytes(cache_path(key), buffer.getvalue())
    except OSError:
        pass
