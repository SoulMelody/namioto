# SPDX-License-Identifier: AGPL-3.0-only
"""Persistent spectra are optional, content-addressed and safe to discard."""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from namioto.analysis import spectrum, spectrum_cache


@pytest.fixture
def audio(tmp_path, monkeypatch):
    monkeypatch.setattr(spectrum_cache, "data_dir", lambda *parts: tmp_path.joinpath(*parts))
    path = tmp_path / "audio.wav"
    sf.write(path, np.sin(2 * np.pi * 440 * np.arange(4000) / 8000), 8000, subtype="FLOAT")
    return path


def fail_decode(*args, **kwargs):
    pytest.fail("A cached spectrum must not decode audio")


def cache_file(audio):
    return next((audio.parent / "spectrum-cache").glob("*.npz"))


def test_hit_skips_decoding_preserves_metadata_and_reports_completion(audio, monkeypatch):
    original = spectrum.analyse(audio)
    monkeypatch.setattr(spectrum, "load_channels", fail_decode)
    updates = []
    timings = {}
    cached = spectrum.analyse(audio, progress=lambda *args: updates.append(args), timings=timings)
    np.testing.assert_array_equal(cached.table, original.table)
    assert cached.summary() == original.summary()
    assert updates == [(1, 1)]
    assert set(timings) == {"cache_read"}


def test_identical_content_is_shared_across_paths_and_ignored_fft_setting(audio, monkeypatch):
    original = spectrum.analyse(audio)
    copy = audio.with_name("copy.wav")
    copy.write_bytes(audio.read_bytes())
    monkeypatch.setattr(spectrum, "load_channels", fail_decode)
    cached = spectrum.analyse(copy, fft_points=2048)
    np.testing.assert_array_equal(cached.table, original.table)


def test_changed_content_with_same_size_and_mtime_invalidates_cache(audio):
    original = spectrum.analyse(audio)
    stamp = audio.stat()
    sf.write(audio, np.sin(2 * np.pi * 880 * np.arange(4000) / 8000), 8000, subtype="FLOAT")
    assert audio.stat().st_size == stamp.st_size
    os.utime(audio, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    changed = spectrum.analyse(audio)
    assert not np.array_equal(changed.table, original.table)
    assert len(list(cache_file(audio).parent.glob("*.npz"))) == 2


@pytest.mark.parametrize("options", [{"channels": "left"}, {"t_num": 20}, {"a4": 432}, {"backend": "notedigger"}])
def test_effective_parameters_invalidate_cache(audio, options):
    spectrum.analyse(audio)
    timings = {}
    spectrum.analyse(audio, timings=timings, **options)
    assert "decode" in timings
    assert len(list(cache_file(audio).parent.glob("*.npz"))) == 2


def test_stft_fft_size_and_algorithm_version_invalidate_cache(audio, monkeypatch):
    spectrum.analyse(audio, backend="notedigger", fft_points=2048)
    spectrum.analyse(audio, backend="notedigger", fft_points=4096)
    monkeypatch.setattr(spectrum_cache, "CACHE_VERSION", spectrum_cache.CACHE_VERSION + 1)
    spectrum.analyse(audio, backend="notedigger", fft_points=4096)
    assert len(list(cache_file(audio).parent.glob("*.npz"))) == 3


@pytest.mark.parametrize("corruption", ["truncated", "shape", "nan", "metadata", "key", "pickle"])
def test_corrupt_cache_is_recomputed(audio, corruption):
    original = spectrum.analyse(audio)
    target = cache_file(audio)
    with np.load(target, allow_pickle=False) as archive:
        table = archive["table"]
        metadata = json.loads(str(archive["metadata"].item()))
    if corruption == "truncated":
        target.write_bytes(b"PK\x03\x04broken")
    else:
        if corruption == "shape":
            table = table[:, :3]
        elif corruption == "nan":
            table[0, 0] = np.nan
        elif corruption == "metadata":
            metadata["hop"] = 0
        elif corruption == "key":
            metadata["key"] = "wrong"
        elif corruption == "pickle":
            table = np.array([object()], dtype=object)
        np.savez(target, table=table, metadata=json.dumps(metadata))
    timings = {}
    restored = spectrum.analyse(audio, timings=timings)
    assert "decode" in timings
    np.testing.assert_array_equal(restored.table, original.table)


def test_disabled_cache_never_reads_or_writes(audio, monkeypatch):
    monkeypatch.setattr(spectrum_cache, "cache_key", lambda *args: pytest.fail("Cache was disabled"))
    spectrum.analyse(audio, cache=False)
    assert not (audio.parent / "spectrum-cache").exists()


def test_unwritable_cache_does_not_prevent_analysis(audio, monkeypatch):
    def refuse(*args):
        raise PermissionError("read-only")

    monkeypatch.setattr(spectrum_cache, "write_bytes", refuse)
    result = spectrum.analyse(audio)
    assert result.frames > 0
    assert not (audio.parent / "spectrum-cache").exists()


def test_audio_changed_during_analysis_is_not_cached(audio, monkeypatch):
    decode = spectrum.load_channels

    def replace_after_decode(path, mode):
        selected = decode(path, mode)
        Path(path).write_bytes(b"replaced")
        return selected

    monkeypatch.setattr(spectrum, "load_channels", replace_after_decode)
    spectrum.analyse(audio)
    assert not (audio.parent / "spectrum-cache").exists()


def test_atomic_write_leaves_no_partial_cache(audio, monkeypatch):
    from namioto import utils

    def refuse(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(utils.os, "replace", refuse)
    spectrum.analyse(audio)
    assert not list((audio.parent / "spectrum-cache").iterdir())


def test_cache_does_not_load_pickled_metadata(audio):
    spectrum.analyse(audio)
    buffer = io.BytesIO()
    np.savez(buffer, table=np.zeros((1, 84), dtype=np.float32), metadata=np.array({"hop": 1}, dtype=object))
    cache_file(audio).write_bytes(buffer.getvalue())
    timings = {}
    spectrum.analyse(audio, timings=timings)
    assert "decode" in timings


@pytest.fixture
def wide_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(spectrum_cache, "data_dir", lambda *parts: tmp_path.joinpath(*parts))
    path = tmp_path / "wide.wav"
    sf.write(path, np.sin(2 * np.pi * 220 * np.arange(22050) / 44100), 44100, subtype="FLOAT")
    return path


def test_the_cqt_backend_is_cached_apart_from_the_stft(wide_audio):
    # the constant-Q path needs a rate whose Nyquist clears its highest band, hence its own fixture
    spectrum.analyse(wide_audio, backend="notedigger")
    spectrum.analyse(wide_audio, backend="cqt")
    assert len(list(cache_file(wide_audio).parent.glob("*.npz"))) == 2
