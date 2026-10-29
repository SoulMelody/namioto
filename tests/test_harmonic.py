# SPDX-License-Identifier: AGPL-3.0-only
"""The overtone estimate: which energy belongs to a note above the one that made it."""

from __future__ import annotations

import numpy as np
import pytest

from namioto.analysis import harmonic, spectrum_cache
from namioto.analysis.spectrum import NOTE_COUNT, NoteSpectrum


def table_of(*bands: tuple[int, float], frames: int = 1) -> np.ndarray:
    table = np.zeros((frames, NOTE_COUNT), dtype=np.float32)
    for note, value in bands:
        table[:, note] = value
    return table


def spectrum_for(table: np.ndarray) -> NoteSpectrum:
    return NoteSpectrum(
        table=table,
        frame_ms=25.0,
        sample_rate=44100,
        fft_points=0,
        hop=1102,
        a4=440.0,
        sigma=1.0,
        backend="wavetone",
    )


def test_template_puts_the_fundamental_at_one_and_decays_each_partial() -> None:
    matrix = harmonic.template(harmonics=4, decay=0.5)
    assert matrix[36, 36] == pytest.approx(1.0)
    assert matrix[48, 36] == pytest.approx(0.5)  # the octave, the second partial
    assert matrix[60, 36] == pytest.approx(0.125)  # two octaves up, the fourth partial
    assert matrix[48, 48] == pytest.approx(1.0)  # every column is its own note
    assert not matrix[40, 36]  # a fourth above is nobody's partial


def test_template_splits_a_partial_that_falls_between_bands() -> None:
    matrix = harmonic.template(harmonics=3, decay=1.0)
    # the third partial sits 19.02 semitones up, between two bands
    assert matrix[55, 36] == pytest.approx(1.0 - 0.0196, abs=1e-3)
    assert matrix[56, 36] == pytest.approx(0.0196, abs=1e-3)


def test_reduce_attributes_the_partials_to_the_note_that_made_them() -> None:
    column = harmonic.template(harmonics=8, decay=0.6)[:, 36]
    result = harmonic.reduce(column[None, :].astype(np.float32), harmonics=8, decay=0.6)
    assert result.table.shape == (1, NOTE_COUNT)
    assert result.table[0, 36] == pytest.approx(0.0, abs=1e-4)  # the fundamental is left out
    assert result.table[0, 48] == pytest.approx(0.6, abs=1e-2)  # the octave is kept
    assert result.table[0, 60] == pytest.approx(0.216, abs=1e-2)  # and so is the fourth partial


def test_reduce_reports_progress_by_block() -> None:
    table = np.zeros((3, NOTE_COUNT), dtype=np.float32)
    table[:, 36] = 1.0
    seen = []
    harmonic.reduce(table, harmonics=2, decay=0.6, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(3, 3)]


def test_reduce_rejects_a_table_that_is_not_the_note_domain() -> None:
    with pytest.raises(ValueError):
        harmonic.reduce(np.zeros((4, 12), dtype=np.float32))


def test_subtract_leaves_the_table_alone_at_zero_and_takes_the_partials_at_one() -> None:
    table = table_of((36, 1.0), (48, 0.6))
    estimate = table_of((48, 0.6))

    np.testing.assert_array_equal(harmonic.subtract(table, estimate, 0.0), table)
    cleaned = harmonic.subtract(table, estimate, 1.0)
    assert cleaned[0, 36] == pytest.approx(1.0)
    assert cleaned[0, 48] == pytest.approx(0.0)


def test_subtract_never_goes_below_zero_and_ignores_a_missing_estimate() -> None:
    table = table_of((48, 0.2))
    cleaned = harmonic.subtract(table, table_of((48, 5.0)), 1.0)
    assert cleaned[0, 48] == 0.0
    np.testing.assert_array_equal(harmonic.subtract(table, None, 1.0), table)


@pytest.fixture
def audio(tmp_path, monkeypatch):
    monkeypatch.setattr(spectrum_cache, "data_dir", lambda *parts: tmp_path.joinpath(*parts))
    path = tmp_path / "audio.wav"
    path.write_bytes(b"content")
    return path


def test_analyse_reuses_the_cached_estimate(audio, monkeypatch) -> None:
    spectrum = spectrum_for(table_of((36, 1.0), (48, 0.6), frames=2))
    first = harmonic.analyse(audio, spectrum)
    monkeypatch.setattr(harmonic, "reduce", lambda *args, **kwargs: pytest.fail("the fit ran twice"))
    second = harmonic.analyse(audio, spectrum)
    np.testing.assert_array_equal(first.table, second.table)


def test_analyse_refits_when_the_fit_parameters_change(audio) -> None:
    spectrum = spectrum_for(table_of((36, 1.0), (48, 0.6)))
    harmonic.analyse(audio, spectrum, harmonics=8, decay=0.6)
    timings = {}

    def count(*args, **kwargs):
        timings["ran"] = True
        return harmonic.HarmonicSpectrum(np.zeros(spectrum.table.shape, np.float32), 4, 0.6)

    original = harmonic.reduce
    harmonic.reduce = count
    try:
        harmonic.analyse(audio, spectrum, harmonics=4, decay=0.6)
    finally:
        harmonic.reduce = original
    assert timings.get("ran")


def test_a_corrupt_estimate_is_refitted(audio) -> None:
    spectrum = spectrum_for(table_of((36, 1.0), (48, 0.6)))
    harmonic.analyse(audio, spectrum)
    key = spectrum_cache.cache_key(audio, {"stage": "harmonic", "harmonics": 8, "decay": 0.6})
    assert key is not None
    spectrum_cache.cache_path(key).write_bytes(b"PK\x03\x04broken")
    timings = {}
    harmonic.analyse(audio, spectrum, progress=lambda done, total: timings.setdefault("progress", (done, total)))
    assert timings["progress"] == (1, 1)
