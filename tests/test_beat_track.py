# SPDX-License-Identifier: AGPL-3.0-only
"""Beat tracking read off a note table: onset envelope, tempo curve and downbeats.

Unlike `test_beats.py`, nothing here decodes audio: the tables are built by hand, so the whole file
runs with the default test selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from namioto.analysis import beats
from namioto.analysis.spectrum import NOTE_COUNT, NoteSpectrum

FRAME_RATE = 40.0
FRAME_MS = 1000.0 / FRAME_RATE


def click_table(bpm: float, seconds: float = 20.0, *, beats_per_bar: int = 4) -> np.ndarray:
    """A note table that attacks on every beat, with the first beat of each bar much louder."""
    frames = int(seconds * FRAME_RATE)
    period = 60.0 / bpm * FRAME_RATE
    table = np.full((frames, NOTE_COUNT), 0.02, dtype=np.float32)
    for index in range(frames):
        if index % round(period) >= 2:
            continue
        beat = round(index / period)
        table[index, 36] = 1.0
        table[index, 43] = 0.4
        if beat % beats_per_bar == 0:
            # a bar's first beat carries two more bands, so its attack is unmistakable
            table[index, 50] = 1.0
            table[index, 57] = 1.0
    return table


def spectrum_for(table: np.ndarray) -> NoteSpectrum:
    return NoteSpectrum(
        table=table,
        frame_ms=FRAME_MS,
        sample_rate=22050,
        fft_points=0,
        hop=1,
        a4=440.0,
        sigma=1.0,
        backend="notedigger",
    )


def test_onset_envelope_peaks_where_a_note_attacks() -> None:
    table = click_table(120.0, 4.0)
    onset = beats.onset_envelope(table)
    assert onset.shape == (len(table),)
    assert np.isfinite(onset).all()
    assert onset.min() == pytest.approx(0.0)
    # the first frames of every half second are the attacks, and they stand above the rest
    attacks = onset[::20]
    rests = np.delete(onset, np.s_[::20])
    assert attacks.mean() > rests.mean()


def test_onset_envelope_rejects_a_table_that_is_not_two_dimensional() -> None:
    with pytest.raises(ValueError):
        beats.onset_envelope(np.zeros(4, dtype=np.float32))


def test_tempo_curve_finds_the_beat_of_a_click_table() -> None:
    table = click_table(120.0, 20.0)
    onset = beats.onset_envelope(table)
    curve = beats.tempo_curve(onset, FRAME_RATE)
    assert len(curve) == len(onset)
    assert float(np.median(curve)) == pytest.approx(120.0, abs=1.0)


def test_tempo_curve_refuses_a_frame_rate_that_is_not_positive() -> None:
    with pytest.raises(ValueError):
        beats.tempo_curve(np.zeros(10), 0.0)


def test_beat_strength_takes_the_peak_near_each_beat() -> None:
    onset = np.zeros(20)
    onset[5] = 1.0
    onset[15] = 0.5
    strength = beats.beat_strength(onset, np.array([5, 14]))
    assert strength.tolist() == pytest.approx([1.0, 0.5])


def test_downbeats_prefers_four_beats_a_bar_when_the_first_is_accented() -> None:
    strength = np.array([1.0, 0.3, 0.3, 0.3] * 8)
    meter, indices = beats.downbeats(strength)
    assert meter == 4
    assert indices[:4] == (0, 4, 8, 12)


def test_downbeats_returns_nothing_for_no_beats() -> None:
    assert beats.downbeats(np.zeros(0)) == (0, ())


def test_track_reads_the_tempo_the_beats_and_the_meter_from_a_table() -> None:
    result = beats.track(spectrum_for(click_table(120.0, 20.0)))
    assert result.bpm == pytest.approx(120.0, abs=1.0)
    assert len(result.beats) >= beats.MIN_BEATS
    assert result.meter == 4
    assert len(result.downbeats) == pytest.approx(len(result.beats) / 4, abs=1)
    assert len(result.per_frame_bpm) == int(20.0 * FRAME_RATE)
    # a beat every half second, and a bar every two
    assert np.diff(result.beats)[:5] == pytest.approx([0.5] * 5, abs=0.05)
    assert np.diff(result.downbeats)[:3] == pytest.approx([2.0] * 3, abs=0.1)


def test_track_refuses_a_table_too_short_to_track() -> None:
    table = np.zeros((3, NOTE_COUNT), dtype=np.float32)
    with pytest.raises(ValueError):
        beats.track(spectrum_for(table))
