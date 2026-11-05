# SPDX-License-Identifier: AGPL-3.0-only
"""The audio properties panel: read-only, honest about what decoding threw away, safe with no audio."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLineEdit

from namioto.analysis.spectrum import NoteSpectrum
from namioto.project import default_settings
from namioto.ui.properties_dialog import AudioProperties, PropertiesDialog, clock, pitch_range


class FakeSong:
    def __init__(self, seconds: float):
        self.is_loaded = True
        self.duration = seconds


class FakeView:
    def __init__(self, spectrum=None):
        self.spectrum = spectrum


class FakeWindow(QObject):
    """The handful of attributes the panel reads; a real window is not needed to check a panel."""

    def __init__(self, spectrum=None, song=None, settings=None):
        super().__init__()
        self.view = FakeView(spectrum)
        self.song = song
        self.project_settings = settings or default_settings()


def spectrum_of(seconds: float, sample_rate: int = 22050, frame_ms: float = 25.0) -> NoteSpectrum:
    frames = int(round(seconds * 1000.0 / frame_ms))
    return NoteSpectrum(
        table=np.zeros((frames, 84), dtype=np.float32),
        frame_ms=frame_ms,
        sample_rate=sample_rate,
        fft_points=8192,
        hop=512,
        a4=440.0,
        sigma=1.0,
    )


def test_a_duration_reads_as_hours_minutes_seconds() -> None:
    assert clock(0.0) == "00:00:00"
    assert clock(8.0) == "00:00:08"
    assert clock(3661.0) == "01:01:01"


def test_the_pitch_range_spans_the_table() -> None:
    assert pitch_range(84) == "C1–B7"
    assert pitch_range(0) == "-"


def test_no_audio_shows_a_zero_clock_and_no_editable_field(qt_app) -> None:
    dialog = PropertiesDialog(AudioProperties.from_window(FakeWindow()))
    assert dialog.values["Duration"].text() == "00:00:00"
    assert dialog.values["Sample rate"].text() == "-"
    assert dialog.findChildren(QLineEdit) == []
    assert dialog.properties.pitch_range == "-"


def test_eight_seconds_of_audio_reads_back_as_eight_seconds(qt_app) -> None:
    window = FakeWindow(spectrum=spectrum_of(8.0), settings=default_settings())
    dialog = PropertiesDialog(AudioProperties.from_window(window))
    assert dialog.values["Duration"].text() == "00:00:08"
    assert dialog.values["Sample rate"].text() == "22050 Hz"
    assert dialog.values["Pitch range"].text() == "C1–B7"


def test_the_values_match_the_project_settings(qt_app) -> None:
    settings = default_settings()
    settings.analysis.a4 = 442.0
    settings.analysis.t_num = 40.0
    window = FakeWindow(spectrum=spectrum_of(8.0, frame_ms=25.0), settings=settings)
    dialog = PropertiesDialog(AudioProperties.from_window(window))
    assert dialog.values["Reference frequency"].text() == "442.0 Hz"
    # 25 ms a frame is 40 frames a second, which is the settings' own t_num
    assert dialog.values["Frames per second"].text() == "40.00"
    assert float(dialog.values["Frames per second"].text()) == settings.analysis.t_num


def test_channels_are_not_invented_when_decoding_mixed_them_away(qt_app) -> None:
    """`load_song` decodes with mono=True, so the source channel count is gone before the dialog runs."""
    window = FakeWindow(spectrum=spectrum_of(8.0))
    dialog = PropertiesDialog(AudioProperties.from_window(window))
    assert dialog.values["Channels"].text() != "2"
    assert dialog.findChildren(QLineEdit) == []


def test_the_player_stands_in_for_the_duration_before_the_spectrum_exists(qt_app) -> None:
    window = FakeWindow(spectrum=None, song=FakeSong(12.0))
    dialog = PropertiesDialog(AudioProperties.from_window(window))
    assert dialog.values["Duration"].text() == "00:00:12"
    # the buffer is known, the analysis is not, so the analysis-only fields stay empty
    assert dialog.values["Frames per second"].text() == "-"


def test_wire_builds_an_action(qt_app) -> None:
    from namioto.ui.properties_dialog import wire

    action = wire(FakeWindow())
    assert action.text()
