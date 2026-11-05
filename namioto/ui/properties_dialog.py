# SPDX-License-Identifier: AGPL-3.0-only
"""The audio's properties, read-only: what the editor holds, not what the file might have been.

WaveTone shows this panel beside its spectrum, and every field on it describes the *analysis* the
editor is holding rather than the container on disk. That matters, because two of the fields cannot
be read back honestly:

* **Channels.** `namioto.ui.song.load_song` decodes with `librosa.load(..., mono=True)` because the
  engine plays one buffer, so a stereo file's second channel is gone by the time anything could ask.
  Reporting "2" would be a guess dressed as a fact, so the field says `mono (mixed)`: true, and the
  reason is here in the code.
* **Bit depth.** Decoding also lands in `float32`, so the source's 16- or 24-bit words are gone too.
  The field names the sample format the editor actually holds.

Everything is a `QLabel` on purpose. A read-only panel built from spin boxes would still own a
`QLineEdit` internally, and a test asserts there is none: the panel cannot be typed into by accident.

`wire(window)` reads these attributes from the window it is given, each of them optional:

* `window.view.spectrum` - a `namioto.analysis.spectrum.NoteSpectrum`, or None before any audio
* `window.project_settings.analysis` - `a4`, the reference frequency the analysis ran with
* `window.song` - a `SongPlayer`; its `duration` stands in when the spectrum is not built yet
* `window.meter_map` (or `window.view.meter`) - a `MeterMap`; its first point's `bar_offset` is the
  start bar. Neither exists today, so the field falls back to 1 and starts being right for free
  once the third batch wires a meter in.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from namioto.analysis.spectrum import NOTE_COUNT, note_label
from namioto.i18n import tr

EMPTY = "-"
NO_AUDIO = "00:00:00"
# the table the editor draws has one column per semitone, so one "block" is one semitone; the CQT
# behind it may resolve three bins per semitone, but the table it produces does not
BLOCKS_PER_SEMITONE = 1
# decoding lands here whatever the file said, so this is the honest answer for a decoded song
DECODED_FORMAT = "32-bit float"


def clock(seconds: float | None) -> str:
    """`hh:mm:ss`, the shape a duration reads in; `00:00:00` when there is no audio."""
    if seconds is None or seconds <= 0.0:
        return NO_AUDIO
    total = int(round(float(seconds)))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def pitch_range(count: int) -> str:
    """The note range a table of `count` semitones covers, low to high."""
    if count <= 0:
        return EMPTY
    return f"{note_label(0)}–{note_label(count - 1)}"


@dataclass(frozen=True)
class AudioProperties:
    """What the panel shows, already resolved: no widget reaches into a window while it paints."""

    sample_rate: str = EMPTY
    bit_depth: str = EMPTY
    channels: str = EMPTY
    duration: str = NO_AUDIO
    pitch_range: str = EMPTY
    a4: str = EMPTY
    frames_per_second: str = EMPTY
    blocks_per_semitone: str = EMPTY
    start_bar: str = "1"

    @classmethod
    def from_window(cls, window) -> AudioProperties:
        """Collect the values from a window, tolerating every one of them being absent."""
        spectrum = getattr(getattr(window, "view", None), "spectrum", None)
        analysis = getattr(getattr(window, "project_settings", None), "analysis", None)
        duration = _duration(window, spectrum)
        has_audio = duration != NO_AUDIO
        if spectrum is None:
            # decoded but not yet analysed: the length is known, the sample rate the analysis read
            # it at is not, so only the fields that describe the buffer itself are filled in
            return cls(
                bit_depth=DECODED_FORMAT if has_audio else EMPTY,
                channels=tr("mono (mixed)") if has_audio else EMPTY,
                duration=duration,
                a4=_a4(analysis),
                start_bar=_start_bar(window),
            )
        frame_ms = float(getattr(spectrum, "frame_ms", 0.0) or 0.0)
        return cls(
            sample_rate=f"{int(getattr(spectrum, 'sample_rate', 0))} Hz",
            bit_depth=DECODED_FORMAT,
            channels=tr("mono (mixed)"),
            duration=duration,
            pitch_range=pitch_range(_columns(spectrum)),
            a4=_a4(analysis),
            frames_per_second=f"{1000.0 / frame_ms:.2f}" if frame_ms > 0.0 else EMPTY,
            blocks_per_semitone=str(BLOCKS_PER_SEMITONE),
            start_bar=_start_bar(window),
        )

    def rows(self) -> tuple[tuple[str, str], ...]:
        """The panel's rows, caption then value, in the order WaveTone lists them."""
        return (
            (tr("Sample rate"), self.sample_rate),
            (tr("Bit depth"), self.bit_depth),
            (tr("Channels"), self.channels),
            (tr("Duration"), self.duration),
            (tr("Pitch range"), self.pitch_range),
            (tr("Reference frequency"), self.a4),
            (tr("Frames per second"), self.frames_per_second),
            (tr("Blocks per semitone"), self.blocks_per_semitone),
            (tr("Start bar"), self.start_bar),
        )


def _columns(spectrum) -> int:
    """How many note columns the table has, or the fixed count when it cannot say."""
    shape = getattr(getattr(spectrum, "table", None), "shape", ())
    return int(shape[1]) if len(shape) == 2 else NOTE_COUNT


def _duration(window, spectrum) -> str:
    """The song's length: the spectrum's own, else the player's, else nothing."""
    if spectrum is not None:
        seconds = float(getattr(spectrum, "duration", 0.0) or 0.0)
        if seconds > 0.0:
            return clock(seconds)
    song = getattr(window, "song", None)
    if song is not None and getattr(song, "is_loaded", False):
        return clock(float(getattr(song, "duration", 0.0) or 0.0))
    return NO_AUDIO


def _a4(analysis) -> str:
    value = getattr(analysis, "a4", None) if analysis is not None else None
    return EMPTY if value is None else f"{float(value):.1f} Hz"


def _start_bar(window) -> str:
    """The number the first bar is called; a window with no meter yet starts at bar 1."""
    meter = getattr(window, "meter_map", None) or getattr(getattr(window, "view", None), "meter", None)
    points = getattr(meter, "points", None)
    if not points:
        return "1"
    return str(int(getattr(points[0], "bar_offset", 1)))


class PropertiesDialog(QDialog):
    """The audio's properties over one document, read-only, with a single Close button."""

    def __init__(self, properties: AudioProperties, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Audio properties"))
        self.properties = properties
        self.values: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for caption, value in properties.rows():
            field = QLabel(value)
            # a value is text to read: not selectable into an editor, and not wrapped into two rows
            field.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            field.setWordWrap(False)
            self.values[caption] = field
            form.addRow(QLabel(caption), field)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def wire(window) -> QAction:
    """An action that opens the panel over `window`'s current audio.

    The panel is built each time it opens, because the values belong to whatever the window holds
    right now; one kept alive would show the previous song's sample rate.
    """

    def show() -> None:
        PropertiesDialog(AudioProperties.from_window(window), window).exec()

    action = QAction(tr("Audio properties…"), window)
    action.triggered.connect(show)
    return action
