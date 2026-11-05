# SPDX-License-Identifier: AGPL-3.0-only
"""The tempo window: what the beats say, what the user taps, and where the grid ends up.

Automatic detection is not always right, and neither is a single tap: this window puts the two
readings side by side over the same 60-240 BPM span so a disagreement is visible before it is
committed, then lets the user type the tempo they know is right.

The histogram is WaveTone's `drawtempogrid` at the same 180 cells over the same 60-240 BPM span.
Its data is `namioto.analysis.tap.tempo_histogram`, which is arithmetic over beat *times*: this
window never runs an FFT or a beat tracker, because the beats it bins already came out of one. If a
caller hands over intervals instead of times, `bins_from_intervals` does the one-BPM binning here,
and it says plainly in its own docstring that this is display-time aggregation, not analysis.

`wire(window)` needs, from the window it is given:

* `window.view.tempo` - the `TempoMap` in force, and `window.view.set_tempo_map(map, text)` to
  change it as one undo step (`set_tempo` alone would not be undoable)
* `window.transport.bpm` - the transport's `TempoBox`; its value follows the accepted tempo
* `window.statusBar()` - for the one-line confirmation
* `window.beats` (optional) - a sequence of beat times in seconds, from the last analysis. Absent
  means the histogram draws empty and only tapping is offered, which is the honest state before any
  analysis has run.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from namioto.analysis import beats as beat_analysis
from namioto.analysis.tap import HISTOGRAM_LOW, HISTOGRAM_STEP, tap_tempo, tempo_histogram
from namioto.i18n import tr
from namioto.tempo_map import TempoMap
from namioto.ui import theme
from namioto.ui.controls import TempoBox

# 60 to 239 at one BPM a cell is the 180 cells WaveTone draws; 240 is the span's top edge rather
# than a cell of its own, which is why the last centre is 239 and not 240
HISTOGRAM_CELLS = 180
HISTOGRAM_TOP = HISTOGRAM_LOW + HISTOGRAM_CELLS - 1
HISTOGRAM_MIN = HISTOGRAM_LOW
HISTOGRAM_MAX = HISTOGRAM_TOP + HISTOGRAM_STEP
BAR_COLOR = QColor("#38b6ff")
AXIS_STEPS = (60, 90, 120, 150, 180, 210, 240)


def bins_from_intervals(intervals: Sequence[float]) -> list[float]:
    """One-BPM bins over `intervals` in seconds, as the 180 weights the histogram draws.

    This is display-time aggregation, not analysis: a caller that has beat *times* should pass them
    to `tap.tempo_histogram` instead, which bins the gaps between them. This exists only for a caller
    that already holds gaps - for instance the ones a previous analysis reported.
    """
    weights = [0.0] * HISTOGRAM_CELLS
    for interval in intervals:
        if interval <= 0.0:
            continue
        bpm = 60.0 / float(interval)
        if not HISTOGRAM_MIN <= bpm <= HISTOGRAM_MAX:
            continue
        index = int(round(bpm - HISTOGRAM_MIN))
        weights[min(HISTOGRAM_CELLS - 1, max(0, index))] += 1.0
    return weights


def histogram_weights(beat_times: Sequence[float]) -> list[float]:
    """The 180 weights for a run of beat times, using the analysis module's own binning."""
    if not beat_times:
        return [0.0] * HISTOGRAM_CELLS
    _centres, weights = tempo_histogram(beat_times, low=HISTOGRAM_MIN, high=HISTOGRAM_MAX, step=HISTOGRAM_STEP)
    padded = list(weights[:HISTOGRAM_CELLS])
    return padded + [0.0] * (HISTOGRAM_CELLS - len(padded))


class TempoHistogram(QWidget):
    """The BPM histogram, painted rather than charted: 180 cells, one BPM each.

    A `QWidget` with a `paintEvent` is what the rest of this app's canvases are, and a chart library
    would bring a whole dependency for one bar chart. Colours come from `theme.canvas()` at paint
    time, so a style switch repaints it in the new set.
    """

    def __init__(self, weights: Sequence[float] = (), parent=None):
        super().__init__(parent)
        self._weights = list(weights) or [0.0] * HISTOGRAM_CELLS
        self.setMinimumHeight(120)
        self.setMinimumWidth(360)

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @property
    def cells(self) -> int:
        return len(self._weights)

    def set_weights(self, weights: Sequence[float]) -> None:
        self._weights = list(weights) or [0.0] * HISTOGRAM_CELLS
        self.update()

    def paintEvent(self, event) -> None:
        del event
        canvas = theme.canvas()
        painter = QPainter(self)
        painter.fillRect(self.rect(), canvas.panel)
        area = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -18.0)
        painter.setPen(QPen(canvas.grid_line))
        painter.drawLine(int(area.left()), int(area.bottom()), int(area.right()), int(area.bottom()))
        peak = max(self._weights) if self._weights else 0.0
        width = area.width() / max(1, len(self._weights))
        painter.setPen(Qt.PenStyle.NoPen)
        for index, weight in enumerate(self._weights):
            if weight <= 0.0 or peak <= 0.0:
                continue
            height = area.height() * (weight / peak)
            cell = QRectF(area.left() + index * width, area.bottom() - height, max(1.0, width), height)
            painter.fillRect(cell, BAR_COLOR)
        painter.setPen(QPen(canvas.text))
        for step in AXIS_STEPS:
            x = area.left() + (step - HISTOGRAM_MIN) / HISTOGRAM_STEP * width
            painter.drawText(QRectF(x - 12.0, area.bottom() + 2.0, 24.0, 14.0), Qt.AlignmentFlag.AlignCenter, str(step))
        painter.end()


class TempoDialog(QDialog):
    """The beats' tempo, a tapped tempo, and the grid the user settles on.

    `bpm()` is the value to commit; `accept()` means the caller adopts it. The dialog itself never
    writes to a window, so it can be built and driven in a test with no `MainWindow` behind it.
    """

    tapped = Signal(float)
    refined = Signal(float)

    def __init__(self, beat_times: Sequence[float] = (), bpm: float = 120.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Tempo"))
        self._beat_times = [float(value) for value in beat_times]
        self._taps: list[float] = []
        self._detected = float(bpm)

        layout = QVBoxLayout(self)

        histogram_group = QGroupBox(tr("Detected tempo"))
        histogram_layout = QVBoxLayout(histogram_group)
        self.histogram = TempoHistogram(histogram_weights(self._beat_times))
        histogram_layout.addWidget(self.histogram)
        self.summary = QLabel(self._summary_text())
        histogram_layout.addWidget(self.summary)
        layout.addWidget(histogram_group)

        tap_group = QGroupBox(tr("Tap the tempo"))
        tap_layout = QHBoxLayout(tap_group)
        self.tap_button = QPushButton(tr("Tap"))
        self.tap_button.setToolTip(tr("Click once per beat; two taps are enough"))
        self.tap_button.clicked.connect(self._on_tap)
        self.reset_button = QPushButton(tr("Reset"))
        self.reset_button.clicked.connect(self._reset_taps)
        self.tap_result = QLabel(tr("Tap at least twice"))
        tap_layout.addWidget(self.tap_button)
        tap_layout.addWidget(self.reset_button)
        tap_layout.addWidget(self.tap_result, 1)
        layout.addWidget(tap_group)

        target_group = QGroupBox(tr("Target tempo"))
        target_layout = QHBoxLayout(target_group)
        self.bpm_box = TempoBox()
        self.bpm_box.setRange(20.0, 300.0)
        self.bpm_box.setDecimals(1)
        self.bpm_box.setValue(self._detected)
        self.bpm_box.setToolTip(tr("The grid snaps to this tempo when the analysis has beats to fit"))
        # refining on `editingFinished` rather than on every keystroke: the box would otherwise move
        # under the user's fingers while they are still typing the tempo they know is right
        self.bpm_box.editingFinished.connect(self._on_refine)
        self.refine_button = QPushButton(tr("Fit to beats"))
        self.refine_button.setToolTip(tr("Move the grid onto the beats nearest this tempo"))
        self.refine_button.clicked.connect(self._on_refine)
        self.refine_button.setEnabled(bool(self._beat_times))
        target_layout.addWidget(QLabel(tr("Tempo")))
        target_layout.addWidget(self.bpm_box)
        target_layout.addWidget(self.refine_button)
        target_layout.addStretch(1)
        layout.addWidget(target_group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- state -------------------------------------------------------------------------------

    def bpm(self) -> float:
        """The tempo the user settled on."""
        return float(self.bpm_box.value())

    def beat_times(self) -> list[float]:
        return list(self._beat_times)

    def tap_times(self) -> list[float]:
        return list(self._taps)

    def _summary_text(self) -> str:
        if not self._beat_times:
            return tr("No beats yet: analyse the audio, or tap the tempo")
        return tr("{count} beats, {bpm} BPM detected", count=len(self._beat_times), bpm=f"{self._detected:.1f}")

    # -- tapping -----------------------------------------------------------------------------

    def _on_tap(self) -> None:
        self._taps.append(time.monotonic())
        self._read_taps()

    def _reset_taps(self) -> None:
        self._taps = []
        self.tap_result.setText(tr("Tap at least twice"))

    def _read_taps(self) -> None:
        tempo = tap_tempo(self._taps)
        if tempo is None:
            self.tap_result.setText(tr("Tap at least twice"))
            return
        self.tap_result.setText(tr("{bpm} BPM from {count} taps", bpm=f"{tempo:.1f}", count=len(self._taps)))
        # the taps are the user's own measurement, so they move the target as they are made
        self.bpm_box.setValue(round(tempo, 1))
        self.tapped.emit(float(tempo))

    # -- refining ----------------------------------------------------------------------------

    def _on_refine(self) -> None:
        if not self._beat_times:
            return
        fitted = refine(self._beat_times, self.bpm())
        self.bpm_box.setValue(round(fitted, 1))
        self.refined.emit(float(fitted))

    def tap_with_times(self, times: Sequence[float]) -> float | None:
        """Record taps at the given times instead of the clock's, so a test can be deterministic."""
        self._taps = [float(value) for value in times]
        self._read_taps()
        return tap_tempo(self._taps)


def refine(beat_times: Sequence[float], target: float) -> float:
    """The tempo near `target` whose grid the beats sit on best, by the analysis module's own fit."""
    if len(beat_times) < 2:
        return float(target)
    return float(beat_analysis.refine_grid(np.asarray(beat_times, dtype=float), float(target)))


def wire(window) -> QAction:
    """An action that opens the tempo window over `window`'s beats and commits the result.

    The commit is one undo step through `set_tempo_map`, so the beats move and the notes stay where
    they sound; a plain `set_tempo` would leave the change out of the undo stack.
    """

    def show() -> None:
        beat_times = list(getattr(window, "beats", ()) or ())
        current = float(getattr(getattr(window, "view", None), "tempo", TempoMap()).bpm)
        dialog = TempoDialog(beat_times, current, window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        tempo = dialog.bpm()
        if abs(tempo - current) < 1e-9:
            return
        window.view.set_tempo_map(TempoMap.single(tempo), tr("Set the tempo"))
        if hasattr(window, "transport"):
            window.transport.bpm.setValue(tempo)
        window.statusBar().showMessage(tr("Tempo set to {bpm} BPM", bpm=f"{tempo:.1f}"))

    action = QAction(tr("Tempo analysis…"), window)
    action.triggered.connect(show)
    return action
