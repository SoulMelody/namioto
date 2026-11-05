# SPDX-License-Identifier: AGPL-3.0-only
"""The record window: arm an input, watch the level, stop, and open the take.

`namioto.audio.input.Recorder` already owns the hard parts - the device callback, the ring per
channel, and the wav writer with its `soundfile`-or-`wave` fallback - so this window owns only what
a person touches: a start and a stop button, a clock, a meter, and where the file goes.

Two things it deliberately does *not* do:

* **It does not write the wav itself.** `Recorder.stop(path)` writes it, on the thread that called
  it, which is the GUI thread here; writing from the device callback would be a real-time violation.
  `soundfile` is not a dependency of this project, and the recorder's `wave` fallback already covers
  that, so this window adds no dependency to write a file.
* **It does not choose a device.** A picker here would be a second truth beside a settings window, so
  the window reads `window.settings.audio.device` / `.rate` when such a section exists. It does not
  exist today - `namioto.settings` has general, devices (GPU), tempo, network, midi, editor and
  remote, and no input-device section - so the dialog falls back to the system's default input and
  `Recorder`'s own rate. Both lookups are `getattr`, so adding the section later needs no change here.

`wire(window)` needs, from the window it is given:

* `window.settings.audio` (optional) - `device` (None for the system default) and `rate`. Absent
  today, which means the system's default input at `Recorder`'s rate; a later batch that adds the
  section gets the fields read for free
* `window.open_file(path)` - called with the finished wav so the take opens in the editor straight
  away; a window without it leaves the file on disk and says where it went
* `window.statusBar()` - for the one-line confirmation

`Recorder` itself takes a `stream_factory`, which is the seam the tests use to record without a
device; this window passes whatever `recorder_factory` it was given, so a test can hand over a fake
recorder outright.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from namioto.audio.input import DEFAULT_CHANNELS, Recorder, available
from namioto.i18n import tr

TICK_MS = 100
LEVEL_SCALE = 1000  # a meter runs 0..1000, so an rms of 0.5 reads as half


def take_name() -> str:
    """A file name for a take that has not been saved yet: a wav in the temporary directory."""
    return str(Path(tempfile.gettempdir()) / "namioto-take.wav")


def clock_text(seconds: float) -> str:
    """`mm:ss.d`, the shape a running take's clock reads in."""
    total = max(0.0, float(seconds))
    return f"{int(total // 60):02d}:{total % 60:04.1f}"


class RecordDialog(QDialog):
    """Record from one input device, then hand the file over.

    The recorder is injected (`recorder_factory`), so a test drives a fake one and the dialog's own
    state machine - idle, recording, saving - is what gets exercised, not PortAudio.
    """

    recorded = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        recorder_factory: Callable[[], Recorder] | None = None,
        device=None,
        rate=None,
        can_record: Callable[[], bool] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Record"))
        self._factory = recorder_factory or Recorder
        self._device = device
        self._rate = rate
        self._can_record = can_record or available
        self._recorder: Recorder | None = None
        self._target: Path | None = None
        self._seconds = 0.0

        layout = QVBoxLayout(self)

        self.status = QLabel(self._idle_text())
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.clock = QLabel(clock_text(0.0))
        font = self.clock.font()
        font.setPointSize(font.pointSize() + 6)
        self.clock.setFont(font)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.clock)

        self.meter = QProgressBar()
        self.meter.setRange(0, LEVEL_SCALE)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        layout.addWidget(self.meter)

        buttons = QHBoxLayout()
        self.start_button = QPushButton(tr("Start recording"))
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton(tr("Stop"))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.choose_button = QPushButton(tr("Save as…"))
        self.choose_button.clicked.connect(self.choose_path)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.choose_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.path_label = QLabel(self._path_text())
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.close_box.rejected.connect(self.reject)
        layout.addWidget(self.close_box)

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

        if not self._can_record():
            # no sounddevice means no device can be opened at all; say so instead of offering a
            # button that can only fail
            self.start_button.setEnabled(False)
            self.status.setText(tr("Recording is unavailable: sounddevice is not installed"))

    # -- state -------------------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None and self._recorder.is_recording

    @property
    def seconds(self) -> float:
        return self._seconds

    def _idle_text(self) -> str:
        return tr("Ready to record from the default input")

    def _path_text(self) -> str:
        if self._target is None:
            return tr("Not saved yet")
        return tr("Will be saved to {path}", path=str(self._target))

    def choose_path(self) -> None:
        chosen, _filter = QFileDialog.getSaveFileName(
            self, tr("Save recording"), str(self._target or take_name()), "WAV (*.wav)"
        )
        if chosen:
            self._target = Path(chosen)
            self.path_label.setText(self._path_text())

    # -- recording ---------------------------------------------------------------------------

    def start(self) -> bool:
        """Open the device and begin; False when it will not open, with the reason on screen."""
        if self.is_recording:
            return True
        recorder = self._factory()
        recorder.level.connect(self._on_level)
        recorder.failed.connect(self._on_failed)
        self._recorder = recorder
        opened = recorder.start(device=self._device, rate=self._rate, channels=DEFAULT_CHANNELS)
        if not opened:
            # a refused device has already emitted `failed`, which put the reason on screen
            self._recorder = None
            self._set_running(False)
            return False
        self._seconds = 0.0
        self.clock.setText(clock_text(0.0))
        self.status.setText(tr("Recording…"))
        self._set_running(True)
        self._timer.start()
        return True

    def stop(self) -> str | None:
        """Stop, write the take to the chosen path, and emit `recorded` with it."""
        self._timer.stop()
        if self._recorder is None:
            self._set_running(False)
            return None
        target = self._target or Path(take_name())
        self._target = target
        try:
            self._recorder.stop(str(target))
        except Exception as error:  # a full disk or a read-only directory is a message, not a crash
            self.status.setText(tr("The recording could not be saved: {error}", error=str(error)))
            self._set_running(False)
            return None
        self._recorder = None
        self._set_running(False)
        self.meter.setValue(0)
        self.status.setText(tr("Recorded {time} to {path}", time=clock_text(self._seconds), path=str(target)))
        self.path_label.setText(self._path_text())
        self.recorded.emit(str(target))
        return str(target)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self._can_record())
        self.stop_button.setEnabled(running)
        self.choose_button.setEnabled(not running)

    def _tick(self) -> None:
        self._seconds += TICK_MS / 1000.0
        self.clock.setText(clock_text(self._seconds))

    def _on_level(self, rms: float) -> None:
        """The recorder's own level signal, already the rms of the last block."""
        self.meter.setValue(int(min(1.0, max(0.0, float(rms))) * LEVEL_SCALE))

    def _on_failed(self, message: str) -> None:
        self._timer.stop()
        self._set_running(False)
        self.status.setText(tr("Recording failed: {error}", error=message))


def wire(window) -> QAction:
    """An action that opens the record window and opens the take once it is saved.

    `window.open_file` is the one entry point a file goes through, so a take opens exactly the way a
    file picked from the Open dialog does - asking about unsaved work included.
    """

    def show() -> None:
        audio = getattr(getattr(window, "settings", None), "audio", None)
        dialog = RecordDialog(
            window,
            device=getattr(audio, "device", None),
            rate=getattr(audio, "rate", None),
        )

        def adopt(path: str) -> None:
            opener = getattr(window, "open_file", None)
            if callable(opener):
                opener(path)
            else:
                window.statusBar().showMessage(tr("Recorded to {path}", path=path))

        dialog.recorded.connect(adopt)
        dialog.exec()

    action = QAction(tr("Record…"), window)
    action.triggered.connect(show)
    return action
