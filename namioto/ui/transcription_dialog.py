# SPDX-License-Identifier: AGPL-3.0-only
"""The transcription window: GAME's parameters, the run's progress and log, and what it found.

The run lives in a process of its own (`namioto.analysis.transcription.transcribe`), so a model crash cannot
take the editor down; this window only spawns it, drains its queue and offers the result.
"""

from __future__ import annotations

import multiprocessing
import pathlib
import queue
import shutil
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from namioto.analysis import devices, transcription
from namioto.i18n import tr
from namioto.params import Field
from namioto.ui.form import add_row, advanced_section, field_editor

POLL_MS = 100
LOG_HEIGHT = 140
# the rows the form folds away: the ones only worth reaching for when the small model misses
ADVANCED = ("batch_size", "seg_threshold", "seg_radius", "est_threshold", "d3pm_t0", "d3pm_steps", "silence_slice")


def start_job(audio: str, parameters: dict):
    """Spawn the GAME child and the queue it reports on; the one seam the tests replace."""
    context = multiprocessing.get_context("spawn")
    channel = context.Queue()
    process = context.Process(
        target=transcription.transcribe,
        args=(audio, parameters, channel),
        daemon=True,
    )
    process.start()
    return process, channel


class TranscriptionDialog(QDialog):
    """GAME's parameters and the run, over one audio file.

    `accept()` means the caller replaces channel 0 with `notes()`; a finished run accepts on its own.
    """

    def __init__(self, audio: str, tempo: float, parent=None, channel_has_notes: bool = False, offset: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle(tr("Transcribe the singing voice with GAME"))
        self.audio = str(audio)
        self.tempo = float(tempo)
        self.offset = float(offset)
        self.channel_has_notes = bool(channel_has_notes)
        self._parameters = transcription.load_parameters()
        self._fields: list[tuple[Field, Callable[[], object], Callable[[object], None]]] = []
        self._process = None
        self._queue = None
        self._notes: list[tuple[float, float, float]] = []
        self._settled = True
        self._downloading = False
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self.form = self._build_form()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label = QLabel(tr("Ready"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(LOG_HEIGHT)
        self.log.hide()
        self.run_button = QPushButton(tr("Transcribe"))
        self.run_button.clicked.connect(self._start)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self._close_or_cancel)

        status = QHBoxLayout()
        status.addWidget(self.progress, 1)
        status.addWidget(self.progress_label)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.run_button)
        actions.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.form)
        layout.addWidget(self.log, 1)
        layout.addLayout(status)
        layout.addLayout(actions)
        self._fit()

    def parameters(self) -> dict:
        """What the form holds now, checked the way the store checks it."""
        return transcription.coerce_parameters({field.name: read() for field, read, _write in self._fields})

    def notes(self) -> list[tuple[float, float, float]]:
        return list(self._notes)

    def _build_form(self) -> QWidget:
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for advanced in (False, True):
            items = [item for item in transcription.PARAMETERS if (item.name in ADVANCED) is advanced]
            if not items:
                continue
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            for item in items:
                editor, read, write = field_editor(self._parameters[item.name], item)
                add_row(form, editor, item)
                self._fields.append((item, read, write))
            if advanced:
                heading = advanced_section(form)
                layout.addWidget(heading)
                heading.toggled.connect(lambda _open: QTimer.singleShot(0, self._fit))
            layout.addLayout(form)
        return widget

    def _fit(self) -> None:
        """Pin the window to its content: Hyprland resizes a native toplevel only while it is rigid,
        `min == max`, and ignores the request otherwise. https://github.com/hyprwm/Hyprland/issues/3167

        Showing or hiding rows leaves the layout's cached size hint behind, so drop it first; the
        deferred callers let the layout update land before the size is read.
        """
        self.form.updateGeometry()
        self.layout().invalidate()
        self.setFixedSize(self.sizeHint())

    def _confirm_target(self) -> bool:
        if not self.channel_has_notes:
            return True
        return (
            QMessageBox.question(
                self,
                tr("Transcribe with GAME"),
                tr("Channel 0 already has notes. Replace them with the transcription?"),
            )
            == QMessageBox.StandardButton.Yes
        )

    def _close_or_cancel(self) -> None:
        """The one button closes the window when idle and stops a running model when it is not."""
        if self._process is None:
            self.reject()
        else:
            self._cancel()

    def _cancel(self) -> None:
        self._stop()
        self._settled = True
        self._notes = []
        self._log_line(tr("Cancelled"))
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Ready"))
        self._set_running(False)

    def _start(self) -> None:
        if not self._confirm_target():
            return
        self._parameters = self.parameters()
        transcription.save_parameters(self._parameters)
        cached = transcription.find_run(self.audio, self._parameters)
        if cached is not None and self._use_cache(len(cached)):
            self._log_line(tr("saved run reused: {count} notes", count=len(cached)))
            self._settle(cached, save=False)
            return
        self._launch()

    def _use_cache(self, count: int) -> bool:
        """Ask before a repeat run is skipped; the notes are never listed anywhere, only offered back."""
        answer = QMessageBox.question(
            self,
            tr("Transcribe with GAME"),
            tr(
                "A run with these exact parameters already found {count} notes.\n"
                "Use it instead of running the model again?",
                count=count,
            ),
        )
        return answer == QMessageBox.StandardButton.Yes

    def _launch(self) -> None:
        self.log.clear()
        problem = devices.validate()
        if problem is not None:
            self._fail(tr(problem))
            return
        self._log_line(tr("Transcribing {name} …", name=pathlib.Path(self.audio).name))
        self._notes = []
        self._settled = False
        self._downloading = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Starting GAME …"))
        self._set_running(True)
        try:
            self._process, self._queue = start_job(self.audio, self._parameters)
        except Exception as error:  # a child that cannot start must not leave the form stuck
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._timer.start()

    def _poll(self) -> None:
        self._drain()
        process = self._process
        if process is None or self._settled or process.is_alive():
            return
        self._drain()  # the last messages can land just after the child exits
        self._timer.stop()
        self._process = None
        self._queue = None
        if not self._settled:
            self._crash(process.exitcode)

    def _drain(self) -> None:
        while self._queue is not None:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                return
            self._handle(message)

    def _handle(self, message) -> None:
        kind = message[0]
        if kind == "log":
            self._log_line(message[1])
        elif kind == "progress":
            self._show_progress(*message[1:])
        elif kind == "done":
            self._settle(list(message[1]), save=True)
        elif kind == "error":
            self._fail(message[1])

    def _show_progress(self, stage: str, done: int, total: int) -> None:
        if stage == "download":
            self._downloading = True
            where = tr(" of {total}", total=total // (1 << 20)) if total else ""
            self.progress_label.setText(
                tr("Downloading the model … {done}{where} MB", done=done // (1 << 20), where=where)
            )
        else:
            self.progress_label.setText(tr("Extracting … {done}/{total}", done=done, total=total))
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _settle(self, notes, save: bool) -> None:
        self._settled = True
        self._timer.stop()
        self._process = None
        self._queue = None
        self._notes = self._snapped(notes)
        if save:
            try:
                transcription.save_run(self.audio, self._parameters, notes)  # raw: the grid is applied here
            except OSError as error:
                self._log_line(tr("the result could not be saved: {error}", error=error))
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress_label.setText(tr("{count} notes", count=len(self._notes)))
        self._log_line(tr("{count} notes", count=len(self._notes)))
        self._set_running(False)
        if self._notes:
            self.accept()

    def _snapped(self, notes) -> list[tuple[float, float, float]]:
        """GAME's own notes on the grid the editor draws: the run's quantize choice, the tempo and the
        grid offset go on here, not in the model run, so changing any of them re-snaps a stored run
        instead of asking for another one."""
        raw = [(float(onset), float(offset), float(pitch)) for onset, offset, pitch in notes]
        cells = self._parameters["quantize"]
        if not cells:
            return raw
        from namioto.analysis import game

        snapped, unit, phase = game.quantized(raw, self.tempo, cells, offset=self.offset)
        self._log_line(f"grid {unit * 1000:.1f} ms, phase {phase * 1000:.1f} ms, offset {self.offset * 1000:.1f} ms")
        return [(float(onset), float(offset), float(pitch)) for onset, offset, pitch in snapped]

    def _fail(self, message: str) -> None:
        self._settled = True
        self._timer.stop()
        self._process = None
        self._queue = None
        for line in str(message).strip().splitlines()[-8:]:
            self._log_line(line)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Failed"))
        self._set_running(False)

    def _crash(self, exitcode) -> None:
        self._log_line(tr("GAME stopped unexpectedly (exit code {code})", code=exitcode))
        self.progress.setValue(0)
        self.progress_label.setText(tr("Failed"))
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.form.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.run_button.setText(tr("Transcribing …") if running else tr("Transcribe"))
        self.close_button.setText(tr("Cancel") if running else tr("Close"))

    def _log_line(self, text: str) -> None:
        self.log.show()
        QTimer.singleShot(0, self._fit)
        self.log.appendPlainText(text)

    def _stop(self) -> None:
        self._timer.stop()
        process, self._process = self._process, None
        self._queue = None
        if process is not None and process.is_alive():
            process.terminate()
            process.join(2000)
            self._sweep()

    def _sweep(self) -> None:
        """A killed download leaves its archive in the model directory; it is worth no keeping."""
        if not self._downloading:
            return
        from namioto.analysis import game

        for stale in game.models_root().glob("tmp*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)

    def reject(self) -> None:
        self._stop()
        super().reject()

    def closeEvent(self, event) -> None:
        self._stop()
        super().closeEvent(event)
