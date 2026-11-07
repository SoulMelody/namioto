# SPDX-License-Identifier: AGPL-3.0-only
"""The alignment window: put a time on every note from the word it sings.

Two stages, like the transcription window. A separation child runs first when the source is a mix,
because the aligner was trained on speech and hears a voice best without the band; the alignment
child then reads only the vocal stem. Both live in processes of their own, so a model crash cannot
take the editor down and this window only spawns them, drains their queues and offers the spans.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import pathlib
import queue
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

from namioto.analysis import alignment, qwen_aligner, separation
from namioto.i18n import tr
from namioto.params import Field
from namioto.ui.form import add_row, field_editor

POLL_MS = 100
LOG_HEIGHT = 140
# which stage a child is running; a run is at most these two, in this order
SEPARATION = "separation"
ALIGNMENT = "alignment"


def start_job(audio: str, words: list[str], starts: list[float], parameters: dict):
    """Spawn the alignment child and the queue it reports on; the one seam the tests replace."""
    context = multiprocessing.get_context("spawn")
    channel = context.Queue()
    process = context.Process(
        target=alignment.align,
        args=(audio, words, starts, parameters, channel),
        daemon=True,
    )
    process.start()
    return process, channel


def start_separation(audio: str, parameters: dict):
    """Spawn the separation child and the queue it reports on; the second seam the tests replace."""
    context = multiprocessing.get_context("spawn")
    channel = context.Queue()
    process = context.Process(
        target=separation.separate,
        args=(audio, parameters, channel),
        daemon=True,
    )
    process.start()
    return process, channel


def has_runtime() -> bool:
    """Whether this environment can run the model at all; the child says so if it cannot."""
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("qwen_asr") is not None


class AlignDialog(QDialog):
    """The aligner's parameters and the run, over one audio file.

    `accept()` means the caller puts `spans()` on the notes it opened the window for; a finished run
    accepts on its own once at least one word was placed.
    """

    def __init__(
        self,
        audio: str,
        words: list[str],
        starts: list[float],
        tempo: float,
        parent=None,
        offset: float = 0.0,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Align the lyrics"))
        self.audio = str(audio)
        self.words = [str(word) for word in words]
        self.starts = [float(start) for start in starts]
        self.tempo = float(tempo)
        self.offset = float(offset)
        self._parameters = alignment.load_parameters()
        self._separation_parameters = separation.load_parameters()
        self._fields: list[tuple[Field, Callable[[], object], Callable[[object], None]]] = []
        self._separation_fields: list[tuple[Field, Callable[[], object], Callable[[object], None]]] = []
        self._process = None
        self._queue = None
        self._stage = ALIGNMENT
        self._input = pathlib.Path(self.audio)
        self._spans: list[tuple[float, float] | None] = []
        self._settled = True
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
        self.run_button = QPushButton(tr("Align"))
        self.run_button.clicked.connect(self._start)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self._close_or_cancel)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.form)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        layout.addLayout(buttons)
        self._fit()

    def parameters(self) -> dict:
        """What the form holds now, checked the way the store checks it."""
        return alignment.coerce_parameters({field.name: read() for field, read, _write in self._fields})

    def separation_parameters(self) -> dict:
        return separation.coerce_parameters({field.name: read() for field, read, _write in self._separation_fields})

    def spans(self) -> list[tuple[float, float] | None]:
        return list(self._spans)

    def _build_form(self) -> QWidget:
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for item in alignment.PARAMETERS:
            editor, read, write = field_editor(self._parameters[item.name], item)
            add_row(form, editor, item)
            self._fields.append((item, read, write))
        layout.addLayout(form)
        # the separation rows follow the aligner's: what they do is about the source those rows name
        source = QFormLayout()
        source.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for item in separation.PARAMETERS:
            editor, read, write = field_editor(self._separation_parameters[item.name], item)
            add_row(source, editor, item)
            self._separation_fields.append((item, read, write))
        layout.addLayout(source)
        return widget

    def _fit(self) -> None:
        """Pin the window to its content: Hyprland resizes a native toplevel only while it is rigid,
        `min == max`, and ignores the request otherwise. https://github.com/hyprwm/Hyprland/issues/3167
        """
        self.form.updateGeometry()
        self.layout().invalidate()
        self.setFixedSize(self.sizeHint())

    def _confirm_target(self) -> bool:
        """Rewriting a note's timing is not undoable once the window closes, so it is asked for."""
        return (
            QMessageBox.question(
                self,
                tr("Align the lyrics"),
                tr(
                    "This rewrites the timing of {count} notes on the active channel. Continue?",
                    count=len(self.words),
                ),
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
        self._spans = []
        self._log_line(tr("Cancelled"))
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Ready"))
        self._set_running(False)

    def _start(self) -> None:
        if not self.words:
            self._fail(tr("The active channel has no lyrics to align"))
            return
        if not self._confirm_target():
            return
        self._parameters = self.parameters()
        self._separation_parameters = self.separation_parameters()
        alignment.save_parameters(self._parameters)
        separation.save_parameters(self._separation_parameters)
        if not self._separation_parameters["enabled"]:
            self._input = pathlib.Path(self.audio)
            self._launch()
            return
        model = separation.model_name(self._separation_parameters["model"])
        stems = separation.find_stems(self.audio, model)
        if stems is not None:
            self._settle_separation({name: str(path) for name, path in stems.items()})
            return
        self._launch_separation()

    def _launch_separation(self) -> None:
        self._stage = SEPARATION
        self.log.clear()
        self._spans = []
        self._settled = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Separating …"))
        self._set_running(True)
        self._log_line(tr("Separating {name} …", name=pathlib.Path(self.audio).name))
        try:
            self._process, self._queue = start_separation(self.audio, self._separation_parameters)
        except Exception as error:  # a child that cannot start must not leave the form stuck
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._timer.start()

    def _launch(self) -> None:
        """Stage two, or the only stage: align `_input`, which a separation may have just named."""
        self._stage = ALIGNMENT
        if not has_runtime():
            self._fail(tr(qwen_aligner.MISSING))
            return
        self._log_line(tr("Aligning {name} …", name=self._input.name))
        self._spans = []
        self._settled = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Starting the aligner …"))
        self._set_running(True)
        try:
            self._process, self._queue = start_job(str(self._input), self.words, self.starts, self._parameters)
        except Exception as error:  # a child that cannot start must not leave the form stuck
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._timer.start()

    def _poll(self) -> None:
        process = self._process
        self._drain()
        # a finished stage may have handed over to the next one, which reports on later ticks
        if process is None or self._process is not process:
            return
        if self._settled or process.is_alive():
            return
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
            if self._stage == SEPARATION:
                self._settle_separation(dict(message[1]))
            else:
                self._settle(list(message[1]), save=True)
        elif kind == "error":
            self._fail(message[1])

    def _settle_separation(self, stems: dict) -> None:
        """The stems are written; hand the vocal to the alignment stage.

        The child has done its whole job, so the timer stops here, but nothing is settled: an
        alignment may still be served from the cache or started against the stem.
        """
        self._timer.stop()
        self._process = None
        self._queue = None
        vocals = stems.get(separation.STEM_VOCALS)
        if vocals is None:
            available = ", ".join(sorted(stems)) or "nothing"
            self._fail(tr("the separator wrote {stems}, with no vocal to align against", stems=available))
            return
        self._input = pathlib.Path(vocals)
        self._log_line(tr("Aligning the vocals stem"))
        cached = alignment.find_run(self._input, self.words, self._parameters)
        if cached is not None and cached:
            self._log_line(tr("a stored alignment was reused: {count} words", count=len(self.words)))
            self._settle(cached, save=False)
            return
        self._launch()

    def _show_progress(self, stage: str, done: int, total: int) -> None:
        if stage == "weights":
            label = tr("Fetching the weights … {done}/{total}", done=done, total=total)
        elif stage == "separating":
            label = tr("Separating … {done}/{total}", done=done, total=total)
        elif stage == "download":
            # the separation catalog reports nothing, so the bar does not pretend to know
            label = tr("Downloading the separation model …")
        else:
            label = tr("Aligning … {done}/{total}", done=done, total=total)
        self.progress_label.setText(label)
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        else:
            self.progress.setRange(0, 0)

    def _settle(self, spans, save: bool) -> None:
        self._settled = True
        self._timer.stop()
        self._process = None
        self._queue = None
        raw = list(spans)
        self._spans = self._snapped(raw)
        if save:
            try:
                alignment.save_run(self._input, self.words, self._parameters, raw)
            except OSError as error:
                self._log_line(tr("the result could not be saved: {error}", error=error))
        placed = sum(1 for span in self._spans if span is not None)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress_label.setText(tr("{count} of {total} words placed", count=placed, total=len(self.words)))
        self._log_line(tr("{count} of {total} words placed", count=placed, total=len(self.words)))
        self._set_running(False)
        if placed:
            self.accept()

    def _snapped(self, spans) -> list[tuple[float, float] | None]:
        """The model's spans on the grid the editor draws, or as they are when quantize is off."""
        cells = self._parameters["quantize"]
        if not cells:
            return list(spans)
        return alignment.snap_onsets(spans, self.tempo, cells, phase=self.offset)

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
        self._log_line(tr("The aligner stopped unexpectedly (exit code {code})", code=exitcode))
        self.progress.setValue(0)
        self.progress_label.setText(tr("Failed"))
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.form.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.run_button.setText(tr("Aligning …") if running else tr("Align"))
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
