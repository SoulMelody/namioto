# SPDX-License-Identifier: AGPL-3.0-only
"""The alignment window: a lyric in, a time on every note out.

Three stages at most, all of them in processes of their own, so a model crash cannot take the editor
down and this window only spawns them, drains their queues and offers what they found. A separation
child runs first when the source is a mix, because both models hear a voice best without the band.
The words themselves are then either pasted into the box or recognized by a Qwen3-ASR child, and the
alignment child, which reads only the vocal stem, puts a time on each of them.

The box is the source of truth for the words: a run aligns exactly what it holds, so a rounded
character in a recognized line is corrected before it moves anything.
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

from namioto import lyrics as words
from namioto.analysis import alignment, qwen_aligner, qwen_asr, separation
from namioto.i18n import tr
from namioto.params import Field
from namioto.ui.form import add_row, field_editor

POLL_MS = 100
LOG_HEIGHT = 140
# which stage a child is running; a run is at most these three, in this order
SEPARATION = "separation"
RECOGNITION = "recognition"
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


def start_recognition(audio: str, parameters: dict):
    """Spawn the ASR child and queue it reports on."""
    context = multiprocessing.get_context("spawn")
    channel = context.Queue()
    process = context.Process(
        target=alignment.recognize,
        args=(audio, parameters, channel),
        daemon=True,
    )
    process.start()
    return process, channel


def has_runtime() -> bool:
    """Whether this environment can run the model at all; the child says so if it cannot."""
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("qwen_asr") is not None


class AlignDialog(QDialog):
    """The aligner's parameters, the words, and the run, over one audio file.

    `accept()` means the caller puts `lyrics_text()` on the notes of the channel and `spans()` on the
    ones that took a word; a finished alignment accepts on its own once at least one word was placed.
    A recognition never accepts: it only fills the box, and the words stay the user's to correct.
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
        # what a separation child hands over to: the stage that reads the vocal it writes
        self._next = ALIGNMENT
        self._input = pathlib.Path(self.audio)
        self._spans: list[tuple[float, float] | None] = []
        self._initial_text = " ".join(self.words)
        self._settled = True
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self.form = self._build_form()
        self.lyrics = QPlainTextEdit()
        self.lyrics.setPlainText(self._initial_text)
        self.lyrics.setPlaceholderText(tr("Paste lyrics here, or recognize them from the loaded audio"))
        # the box is the window's one input, so it is what decides how wide the window opens
        self.lyrics.setMinimumSize(560, 150)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label = QLabel(tr("Ready"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(LOG_HEIGHT)
        self.log.hide()
        self.recognize_button = QPushButton(tr("Recognize lyrics"))
        self.recognize_button.clicked.connect(self._recognize)
        self.run_button = QPushButton(tr("Align"))
        self.run_button.clicked.connect(self._start)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self._close_or_cancel)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.recognize_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.form)
        layout.addWidget(QLabel(tr("Lyrics")))
        layout.addWidget(self.lyrics, 1)
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

    def lyrics_text(self) -> str:
        return self.lyrics.toPlainText()

    def _word_starts(self, tokens: list[str]) -> list[float]:
        """Where each word starts, which only decides how the audio is cut into windows.

        One word per note keeps the notes' own starts; a text whose word count no longer matches
        the notes is spread evenly over them, so the windows still follow the sung phrases.
        """
        if not tokens:
            return []
        if len(tokens) == len(self.starts):
            return list(self.starts)
        first, last = (self.starts[0], self.starts[-1]) if self.starts else (0.0, 0.0)
        return [first + (last - first) * index / max(1, len(tokens) - 1) for index in range(len(tokens))]

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
        """Rewriting a note's words and timing is not undoable once the window closes, so it is asked for."""
        return (
            QMessageBox.question(
                self,
                tr("Align the lyrics"),
                tr("This rewrites the lyrics and the timing of the notes on the active channel. Continue?"),
            )
            == QMessageBox.StandardButton.Yes
        )

    def _close_or_cancel(self) -> None:
        """The one button closes the window when idle and stops a running model when it is not."""
        if self._process is None:
            self.reject()
        else:
            self._cancel()

    def _set_busy(self, busy: bool) -> None:
        self.form.setEnabled(not busy)
        self.recognize_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.lyrics.setReadOnly(busy)
        if busy:
            self.run_button.setText(tr("Recognizing …") if self._stage == RECOGNITION else tr("Aligning …"))
        else:
            self.run_button.setText(tr("Align"))
        self.close_button.setText(tr("Cancel") if busy else tr("Close"))

    def _recognize(self) -> None:
        """Fill the text box with the words Qwen3-ASR hears in the audio the run reads."""
        if self._process is not None:
            return
        self._parameters = self.parameters()
        self._separation_parameters = self.separation_parameters()
        alignment.save_parameters(self._parameters)
        separation.save_parameters(self._separation_parameters)
        self.log.clear()
        self._spans = []
        self._source(RECOGNITION)

    def _cancel(self) -> None:
        self._stop()
        self._settled = True
        self._spans = []
        self._log_line(tr("Cancelled"))
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Ready"))
        self._set_busy(False)

    def _start(self) -> None:
        text = self.lyrics_text()
        self.words = words.split(text)
        self.starts = self._word_starts(self.words)
        if not self.words:
            self._fail(tr("Enter lyrics or recognize them before aligning"))
            return
        if not self._confirm_target():
            return
        self._parameters = self.parameters()
        self._separation_parameters = self.separation_parameters()
        alignment.save_parameters(self._parameters)
        separation.save_parameters(self._separation_parameters)
        self._source(ALIGNMENT)

    def _source(self, destination: str) -> None:
        """The audio the stage `destination` reads: the file itself, or a vocal stem a separation leaves.

        A stem already on disk is taken at once; one that still has to be computed makes the
        separation child the first stage, and `_next` is what it hands over to.
        """
        self._next = destination
        self._input = pathlib.Path(self.audio)
        if not self._separation_parameters["enabled"]:
            self._advance()
            return
        model = separation.model_name(self._separation_parameters["model"])
        stems = separation.find_stems(self.audio, model)
        if stems is None:
            self._launch_separation()
            return
        self._settle_separation({name: str(path) for name, path in stems.items()})

    def _advance(self) -> None:
        """Hand `_input` to the stage that was waiting for it."""
        if self._next == RECOGNITION:
            self._launch_recognition()
        else:
            self._launch_alignment()

    def _launch_separation(self) -> None:
        self._stage = SEPARATION
        self.log.clear()
        self._spans = []
        self._settled = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Separating …"))
        self._set_busy(True)
        self._log_line(tr("Separating {name} …", name=pathlib.Path(self.audio).name))
        try:
            self._process, self._queue = start_separation(self.audio, self._separation_parameters)
        except Exception as error:  # a child that cannot start must not leave the form stuck
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._timer.start()

    def _launch_recognition(self) -> None:
        """The optional first stage: what the model hears, into the text box for the user to fix."""
        self._stage = RECOGNITION
        if not has_runtime():
            self._fail(tr(qwen_asr.MISSING))
            return
        self._spans = []
        self._settled = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Starting lyric recognition …"))
        self._set_busy(True)
        self._log_line(tr("Recognizing {name} …", name=self._input.name))
        try:
            self._process, self._queue = start_recognition(str(self._input), self._parameters)
        except Exception as error:  # a child that cannot start must not leave the form stuck
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._timer.start()

    def _launch_alignment(self) -> None:
        """The last stage: align `_input`, which a separation may have just named."""
        self._stage = ALIGNMENT
        if not has_runtime():
            self._fail(tr(qwen_aligner.MISSING))
            return
        cached = alignment.find_run(self._input, self.words, self._parameters)
        if cached is not None and cached:
            self._log_line(tr("a stored alignment was reused: {count} words", count=len(self.words)))
            self._settle(cached, save=False)
            return
        self._spans = []
        self._settled = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText(tr("Starting the aligner …"))
        self._set_busy(True)
        self._log_line(tr("Aligning {name} …", name=self._input.name))
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
            if self._stage == RECOGNITION:
                text = str(message[1])
                self._settled = True
                self._timer.stop()
                self._process = None
                self._queue = None
                self.lyrics.setPlainText(text)
                self.progress.setRange(0, 1)
                self.progress.setValue(1)
                self.progress_label.setText(tr("Lyrics recognized"))
                self._log_line(tr("{count} words recognized", count=len(words.split(text))))
                self._set_busy(False)
            elif self._stage == SEPARATION:
                self._settle_separation(dict(message[1]))
            else:
                self._settle(list(message[1]), save=True)
        elif kind == "error":
            self._fail(message[1])

    def _settle_separation(self, stems: dict) -> None:
        """The stems are written; name the vocal and hand over to the stage that asked for it.

        The child has done its whole job, so the timer stops here, but nothing is settled: the next
        stage may still be served from the cache or started against the stem.
        """
        self._timer.stop()
        self._process = None
        self._queue = None
        vocals = stems.get(separation.STEM_VOCALS)
        if vocals is None:
            available = ", ".join(sorted(stems)) or "nothing"
            self._fail(tr("the separator wrote {stems}, with no vocal to read", stems=available))
            return
        self._input = pathlib.Path(vocals)
        self._log_line(tr("Reading the vocals stem"))
        self._advance()

    def _show_progress(self, stage: str, done: int, total: int) -> None:
        if stage == "weights":
            label = tr("Fetching the weights … {done}/{total}", done=done, total=total)
        elif stage == "separating":
            label = tr("Separating … {done}/{total}", done=done, total=total)
        elif stage == "download":
            # the separation catalog reports nothing, so the bar does not pretend to know
            label = tr("Downloading the separation model …")
        elif stage == "recognizing":
            label = tr("Recognizing …")
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
        self._set_busy(False)
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
        self._set_busy(False)

    def _crash(self, exitcode) -> None:
        if self._stage == RECOGNITION:
            self._log_line(tr("Lyric recognition stopped unexpectedly (exit code {code})", code=exitcode))
        else:
            self._log_line(tr("The aligner stopped unexpectedly (exit code {code})", code=exitcode))
        self.progress.setValue(0)
        self.progress_label.setText(tr("Failed"))
        self._set_busy(False)

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
