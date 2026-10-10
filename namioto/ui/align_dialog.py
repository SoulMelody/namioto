# SPDX-License-Identifier: AGPL-3.0-only
"""The alignment window: put a time on every sound by forcing the lyrics onto the audio.

The run is a `LoadingThread` of its own (`Aligner`), because the model is an ONNX graph and the
audio is read once; the window only starts it, shows the progress, and shows the failures
`namioto.analysis.align` reports. The whole stream is aligned in one pass, the way FA-Kara does it. Its
choices are `align.PARAMETERS`, remembered between runs in the file `align.parameter_path()` names.
Its Quantize choice (Off / 1/4 / 1/8 / ..., GAME's list) snaps onsets onto the drawn beat grid
before mapping, preserving the reference durations captured from the alignment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from namioto.analysis import align, devices
from namioto.i18n import tr
from namioto.karaoke.sounds import natural_sounds, natural_tokens, split_tokens
from namioto.lyricmap import from_spans, snap_to_beats
from namioto.params import Field
from namioto.ui.form import add_row, field_editor
from namioto.ui.loading import LoadingThread

LOG_HEIGHT = 120


class Aligner(LoadingThread):
    """One pass of the forced aligner over the whole audio: the times, and the lines it doubts."""

    progress = Signal(int, int)
    aligned = Signal(object)
    message = Signal(str)

    def __init__(
        self,
        audio: str,
        text: str,
        model: str = align.DEFAULT_MODEL,
        provider: str = "cpu",
        chunk: str = align.DEFAULT_MODE,
        parent=None,
    ):
        super().__init__(audio, parent)
        self.text = text
        self.model = model
        self.provider = provider
        self.chunk = chunk
        self._downloaded = -10

    def load(self) -> None:
        started = time.monotonic()
        lines = natural_sounds(self.text)
        sounds = sum(len(line.sounds) for line in lines)
        self.message.emit(tr("Read {lines} lines, {sounds} sounds", lines=len(lines), sounds=sounds))
        audio = align.load_audio(self.path)
        seconds = len(audio) / align.SAMPLE_RATE
        emission = align.load_emissions(self.path, self.model, self.provider, self.chunk)
        directory = align.resolve_model(None, self.model, "ja", progress=self._downloading)
        if emission is None:
            self.message.emit(
                tr("Loading the {model} model on {provider}\u2026", model=self.model, provider=self.provider)
            )
            align.fetch_fp16(directory, self.provider, self.model, progress=self._downloading)
            backend = align.OnnxBackend(align.model_file(directory, self.provider), provider=self.provider)
            if self.chunk == "off":
                self.message.emit(tr("Aligning over {seconds:.1f}s of audio in one pass\u2026", seconds=seconds))
            else:
                backend = align.ChunkedBackend(backend, silence=self.chunk == "silence", progress=self.progress.emit)
                self.message.emit(tr("Aligning over {seconds:.1f}s of audio in chunks\u2026", seconds=seconds))
            emission = align.whole_emissions(backend, audio)
            with suppress(OSError):
                align.save_emissions(self.path, self.model, self.provider, self.chunk, emission)
        else:
            self.message.emit(tr("Reusing the model's pass over the audio\u2026"))
        dictionary, blank_id = align.load_dictionary(directory / align.VOCAB_FILE)
        segment = align.Segment(0.0, seconds, tuple(natural_tokens(lines)))
        found = align.align_whole(segment, emission, dictionary, blank_id=blank_id)
        self.message.emit(tr("Fitting the sounds to the voice\u2026"))
        rows = align.correct_times(split_tokens(found.tokens, lines), audio)
        flagged, problems = self._problems(found, lines)
        with suppress(OSError):  # the cache is disposable, and the alignment itself already came back
            align.save_alignment(self.path, self.model, self.provider, self.text, rows, problems, flagged, self.chunk)
        self.message.emit(tr("Alignment took {seconds:.1f}s", seconds=time.monotonic() - started))
        self.aligned.emit((rows, self.model, problems, flagged))

    def _downloading(self, done: int, total: int) -> None:
        """A download of a model that was not installed, logged a tenth at a time."""
        if not total:
            return
        percent = done * 100 // total
        if percent < self._downloaded:  # the half-precision package starts a second download
            self._downloaded = -10
        if percent >= self._downloaded + 10:
            self._downloaded = percent
            self.message.emit(tr("Downloading the {model} model\u2026 {percent}%", model=self.model, percent=percent))

    @staticmethod
    def _problems(found, lines) -> tuple[list[bool], list[str]]:
        """One flag per line the aligner doubted, and the same trouble named for the log."""
        flagged, reported = [], []
        at = 0
        for line in lines:
            chunk = found.tokens[at : at + len(line.sounds)]
            at += len(chunk)
            trouble = align.problems(align.AlignedSegment(found.start, found.end, tuple(chunk)))
            flagged.append(bool(trouble))
            if trouble:
                reported.append(f"{line.text}: {', '.join(trouble)}")
        return flagged, reported


class AlignDialog(QDialog):
    """The model choice, the run, and what it found. The result is handed over as it arrives."""

    aligned = Signal(object, str, object)

    def __init__(self, audio: str, text: str, tempo: float, parent=None, offset: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle(tr("Align lyrics"))
        self.audio = audio
        self.text = text
        self.tempo = tempo
        self.offset = offset
        self._thread: Aligner | None = None
        # the choices the last run was made with, straight onto the form
        self._parameters = align.load_parameters()
        self._fields: list[tuple[Field, Callable[[], object], Callable[[object], None]]] = []

        form = self._build_form()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status = QLabel(tr("Ready"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(LOG_HEIGHT)
        self.run = QPushButton(tr("Run"))
        self.run.clicked.connect(self._start)
        buttons = QDialogButtonBox()
        buttons.addButton(self.run, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(form)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        layout.addWidget(buttons)

    def parameters(self) -> dict:
        """What the form holds now, checked the way the store checks it."""
        return align.coerce_parameters({field.name: read() for field, read, _write in self._fields})

    def _build_form(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for item in align.PARAMETERS:
            editor, read, write = field_editor(self._parameters[item.name], item)
            add_row(form, editor, item)
            self._fields.append((item, read, write))
        return widget

    def _remember(self) -> None:
        """Keep what the form holds, so the next window opens on it."""
        self._parameters = self.parameters()
        align.save_parameters(self._parameters)

    def _start(self) -> None:
        if self._thread is not None:
            return
        problem = devices.validate()
        if problem is not None:
            self._fail(tr(problem))
            return
        self.log.clear()
        self.run.setEnabled(False)
        self._remember()
        model = self._parameters["model"]
        provider = devices.resolve(self._parameters["device"])
        if self._parameters["device"] == "gpu" and provider == "cpu":
            self.log.appendPlainText(tr("No GPU backend is available; running on the CPU"))
        chunk = self._parameters["chunk"]
        cached = align.find_alignment(self.audio, model, provider, self.text, chunk)
        if cached is not None:
            rows, problems, flagged = cached
            self._done((rows, model, problems, flagged))  # re-snapped with the tempo and grid offset in hand
            self.log.setPlainText("\n".join([tr("saved alignment reused"), *problems]))
            return
        self.progress.setRange(0, 0)  # busy until the first chunk of the model's pass comes back
        self.status.setText(tr("Aligning…"))
        self._thread = Aligner(self.audio, self.text, model, provider, chunk, self)
        self._thread.progress.connect(self._show_progress)
        self._thread.message.connect(self.log.appendPlainText)
        self._thread.aligned.connect(self._done)
        self._thread.failed.connect(self._fail)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _done(self, result) -> None:
        spans, model, problems, flagged = result
        times = from_spans(spans)
        cells = self.parameters()["quantize"]  # read now, so a change made while the run went counts
        if cells:
            times = snap_to_beats(times, self.tempo, 1.0 / cells, self.offset)
        self._thread = None
        self.run.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status.setText(tr("Aligned {lines} lines", lines=len(times)))
        if problems:
            self.log.appendPlainText("\n".join(problems))
        self.aligned.emit(times, model, flagged)

    def _fail(self, message: str) -> None:
        self._thread = None
        self.run.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(tr("Alignment failed"))
        self.log.setPlainText(message)

    def reject(self) -> None:
        self._remember()
        if self._thread is not None:
            self._thread.wait()
        super().reject()
