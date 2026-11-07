# SPDX-License-Identifier: AGPL-3.0-only
"""The editor window and its wiring: the bars, the roll, the sidebar and the status bar.

`MainWindow` owns the document, the players, the program's own preferences and the window's state,
and every path the program opens or writes, and runs in this one process - the analysis that must not
touch the GUI goes out to the `LoadingThread` loaders at the top of the file, or to a spawned
`namioto-transcription` child. Run it with `uv run namioto`.

Every bar value is one `_make_bindings` entry; `_remember_configuration` and the apply path both
walk that table, and `_seeding` keeps an apply's quiet writes from counting as user edits. A project
is the document: opening audio needs one (the sibling `.nto` when there is one, else a name from the
chooser), a MIDI is imported into the project already open and never on its own, and the Open dialog
offers its filter only inside a project; switching documents goes through `_confirm_discard` first.
The Export button writes the roll out as a MIDI file. A lyric is a word on a note, so the project
file keeps it with the note and nothing else carries a copy of the words.

Dirty state covers notes, channels, tempo and audio, and is asked about only once
the document has a file name. `general.auto_save` (off by default) writes the open project once an
edit settles (`AUTOSAVE_DELAY_MS`) and when the window loses focus; only a named document is written,
and a modal dialog holds the focus trigger back so the save-on-close prompt never answers itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QByteArray,
    QCoreApplication,
    QEvent,
    QLibraryInfo,
    Qt,
    QThread,
    QTimer,
    QTranslator,
    Signal,
)
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from namioto import i18n, ipc, midi, midi_port, params, project, qtplugins
from namioto import lyrics as words
from namioto import settings as store
from namioto import state as window_state
from namioto.analysis import bpm, harmonic
from namioto.analysis.spectrum import CHANNEL_MODES, NoteSpectrum
from namioto.channels import Channel, free_channel
from namioto.channels import audible as audible_channels
from namioto.channels import set_field as channel_set_field
from namioto.markers import MarkerList
from namioto.meter_map import MeterMap, MeterPoint
from namioto.mtc import Rate
from namioto.playback import note_frequency
from namioto.tempo_map import TempoMap
from namioto.ui import theme
from namioto.ui.align_dialog import AlignDialog
from namioto.ui.audio import open_player
from namioto.ui.channel_panel import ChannelPanel
from namioto.ui.controls import ControlArea, EditBar, MixBar, TransportBar
from namioto.ui.ipc_commands import WindowBridge
from namioto.ui.ipc_server import IpcServer
from namioto.ui.loading import LoadingThread
from namioto.ui.lyrics_dialog import LyricsDialog
from namioto.ui.midi_dialog import MidiImportDialog
from namioto.ui.mtc_port import MtcPort
from namioto.ui.open_dialog import OpenAudioDialog
from namioto.ui.properties_dialog import wire as properties_action
from namioto.ui.record_dialog import wire as record_action
from namioto.ui.roll import PITCH_MAX, PITCH_MIN, SNAP_CHOICES, MarkerTrack, PianoKeyboard, PianoRollView, TimelineRuler
from namioto.ui.settings_dialog import SettingsDialog, SettingsStore
from namioto.ui.shortcuts_dialog import wire as shortcuts_action
from namioto.ui.song import SongPlayer, load_song
from namioto.ui.spectrogram import HarmonicLoader, SpectrumLoader
from namioto.ui.tempo_dialog import wire as tempo_action
from namioto.ui.text import format_time, note_name
from namioto.ui.transcription_dialog import TranscriptionDialog

POSITION_INTERVAL_MS = 40
SPEED_SETTLE_MS = 100
AUTOSAVE_DELAY_MS = 3000
# long enough to read the span and short enough that the message does not hold the status bar
LOOP_MESSAGE_MS = 3000
# the same bargain for the opening hint: it says where the help is, then gives the bar back
INTRO_MESSAGE_MS = 8000
# MTC chase, in the units the sync port hands over: seconds of drift and a rate around 1.0
MTC_SEEK_SECONDS = 0.05  # 50 ms: past this a jump is cheaper than a nudge
MTC_TRIM_GAIN = 0.5  # the proportional term of the nudge
MTC_TRIM_LIMIT = 0.05  # any more than 5% and the correction is heard as a wobble
# the file's spelling of each rate, so a settings value can be a person's word rather than an enum
MTC_RATES = {"24": Rate.FPS_24, "25": Rate.FPS_25, "29.97drop": Rate.FPS_29_97_DROP, "30": Rate.FPS_30}
# common libsndfile formats; anything rarer is reachable through All files
AUDIO_SUFFIXES = (
    ".wav",
    ".wave",
    ".flac",
    ".mp3",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".aifc",
    ".au",
    ".caf",
    ".w64",
    ".rf64",
)


def _project_filter() -> str:
    return i18n.tr("Namioto project (*{suffix})", suffix=project.SUFFIX)


def _audio_filter() -> str:
    return i18n.tr("Audio file ({patterns})", patterns=" ".join(f"*{suffix}" for suffix in AUDIO_SUFFIXES))


def _midi_filter() -> str:
    return i18n.tr("MIDI file ({patterns})", patterns=" ".join(f"*{suffix}" for suffix in midi.SUFFIXES))


def _label(text: str) -> QLabel:
    """A caption in the interface's language: the dialogs here build their own rows."""
    return QLabel(i18n.tr(text))


def loop_target(span: tuple[float, float] | None, position: float) -> float | None:
    """Where a repeating transport jumps to, or None to let it run on.

    The whole A-B policy is this one comparison, kept out of the tick handler so a test can drive it
    without a timer: an armed span sends the playhead back to A as soon as it reaches B. Seconds, the
    same clock `player.position` runs on, so no beat conversion stands between the span and the seek.
    """
    if span is None:
        return None
    start, end = span
    return start if position >= end else None


def loop_text(span: tuple[float, float] | None) -> str:
    """The span as the status bar shows it: the two times, or that there is none.

    The label that used to carry this text is gone with the bar's A-B card, so the status bar is
    where the times are now; `_show_loop` gives them a timeout, since the window shows this while it
    is still opening and a message without one would sit over everything said afterwards.
    """
    if span is None:
        return i18n.tr("A-B off")
    return f"A {format_time(span[0])}  B {format_time(span[1])}"


def mtc_correction(drift: float) -> tuple[str, float]:
    """What a position from the sync port does to the transport: `("seek", 0.0)` or `("trim", rate)`.

    50 ms is where a jump is worth more than a nudge; below that the transport keeps its place and
    only the rate moves - by half the drift, clamped to +/-5%, which is as far as a rate change stops
    being audible as a wobble. Negative drift (the port is behind us) follows the same rule.
    """
    if abs(drift) > MTC_SEEK_SECONDS:
        return ("seek", 0.0)
    return ("trim", clamp_rate(1.0 + MTC_TRIM_GAIN * drift))


def clamp_rate(speed: float) -> float:
    """A rate change is only ever a nudge: outside +/-5% it stops correcting and starts wobbling."""
    return max(1.0 - MTC_TRIM_LIMIT, min(1.0 + MTC_TRIM_LIMIT, speed))


_translators: list[QTranslator] = []


def _install_translations(language: str) -> None:
    """Load Qt's own translations, so standard buttons and file dialogs follow the chosen language."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    for translator in _translators:
        app.removeTranslator(translator)
    _translators.clear()
    locale = i18n.LOCALES.get(language)
    if language == i18n.DEFAULT or locale is None:
        return
    folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for name in (f"qtbase_{locale}", f"qt_{locale}"):
        translator = QTranslator()
        if translator.load(name, folder):
            app.installTranslator(translator)
            _translators.append(translator)


@dataclass(frozen=True)
class _Binding:
    """One value the bars own: where it is read and written, and what a change to it means.

    The window keeps no second copy of these: `read` and `write` are the model's side, `show` is what
    a new value does to the roll and the players, and both directions - remember on screen, apply a
    document - walk this one table, so a bar value is a line here and nowhere else.
    """

    section: str
    name: str
    read: Callable[[], Any]
    write: Callable[[Any], None]
    signal: Any
    show: Callable[[], None] | None = None
    override: str = ""
    remember: Callable[[], bool] | None = None
    model: Callable[[], Any] | None = None  # None: the program's preferences, else this document's


class TempoLoader(LoadingThread):
    """Estimates the tempo of a file off the GUI thread, with the chosen algorithm."""

    loaded = Signal(object)

    def __init__(
        self,
        path: str | Path,
        algorithm: str = bpm.DEFAULT,
        window_seconds: float = 12.0,
        window_hop_seconds: float = 6.0,
        parent=None,
    ):
        super().__init__(path, parent)
        self.algorithm = algorithm
        self.window_seconds = window_seconds
        self.window_hop_seconds = window_hop_seconds

    def load(self) -> None:
        self.loaded.emit(
            bpm.estimate(
                self.path,
                self.algorithm,
                window_seconds=self.window_seconds,
                window_hop_seconds=self.window_hop_seconds,
            )
        )


class SongLoader(LoadingThread):
    """Decodes the file for playback off the GUI thread."""

    loaded = Signal(object, int)

    def load(self) -> None:
        self.loaded.emit(*load_song(self.path))


class SpectrumTempoLoader(LoadingThread):
    """Tracks the beat of the spectrum already on screen, off the GUI thread."""

    loaded = Signal(object)

    def __init__(self, spectrum, window_seconds: float = 8.0, hop_seconds: float = 1.0, parent=None):
        super().__init__("", parent)
        self.spectrum = spectrum
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds

    def load(self) -> None:
        self.loaded.emit(
            bpm.from_spectrum(self.spectrum, window_seconds=self.window_seconds, hop_seconds=self.hop_seconds)
        )


class MainWindow(QMainWindow):
    transport_changed = Signal(bool)  # whether the transport is running, on every change

    def __init__(self, settings=None, overrides: dict | None = None):
        super().__init__()
        self.setWindowTitle("Namioto")
        self.resize(1200, 720)
        self.settings = settings if settings is not None else store.load()
        self.state = window_state.load()
        self.project_settings = project.default_settings(self.state.project)
        i18n.set_language(self.settings.general.language)
        _install_translations(i18n.current())
        self.settings_store = SettingsStore(self.settings, parent=self)
        self.settings_store.changed.connect(self._on_settings_changed)
        self.settings_store.failed.connect(lambda message: self.statusBar().showMessage(message))
        self._theme = theme.apply(theme.running_app(), self.settings.general.style)
        self._startup_style = self.settings.general.style
        self._startup_plugins = self.settings.general.system_qt_plugins
        theme.hints().colorSchemeChanged.connect(self._on_color_scheme)
        self.overrides = dict(overrides or {})  # values this run was asked for, never written back
        self._display_overrides: dict[str, float] = {}
        self._seeding = False
        self._loading = False
        self.project_path: Path | None = None
        self.project_dirty = False
        self.audio_path: str | None = None
        self.loader: SpectrumLoader | None = None
        self.tempo_loader: TempoLoader | None = None
        self._tempo_manual = False
        self.song_loader: SongLoader | None = None
        # the audio a result belongs to: a loader the window replaced may still report its own
        self._audio_generation = 0
        # the overtone estimate of the spectrum on screen, and the parameters it was fitted with
        self._harmonic = None
        self._harmonic_fit: tuple[int, float] | None = None
        self._workers: set[QThread] = set()
        self.ipc: IpcServer | None = None
        self._bridge: WindowBridge | None = None
        self._reported_playing: bool | None = None
        # every menu item is looked up here by name, so a command has one action and one place it
        # is wired; the bars' own buttons stay the state, and the actions that stand for them mirror
        self._actions: dict[str, QAction] = {}
        # the A-B ends the user set, whether or not the loop is armed: `state.loop` holds the span in
        # force, and these keep the ends so switching the loop off and on loses nothing
        self._loop_ends: list[float | None] = [None, None]
        # what the status bar last said about the span, so showing it again does not repeat itself
        self._loop_text_shown = loop_text(None)
        self.mtc_mode = ""  # "", "send" or "receive"
        self.mtc = MtcPort(self)
        self.setAcceptDrops(True)  # a project dropped on the window opens, the way the chooser would

        editor = self.project_settings.editor
        self.view = PianoRollView()
        self.view.set_zoom(editor.zoom_x, editor.zoom_y)
        self.view.initial_center = (self.project_settings.view.center_x, self.project_settings.view.center_y)
        self.view.overtone_highlight = self.settings.editor.overtone_highlight
        self.view.division = editor.division
        self.ruler = TimelineRuler(self.view)
        self.keyboard = PianoKeyboard(self.view)
        self.marker_track = MarkerTrack(self.view)
        self.player, self.player_name = self._make_player()
        self._current_player_key = self._player_key()
        self.player.gain = self.project_settings.playback.midi_volume / 100.0
        self.song = SongPlayer(self)
        self.position_timer = QTimer(self)
        self.position_timer.setInterval(POSITION_INTERVAL_MS)
        self.position_timer.timeout.connect(self._show_position)
        self.speed_timer = QTimer(self)
        self.speed_timer.setSingleShot(True)
        self.speed_timer.setInterval(SPEED_SETTLE_MS)
        self.speed_timer.timeout.connect(self._apply_speed)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(AUTOSAVE_DELAY_MS)
        self.autosave_timer.timeout.connect(self._autosave)

        corner = QWidget()
        corner.setFixedSize(self.keyboard.width(), self.ruler.height())

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(corner, 0, 0)
        layout.addWidget(self.ruler, 0, 1)
        # the bookmarks lane sits between the ruler and the roll: it reads the roll's markers and
        # scroll position, so it is a strip over the same viewport rather than a widget of its own
        layout.addWidget(self.marker_track, 1, 1)
        layout.addWidget(self.keyboard, 2, 0)
        layout.addWidget(self.view, 2, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(2, 1)
        panel = QWidget()
        panel.setLayout(layout)

        self.transport = TransportBar(self)
        self.edit = EditBar(SNAP_CHOICES, self)
        self.mix = MixBar(self)
        self.controls = ControlArea((self.transport, self.edit, self.mix))

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.controls)
        self.channel_panel = ChannelPanel(self.view)
        self.channel_panel.setVisible(False)  # one channel needs no sidebar; the icon opens it
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.channel_panel)
        row.addWidget(panel, 1)
        column.addLayout(row, 1)
        self.setCentralWidget(central)

        document = self.project_settings
        self.edit.snap.setCurrentIndex(max(0, self.edit.snap.findData(document.editor.snap)))
        self.transport.division.setChecked(document.editor.division == "beats")
        self.transport.auto_page.setChecked(self.settings.editor.auto_page)
        self.transport.overtone.setChecked(self.settings.editor.overtone_highlight)
        self.transport.harmonic.setChecked(self.settings.editor.harmonic_removal)
        self.view.snap = self.edit.snap.currentData()
        self.transport.bpm.setValue(document.tempo.bpm)
        self.transport.grid_offset.setValue(document.editor.grid_offset_ms)
        self.view.set_offset(self.transport.grid_offset.value() / 1000.0)
        self.transport.speed.set_value(document.playback.speed)
        self.mix.gain.set_value(document.spectrum.gain)
        self.mix.contrast.set_value(document.spectrum.contrast)
        self.mix.audio_volume.set_value(document.playback.audio_volume)
        self.mix.midi_volume.set_value(document.playback.midi_volume)

        # one table for every bar value: it feeds the roll and is what the program or the document remembers
        self._bindings = self._make_bindings()
        for binding in self._bindings:
            binding.signal.connect(partial(self._binding_changed, binding))
        self.edit.interaction_changed.connect(self.view.apply_interaction)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.transport.detect.clicked.connect(lambda: self._start_tempo(manual=True))
        self.transport.suggestion.applied.connect(self._apply_tempo)
        self.transport.suggestion.dismissed.connect(self.transport.suggestion.hide)
        self.transport.rewind_requested.connect(lambda: self._seek(0.0))
        self.transport.forward_requested.connect(lambda: self._seek(self._duration()))
        self.transport.play_pause_requested.connect(self._toggle_play)
        self.transport.play_from_start_requested.connect(self._play_from_start)
        self.transport.stop_requested.connect(self._stop)
        self.transport.open_requested.connect(self._on_open)
        self.transport.save_requested.connect(self._on_save)
        self.transport.export_midi_requested.connect(self._on_export_midi)
        self.edit.transcribe_requested.connect(self._open_transcription)
        self.edit.align_requested.connect(self._open_align)
        self.edit.lyrics_requested.connect(self._open_lyrics)
        self.transport.note_lyrics.toggled.connect(self.view.set_note_lyrics_visible)
        self.player.finished.connect(self._on_playback_finished)
        self.song.finished.connect(self._on_playback_finished)
        self.song.failed.connect(self._on_song_failed)
        self.view.seek_requested.connect(self._seek)
        self.view.hover_changed.connect(self._on_hover_changed)
        self.view.note_preview.connect(self._on_note_preview)
        self.keyboard.key_preview.connect(self._on_note_preview)
        self.transport.settings_button.clicked.connect(self._open_settings)
        self._sync_midi_volume()
        self.view.gain = self.mix.gain.value()
        self.view.contrast = self.mix.contrast.value()
        self.view.bpm = self.transport.bpm.value()

        self.view.notes_changed.connect(self._update_status)
        self.view.notes_changed.connect(self._mark_dirty)
        self.view.channels_changed.connect(self._on_channels_changed)
        self.view.active_channel_changed.connect(self._on_active_channel_changed)
        self.transport.channels.toggled.connect(self.channel_panel.setVisible)
        self.edit.quantize_requested.connect(self.view.quantize_notes)
        self.transport.bpm.valueChanged.connect(self._mark_dirty)
        self.play_shortcut = QShortcut(QKeySequence("Space"), self)
        self.play_shortcut.activated.connect(self._toggle_play)
        # F1 is the help key the platform already teaches, and the action the Help menu carries is
        # the one thing it triggers: `_mirror`'s rule, one binding per command
        self._shortcuts_action = shortcuts_action(self)
        QShortcut(QKeySequence("F1"), self).activated.connect(self._shortcuts_action.trigger)
        for keys, slot in (
            ("Ctrl+N", self.new_project),
            ("Ctrl+O", self._on_open),
            ("Ctrl+S", self._on_save),
            ("Ctrl+Shift+S", self._on_save_as),
        ):
            QShortcut(QKeySequence(keys), self).activated.connect(slot)
        for keys, slot in (
            ("Ctrl+C", self.view.copy_selection),
            ("Ctrl+V", self.view.paste_notes),
            ("Ctrl+D", self.view.delete_selection),
        ):
            # a widget shortcut, so these mean the notes only while the roll holds the keyboard and a
            # field keeps its own copy, paste and delete
            shortcut = QShortcut(QKeySequence(keys), self.view)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(slot)
        for standard, slot in (
            (QKeySequence.StandardKey.Undo, self.view.undo),
            (QKeySequence.StandardKey.Redo, self.view.redo),
        ):
            # `QKeySequence(standard)` keeps only the first binding, so Redo would lose Ctrl+Shift+Z
            # on X11 and Windows; the whole platform list is bound instead
            shortcut = QShortcut(self)
            shortcut.setKeys(QKeySequence.keyBindings(standard))
            shortcut.activated.connect(slot)
        self.cursor_note = QLabel()
        self.statusBar().addPermanentWidget(self.cursor_note)
        # the commands this batch adds: the audio tone, the sync port and the menus. The A-B loop is not
        # among them: it lives in the Transport menu, where WaveTone keeps it too, rather than adding a
        # second row to the bar. The twenty connects above stay as they are - rewriting them would be
        # a diff with no behaviour behind it, and the bars are still the one place each of those lives.
        self.mix.pitch_changed.connect(self._on_pitch)
        self.mix.eq_changed.connect(self._on_eq)
        self.mtc.position.connect(self._on_mtc_position)  # queued: RtMidi calls back on its own thread
        self.mtc.failed.connect(self._on_mtc_failed)
        self._apply_tone()
        self._build_menus()
        self._restore_loop()  # after the menu, because showing the span also fills its menu item
        self._restore_session()
        # a hint, not a manual: the keys live in the help dialog, and the bar is free for messages
        self.statusBar().showMessage(i18n.tr("Press F1 for the keyboard shortcuts"), INTRO_MESSAGE_MS)
        self._update_status()
        self.view.setFocus()  # the roll holds the keyboard, so the bar opens without a focus ring on its first button
        if self.settings.remote.enabled:
            self.start_ipc()

    def _make_player(self):
        player, name = open_player(self)
        player.delay_ms = self.settings.midi.playback_delay_ms
        return player, name

    def _player_key(self) -> tuple:
        """What a player is built from: a change to any of it means building a new one."""
        return (self.project_settings.analysis.a4,)

    def apply_overrides(self, gain: float | None = None, contrast: float | None = None) -> None:
        """Values a command line asked for: they shape this run, not what is remembered."""
        if gain is not None:
            self._display_overrides["gain"] = gain
        if contrast is not None:
            self._display_overrides["contrast"] = contrast
        self._seeding = True  # the panel is being filled in, not touched
        try:
            if gain is not None:
                self.mix.gain.set_value(gain)
            if contrast is not None:
                self.mix.contrast.set_value(contrast)
        finally:
            self._seeding = False
        self._on_spectrum_parameters()  # the seed was quiet, so show the roll what it landed on

    def start_ipc(self, address: str | None = None) -> ipc.Endpoint | None:
        """Listen for the remote-control interface and say where on the status bar.

        `address` comes from the command line; without one the setting and the environment decide, and
        an editor already listening keeps what it has. A port already taken is reported and leaves
        the editor running without the interface.
        """
        if self.ipc is not None:
            return self.ipc.endpoint
        endpoint = ipc.listen_endpoint(address or self.settings.remote.address or None)
        found = ipc.read_run()
        if found is not None:
            running, pid = found
            if pid != os.getpid() and running.kind == endpoint.kind and running.address == endpoint.address:
                # a unix socket is removed before the bind, so refusing here keeps a live editor's own
                self.statusBar().showMessage(
                    i18n.tr("Remote control is already listening on {address}", address=running.describe())
                )
                return None
        server = IpcServer(endpoint, parent=self)
        self._bridge = WindowBridge(self, server)
        try:
            bound = server.start()
        except OSError as error:
            self.statusBar().showMessage(i18n.tr("Remote control could not start: {error}", error=error))
            self._bridge = None
            return None
        self.ipc = server
        ipc.write_run(bound)
        self.statusBar().showMessage(i18n.tr("Remote control listening on {address}", address=bound.describe()))
        return bound

    def stop_ipc(self) -> None:
        if self.ipc is None:
            return
        if self._bridge is not None:
            self._bridge.close()
        self.ipc.stop()
        ipc.remove_run(os.getpid())
        self.ipc = None
        self._bridge = None

    def reset_document(self, analysis: dict | None = None) -> None:
        """Start an untitled document from the remembered habits and the analysis values given.

        The remote interface uses this where the open dialog would otherwise ask: a fresh document
        points at no file, its notes and lyrics are gone, and the analysis values that came with it
        are the ones the next `load_audio` reads.
        """
        self._loading = True
        try:
            self.project_settings = project.default_settings(self.state.project)
            for name, value in (analysis or {}).items():
                if ("analysis", name) in project.FIELD_SPECS:
                    params.set_value(self.project_settings, "analysis", name, value)
            self.project_path = None
            self.project_dirty = False
            self.autosave_timer.stop()
            self.transport.bpm.setValue(project.FIELD_SPECS[("tempo", "bpm")].default)
            self.transport.grid_offset.setValue(project.FIELD_SPECS[("editor", "grid_offset_ms")].default)
            # the notes go first: `set_channels` gives a channel back to every note it still finds,
            # so setting the channels over the old notes would leave their channels behind
            self.view.clear_notes()
            self.view.set_channels([Channel(channel=0)])
            self.view.set_meter_map(MeterMap.single())
            self.view.set_markers(())
            self.view.set_spectrum(None)
            self.view.undo_stack.clear()
            self._apply_project_settings()
        finally:
            self._loading = False
        self._update_status()

    def _open_lyrics(self) -> None:
        """Edit the words the notes of the active channel sing, one word per note."""
        if self.view.gesture_active:
            return
        notes = [item.note for item in self.view.notes() if item.note.channel == self.view.active_channel]
        dialog = LyricsDialog(words.join(notes), parent=self)
        dialog.saved.connect(partial(self._adopt_lyrics, notes))
        dialog.exec()

    def _adopt_lyrics(self, notes, text: str) -> bool:
        """Put the words on the notes, one per note, as one undo step."""
        if text == words.join(notes):
            return True
        self.view.begin_gesture(i18n.tr("Edit lyrics"))
        try:
            words.distribute(notes, text)
        finally:
            self.view.commit_gesture()
        self.view.refresh()
        self._mark_dirty()
        return True

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, parent=self)
        dialog.applied.connect(self.settings_store.apply)
        dialog.exec()

    def _make_bindings(self) -> tuple[_Binding, ...]:
        """Every bar value the settings hold, one line each: read, write, the signal, and its effect."""
        return (
            _Binding(
                "spectrum",
                "gain",
                self.mix.gain.value,
                self.mix.gain.set_value,
                self.mix.gain.value_changed,
                self._on_spectrum_parameters,
                "gain",
                model=lambda: self.project_settings,
            ),
            _Binding(
                "spectrum",
                "contrast",
                self.mix.contrast.value,
                self.mix.contrast.set_value,
                self.mix.contrast.value_changed,
                self._on_spectrum_parameters,
                "contrast",
                model=lambda: self.project_settings,
            ),
            _Binding(
                "playback",
                "audio_volume",
                self.mix.audio_volume.value,
                self.mix.audio_volume.set_value,
                self.mix.audio_volume.value_changed,
                self._on_audio_volume,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "playback",
                "midi_volume",
                self.mix.midi_volume.value,
                self.mix.midi_volume.set_value,
                self.mix.midi_volume.value_changed,
                self._on_midi_volume,
                remember=lambda: not self.player.silent,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "playback",
                "speed",
                self.transport.speed.value,
                self.transport.speed.set_value,
                self.transport.speed.slider.valueChanged,
                self._on_speed_changed,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "editor",
                "grid_offset_ms",
                self.transport.grid_offset.value,
                self.transport.grid_offset.setValue,
                self.transport.grid_offset.valueChanged,
                self._on_grid_offset_changed,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "tempo",
                "bpm",
                self.transport.bpm.value,
                self.transport.bpm.setValue,
                self.transport.bpm.valueChanged,
                self._on_bpm_changed,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "editor",
                "snap",
                self._snap_value,
                self._set_snap,
                self.edit.snap.currentIndexChanged,
                self._on_snap_changed,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "editor",
                "division",
                self._division_value,
                self._set_division,
                self.transport.division_changed,
                self._on_division_changed,
                model=lambda: self.project_settings,
            ),
            _Binding(
                "editor",
                "auto_page",
                self.transport.auto_page.isChecked,
                self.transport.auto_page.setChecked,
                self.transport.auto_page_toggled,
            ),
            _Binding(
                "editor",
                "overtone_highlight",
                self.transport.overtone.isChecked,
                self.transport.overtone.setChecked,
                self.transport.overtone_toggled,
                self._on_overtone,
            ),
            _Binding(
                "editor",
                "harmonic_removal",
                self.transport.harmonic.isChecked,
                self.transport.harmonic.setChecked,
                self.transport.harmonic_toggled,
                self._on_harmonic_toggle,
            ),
        )

    def _snap_value(self) -> float:
        return self.edit.snap.currentData()

    def _set_snap(self, value: float) -> None:
        self.edit.snap.setCurrentIndex(max(0, self.edit.snap.findData(value)))

    def _division_value(self) -> str:
        return "beats" if self.transport.division.isChecked() else "seconds"

    def _set_division(self, value: str) -> None:
        self.transport.division.setChecked(value == "beats")

    def _binding_changed(self, binding: _Binding, *_args) -> None:
        """A bar value the user moved: it reaches the roll, and it is what the program remembers."""
        if self._seeding:  # the window filled the bar in, so this is not a change to write back
            return
        if binding.show is not None:
            binding.show()
        self._display_overrides.clear()  # the bar was touched after all, so it is what is remembered
        self._remember_configuration()
        if binding.model is not None:
            spec = project.FIELD_SPECS.get((binding.section, binding.name))
            if spec is not None and spec.reuse:  # a habit the user chose, kept for the next document
                project.remember(self.state.project, binding.section, binding.name, binding.read())
        self.settings_store.touch()

    def _apply_theme(self) -> None:
        """Dress the window the way the settings ask, and follow the colours the desktop hands out.

        The colour the roll and the ruler draw with is read while they paint, so a switch is those
        two painting again; the keyboard keeps its fixed colours, and everything else is the
        style's to draw.
        """
        name = theme.apply(theme.running_app(), self.settings.general.style)
        if name == self._theme:
            return
        self._theme = name
        self.view.refresh()
        self.ruler.update()

    def _on_color_scheme(self, _scheme) -> None:
        """The desktop switched between light and dark, so the canvas follows it."""
        self._apply_theme()

    def _owner(self, binding: _Binding):
        """The object a bar reads and writes: the program's preferences, or the open document's."""
        return self.settings if binding.model is None else binding.model()

    def _seed_bindings(self, chosen: Callable[[_Binding], bool]) -> None:
        """Fill the chosen bars in quietly, then let each value show, even when the widget held it.

        `_seeding` keeps a write that is not the user's from being remembered as one.
        """
        self._seeding = True
        try:
            for binding in self._bindings:
                if not chosen(binding):
                    continue
                binding.write(params.get_value(self._owner(binding), binding.section, binding.name))
                if binding.show is not None:
                    binding.show()
        finally:
            self._seeding = False

    def _on_settings_changed(self, settings) -> None:
        """The settings window applied: the program's own preferences take over at once."""
        previous = self.settings
        self.settings = settings
        self.player.delay_ms = settings.midi.playback_delay_ms
        self._seed_bindings(lambda binding: binding.model is None)
        self._apply_tone()  # a pitch or a curve the settings hold reaches the player at once
        self._apply_mtc_settings(previous)  # a rate or a port changed under a running port
        self._apply_theme()
        self._update_status()
        if settings.remote.enabled and self.ipc is None:
            self.start_ipc()  # a switch flipped on reaches a running editor at once
        elif not settings.remote.enabled and self.ipc is not None:
            self.stop_ipc()
        self._advise_restart()

    def _advise_restart(self) -> None:
        """Say what the settings window changed that only the next run can take up."""
        pending = []
        if self.settings.general.system_qt_plugins != self._startup_plugins:
            pending.append(i18n.tr("Qt plugins"))
        if self.settings.general.style != self._startup_style:
            pending.append(i18n.tr("widget style"))
        if pending:
            self.statusBar().showMessage(i18n.tr("Restart namioto to apply: {what}", what=", ".join(pending)))

    def _apply_project_settings(self) -> None:
        """A document was opened or started: its values take over the bars, the roll and the player."""
        self._seed_bindings(lambda binding: binding.model is not None)
        document = self.project_settings
        self.view.set_zoom(document.editor.zoom_x, document.editor.zoom_y)
        self.view.center_on(document.view.center_x, document.view.center_y)
        self.view.refresh()
        if self._player_key() != self._current_player_key:
            self._rebuild_player()
        self._sync_midi_volume()

    def _rebuild_player(self) -> None:
        """A player that has to be built again - for another tuning - takes the notes over."""
        playing = self.player.is_playing
        position = self._position()
        self.player.stop()
        self.player, self.player_name = self._make_player()
        self._current_player_key = self._player_key()
        self.player.gain = self.mix.midi_volume.value() / 100.0
        self.player.finished.connect(self._on_playback_finished)
        self._sync_midi_volume()
        self._send_program()
        if playing:
            self.player.play(position)

    def _sync_midi_volume(self) -> None:
        """The MIDI slider follows the note output: no MIDI service means zero and out of reach.

        Filling the bar in is not a change the user made, so it must not write the zero back.
        """
        silent = self.player.silent
        seeding, self._seeding = self._seeding, True
        try:
            self.mix.midi_volume.setEnabled(not silent)
            if silent:
                self.mix.midi_volume.set_value(0)
                self.mix.midi_volume.slider.setToolTip(i18n.tr("No MIDI output is available on this machine"))
            else:
                self.mix.midi_volume.slider.setToolTip(
                    i18n.tr("Volume of the note playback through {player}", player=self.player_name)
                )
        finally:
            self._seeding = seeding

    def _program(self) -> tuple[tuple, tuple]:
        """The notes of every channel that sounds, and each sounding channel's instrument, volume."""
        audible = set(audible_channels(self.view.channels))
        notes = tuple(
            (
                note.pitch,
                self.view.to_seconds(note.start),
                self.view.length_to_seconds(note.duration, note.start),
                note.channel,
            )
            for note in self.view.notes()
            if note.channel in audible
        )
        programs = tuple(
            (channel.channel, channel.program, channel.volume)
            for channel in self.view.channels
            if channel.channel in audible
        )
        return notes, programs

    def _send_program(self) -> None:
        """Hand the current notes and their channels to the player, as playback speed leaves them."""
        notes, programs = self._program()
        self.player.delay_ms = self.settings.midi.playback_delay_ms
        self.player.set_program(notes, self.transport.speed.value(), programs)

    def _remember_configuration(self) -> None:
        """The bar values are the model's, so what is on screen is what comes back next time.

        A bar the program filled in rather than the user - the silent MIDI volume - is not on
        screen either, so its binding is left alone.
        """
        for binding in self._bindings:
            if binding.remember is not None and not binding.remember():
                continue
            if binding.override and binding.override in self._display_overrides:
                continue
            params.set_value(self._owner(binding), binding.section, binding.name, binding.read())
        params.set_value(self.project_settings, "editor", "zoom_x", self.view.zoom[0])
        params.set_value(self.project_settings, "editor", "zoom_y", self.view.zoom[1])

    def _restore_session(self) -> None:
        if self.state.geometry:
            self.restoreGeometry(QByteArray.fromBase64(self.state.geometry.encode()))

    def _remember_session(self) -> None:
        self.state.geometry = self.saveGeometry().toBase64().data().decode()
        centre = self.view.mapToScene(self.view.viewport().rect().center())
        self.project_settings.view.center_x = round(centre.x(), 1)
        self.project_settings.view.center_y = round(centre.y(), 1)

    def _on_zoom_changed(self) -> None:
        """A zoom the user asked for is a habit: the next document starts where this one was left."""
        zoom_x, zoom_y = self.view.zoom
        project.remember(self.state.project, "editor", "zoom_x", zoom_x)
        project.remember(self.state.project, "editor", "zoom_y", zoom_y)

    def _save_state(self) -> None:
        try:
            window_state.save(self.state)
        except OSError as error:  # a read-only home must not take the editor down
            self.statusBar().showMessage(i18n.tr("Window state could not be saved: {error}", error=error))

    def _document_name(self) -> str:
        name = self.project_path.stem if self.project_path is not None else i18n.tr("Untitled")
        return f"{name}*" if self.project_dirty else name

    def _mark_dirty(self, *_args) -> None:
        """What a save would otherwise lose: the notes, the tempo and the audio. The view and the
        listening values are written with a project, but do not mark it as changed."""
        if self._loading:
            return
        if not self.project_dirty:
            self.project_dirty = True
            self._update_status()
        if self.settings.general.auto_save:
            self.autosave_timer.start()

    def _autosave(self) -> None:
        """Write the open project out once editing stops, and when the window loses focus.

        Only a document that already has a file is written: a sketch with no name waits for Save As.
        A modal dialog is asking something of its own, so the window losing focus to it is not a
        reason to write - the save-on-close prompt would otherwise answer itself.
        """
        if not self.settings.general.auto_save or self._loading:
            return
        if not self.project_dirty or self.project_path is None:
            return
        if QApplication.activeModalWidget() is not None:
            return
        self.save_project(self.project_path)

    def _start_directory(self) -> str:
        """Where a file dialog opens: the folder of the project in use, else the last one opened."""
        if self.project_path is not None:
            return str(self.project_path.parent)
        return self.state.last_audio_dir or str(Path.home())

    def _confirm_discard(self) -> bool:
        """Ask before unsaved notes go; False means the caller should do nothing. A roll that has no
        file yet is a sketch, so it is not worth interrupting anyone over."""
        if self.project_path is None or not self.project_dirty:
            return True
        choice = QMessageBox.warning(
            self,
            "Namioto",
            i18n.tr("Save the changes to {name}?", name=self.project_path.name),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            return self._on_save()
        return True

    def _on_open(self) -> None:
        filters = [i18n.tr("All files (*)"), _project_filter(), _audio_filter()]
        if self.project_path is not None:  # a MIDI is a project's, so it is only worth offering inside one
            filters.append(_midi_filter())
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            i18n.tr("Open"),
            self._start_directory(),
            ";;".join(filters),
        )
        if chosen:
            self.open_file(chosen)

    def open_file(self, path: str | Path) -> bool:
        """Open a file by what it is - a project, a MIDI file to import, or audio to analyse.

        The command line and the Open dialog both come through here, so a file behaves the same
        whichever way it arrives. A project and audio are a document each, so switching to one asks
        about unsaved work first; a MIDI is only a part that goes into the project already open.
        """
        if project.looks_like_project(path):
            return self._confirm_discard() and self.load_project(path)
        if midi.looks_like_midi(path):
            return self.import_midi(path)
        return self._confirm_discard() and self.open_audio(str(path))

    def open_audio(self, path: str) -> bool:
        """Open audio as part of a project: the sibling project if there is one, else a new one.

        A new one is asked about first - where its project is saved, and how the audio is analysed -
        because the notes and the analysis all belong to that file; a cancelled window leaves the
        audio untouched, and only the values the user changed carry on to the next document.
        """
        beside = Path(path).with_suffix(project.SUFFIX)
        if beside.exists():
            return self.load_project(beside)
        document = self._seed_analysis()
        dialog = OpenAudioDialog(path, document, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        chosen = dialog.document()
        if project.remember_changes(self.state.project, document, chosen):
            self._save_state()
        self._consume_analysis()
        self.reset_document()
        self.project_settings = chosen
        self._apply_project_settings()
        self.load_audio(path)
        return self.save_project(dialog.target())

    def _seed_analysis(self) -> project.ProjectSettings:
        """What a new document starts from, with the analysis values the command line asked for."""
        document = project.default_settings(self.state.project)
        for name, value in self.overrides.items():
            if value is not None and ("analysis", name) in project.FIELD_SPECS:
                params.set_value(document, "analysis", name, value)
        return document

    def _consume_analysis(self) -> None:
        """The dialog took the command line's analysis values over: they are the document's now."""
        for name in tuple(self.overrides):
            if ("analysis", name) in project.FIELD_SPECS:
                del self.overrides[name]

    def _on_save(self) -> bool:
        if self.project_path is None:
            return self._on_save_as()
        return self.save_project(self.project_path)

    def _on_save_as(self) -> bool:
        name = Path(self.audio_path).stem if self.audio_path is not None else "untitled"
        suggested = Path(self._start_directory()) / f"{name}{project.SUFFIX}"
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            i18n.tr("Save project"),
            str(self.project_path or suggested),
            _project_filter(),
        )
        if not chosen:
            return False
        target = Path(chosen)
        if not project.looks_like_project(target):
            target = target.with_name(target.name + project.SUFFIX)
        return self.save_project(target)

    def _on_export_midi(self) -> bool:
        """Write the roll out as a MIDI file, a file of its own and never the project's name."""
        name = Path(self.audio_path).stem if self.audio_path is not None else "untitled"
        suggested = Path(self._start_directory()) / f"{name}{midi.SUFFIXES[0]}"
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            i18n.tr("Export MIDI"),
            str(suggested),
            _midi_filter(),
        )
        if not chosen:
            return False
        target = Path(chosen)
        if not midi.looks_like_midi(target):
            target = target.with_name(target.name + midi.SUFFIXES[0])
        return self.export_midi(target)

    def new_project(self) -> bool:
        """Start an empty project: no audio, no notes, the default values.

        Audio is not what makes a project: a MIDI file can be imported into it, notes can be drawn
        on the default grid, and the spectrum simply stays empty until a recording is opened on top.
        """
        if not self._confirm_discard():
            return False
        self._loading = True
        try:
            self.project_settings = project.default_settings()
            self.project_path = None
            self.project_dirty = False
            self.autosave_timer.stop()
            self._clear_audio()
            self.view.set_tempo(TempoMap.single(self.project_settings.tempo.bpm), keep_notes=False)
            self.view.set_meter_map(MeterMap.single())
            self.view.set_markers(())
            self.view.set_channels((Channel(channel=0),))
            self.view.clear_notes()
            self._apply_project_settings()
            self.view.undo_stack.clear()
        finally:
            self._loading = False
        self._update_status()
        self.statusBar().showMessage(i18n.tr("New project"))
        return True

    def load_project(self, path: str | Path) -> bool:
        """Open a project: its values come over the running ones, and its notes replace the roll."""
        try:
            opened = project.load(path)
        except (OSError, ValueError) as error:
            self.statusBar().showMessage(i18n.tr("Project could not be opened: {error}", error=error))
            return False
        self._loading = True
        try:
            self.project_settings = opened.settings
            self.project_path = Path(path)
            self.project_dirty = False
            self.autosave_timer.stop()
            # the map comes before the notes: their beats are read out of it
            self.view.set_tempo(opened.tempo_map, keep_notes=False)
            self.view.set_meter_map(opened.meter_map)
            self.view.set_markers(opened.markers)
            missing = self._open_audio(project.resolve_audio(self.project_path, opened.audio))
            self.view.set_channels(opened.channels)
            loaded = (
                (
                    note.pitch,
                    self.view.to_beats(note.start),
                    self.view.to_beats(note.start + note.duration) - self.view.to_beats(note.start),
                    note.channel,
                    note.id,
                    note.lyric,
                )
                for note in opened.notes
            )
            self.view.set_notes(loaded, opened.next_id)
            self._apply_project_settings()  # the bars, the zoom and the player take the document's values
            self.view.undo_stack.clear()  # another document starts its history over
        finally:
            self._loading = False
        self._update_status()
        self.statusBar().showMessage(
            i18n.tr("Opened {name} — {notes} notes", name=self.project_path.name, notes=len(opened.notes)) + missing
        )
        return True

    def save_project(self, path: str | Path) -> bool:
        """Write the notes and the values that belong to them out to `path`."""
        self._remember_configuration()
        self._remember_session()
        target = Path(path)
        # scene order is not a file's order: sorted notes keep a saved project stable to diff
        notes = tuple(
            sorted(
                project.Note(
                    self.view.to_seconds(note.start),
                    self.view.length_to_seconds(note.duration, note.start),
                    note.pitch,
                    note.channel,
                    note.id,
                )
                for note in self.view.notes()
            )
        )
        payload = project.Project(
            settings=self.project_settings,
            audio=project.store_audio(target, self.audio_path),
            channels=tuple(self.view.channels),
            notes=notes,
            next_id=self.view.document.next_id,
            tempo_map=self.view.tempo,
            meter_map=self.view.meter_map,
            markers=self.view.markers,
        )
        try:
            project.save(payload, target)
        except OSError as error:
            self.statusBar().showMessage(i18n.tr("Project could not be saved: {error}", error=error))
            return False
        self.project_path = target
        self.project_dirty = False
        self.autosave_timer.stop()
        self.state.last_audio_dir = str(target.parent)
        self._save_state()
        self._update_status()
        self.statusBar().showMessage(i18n.tr("Saved {name} — {notes} notes", name=target.name, notes=len(notes)))
        return True

    def import_midi(self, path: str | Path) -> bool:
        """Put the notes and channels of a MIDI file into the running session, audio and view and all.

        Importing a MIDI over analyzed audio is the way a transcription made elsewhere is checked
        against the sound it came from, so nothing here is cleared away but the notes - and when
        there are notes to lose, the dialog says so and offers to merge into the channels instead.
        """
        if self.project_path is None:
            self.statusBar().showMessage(
                i18n.tr("Open a song first: a MIDI file is imported into a project, not on its own")
            )
            return False
        try:
            imported = midi.read(path, wavetone=self.settings.midi.wavetone)
        except (OSError, EOFError, ValueError) as error:
            self.statusBar().showMessage(i18n.tr("MIDI file could not be read: {error}", error=error))
            return False
        # the file's notes land on the roll's channels, keeping the instrument and volume the file
        # gave them; a name is not among them, because a MIDI channel cannot carry one
        origin = Path(path).name
        mode, mapping = "replace", ()
        if self.view.notes():
            occupied = {note.channel for note in self.view.notes()}
            dialog = MidiImportDialog(imported, self.view.channels, origin, occupied, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
            mode, mapping = dialog.mode(), tuple(dialog.mapping())
        self.apply_midi(imported, mode, mapping)
        verb = i18n.tr("Merged") if mode == "merge" else i18n.tr("Imported")
        message = [i18n.tr("{verb} {notes} notes from {origin}", verb=verb, notes=len(imported.notes), origin=origin)]
        if imported.tempo_changes:
            message.append(
                i18n.tr("{notes} tempo changes", notes=imported.tempo_changes)
                if mode == "replace"
                else i18n.tr("{notes} tempo changes; the grid keeps its own", notes=imported.tempo_changes)
            )
        if imported.dropped:
            message.append(i18n.tr("{notes} note events left out", notes=imported.dropped))
        if imported.left_out:
            numbers = " ".join(str(channel) for channel in imported.left_out)
            message.append(i18n.tr("channels {numbers} left out", numbers=numbers))
        self.statusBar().showMessage(" — ".join(message))
        return True

    def apply_midi(self, imported, mode: str = "replace", mapping=()) -> None:
        """Put an imported MIDI into the running session, as the import dialog decided."""
        self._loading = True
        try:
            if mode == "merge":
                self._merge_midi(imported, mapping)
            else:
                # the file's own tempo map becomes the grid's, before the notes are placed on it
                self.view.set_tempo(imported.tempo_map, keep_notes=False)
                self.transport.bpm.setValue(imported.tempo_map.bpm)
                self.view.replace(
                    imported.channels,
                    [
                        (
                            note.pitch,
                            self.view.to_beats(note.start),
                            self.view.to_beats(note.start + note.duration) - self.view.to_beats(note.start),
                            note.channel,
                        )
                        for note in imported.notes
                    ],
                    "Import MIDI",
                )
        finally:
            self._loading = False
        self._mark_dirty()

    def _merge_midi(self, imported, mapping) -> None:
        """Add the file's notes to the roll, on the channels the dialog pointed them at.

        The tempo stays where it is: both sets of notes are timed against the same audio, and moving
        the grid under them would take the ones already there off it. A channel that already carries
        notes is merged into and keeps what it is; an empty one the file walks into takes the file's
        instrument and volume, exactly as a new one would.
        """
        channels = {channel.channel: channel for channel in self.view.channels}
        carried = {note.channel for note in self.view.notes()}
        landing: dict[int, int] = {}
        for source, wanted in enumerate(imported.channels):
            target = mapping[source]
            if target >= 0 and target in carried:
                landing[wanted.channel] = target
                continue
            place = target
            if place < 0:
                free = free_channel(channels.values())
                place = wanted.channel if wanted.channel not in channels else free
                if place is None:
                    place = wanted.channel  # sixteen channels are full; the file shares its own
            channels[place] = channel_set_field(
                channels.get(place, Channel(channel=place)), program=wanted.program, volume=wanted.volume
            )
            landing[wanted.channel] = place
        kept = [(note.pitch, note.start, note.duration, note.channel, note.id) for note in self.view.notes()]
        arriving = [
            (
                note.pitch,
                self.view.to_beats(note.start),
                self.view.to_beats(note.start + note.duration) - self.view.to_beats(note.start),
                landing[note.channel],
            )
            for note in imported.notes
        ]
        self.view.replace(channels.values(), kept + arriving, "Merge MIDI")

    def _midi_export(self) -> dict:
        notes = tuple(
            project.Note(
                self.view.to_seconds(note.start),
                self.view.length_to_seconds(note.duration, note.start),
                note.pitch,
                note.channel,
            )
            for note in self.view.notes()
        )
        return {
            "channels": tuple(self.view.channels),
            "notes": notes,
            "tempo": self.view.tempo,
            "wavetone": self.settings.midi.wavetone,
        }

    def export_midi(self, path: str | Path) -> bool:
        """Write every channel out as MIDI, on the project's own tempo."""
        exported = self._midi_export()
        try:
            midi.write(path, **exported)
        except OSError as error:
            self.statusBar().showMessage(i18n.tr("MIDI file could not be written: {error}", error=error))
            return False
        self.statusBar().showMessage(
            i18n.tr("Exported {name} — {notes} notes", name=Path(path).name, notes=len(exported["notes"]))
        )
        return True

    def _open_transcription(self) -> None:
        """Ask the chosen engine for notes, over the audio the session is already listening to."""
        if self.audio_path is None:
            return
        occupied = any(note.channel == 0 for note in self.view.notes())
        dialog = TranscriptionDialog(
            self.audio_path,
            self.transport.bpm.value(),
            self,
            channel_has_notes=occupied,
            offset=self.view.offset,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.notes():
                self.transport.grid_offset.setValue(dialog.offset * 1000.0)
            if dialog.parts():
                self._adopt_parts(dialog.parts())
            else:
                self._adopt_transcription(dialog.notes())

    def _open_align(self) -> None:
        """Ask the aligner for a time on every note, from the word it sings."""
        if self.view.gesture_active:
            return
        if self.audio_path is None:
            self.statusBar().showMessage(i18n.tr("Load the audio before aligning the lyrics"))
            return
        notes = [
            item.note
            for item in self.view.notes()
            if item.note.channel == self.view.active_channel and item.note.lyric
        ]
        if not notes:
            self.statusBar().showMessage(i18n.tr("The selected channel has no lyrics to align"))
            return
        dialog = AlignDialog(
            self.audio_path,
            [note.lyric for note in notes],
            [self.view.to_seconds(note.start) for note in notes],
            self.transport.bpm.value(),
            self,
            offset=self.view.offset,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._adopt_alignment(notes, dialog.spans())

    def _adopt_alignment(self, notes, spans) -> None:
        """Lay the aligned spans onto the notes, as one undo step."""
        timed = [(note, span) for note, span in zip(notes, spans, strict=True) if span is not None]
        if not timed:
            self.statusBar().showMessage(i18n.tr("The aligner put a time on no note"))
            return
        self.view.begin_gesture(i18n.tr("Align lyrics"))
        try:
            for note, (start, end) in timed:
                note.set_range(self.view.to_beats(start), note.pitch)
                note.set_duration(self.view.to_beats(end) - self.view.to_beats(start))
        finally:
            self.view.commit_gesture()
        self.view.refresh()
        self._mark_dirty()
        self.statusBar().showMessage(i18n.tr("Aligned {notes} notes", notes=len(timed)))

    def _adopt_transcription(self, notes) -> None:
        """Replace channel 0's notes with the transcription."""
        if not notes:
            self.statusBar().showMessage(i18n.tr("The transcription found no notes in the loaded audio"))
            return
        arriving = [
            (round(pitch), self.view.to_beats(max(0.0, onset)), self.view.to_beats(max(0.0, offset - onset)))
            for onset, offset, pitch in notes
        ]
        channels = list(self.view.channels)
        if not any(channel.channel == 0 for channel in channels):
            channels.append(Channel(name="GAME", channel=0))
        kept = [
            (note.pitch, note.start, note.duration, note.channel, note.id)
            for note in self.view.notes()
            if note.channel != 0
        ]
        self.view.replace(channels, kept + [(*note, 0) for note in arriving], "Transcribe")
        self.statusBar().showMessage(
            i18n.tr("Transcription found {notes} notes on channel {channel}", notes=len(arriving), channel=0)
        )

    def _adopt_parts(self, parts) -> None:
        """Give each instrument its own channel, starting at the first free one.

        A part that finds no free channel is left out and said so: sixteen channels is the MIDI
        limit, and folding a leftover part into one that is taken would play the wrong instrument.
        """
        from namioto.channels import free_channel

        channels = list(self.view.channels)
        notes = [(note.pitch, note.start, note.duration, note.channel, note.id) for note in self.view.notes()]
        placed = 0
        skipped = 0
        for name, program, found in parts:
            number = free_channel(channels)
            if number is None:
                skipped += 1
                continue
            channels.append(Channel(name=name.replace("_", " "), channel=number, program=int(program)))
            notes.extend(
                (
                    round(pitch),
                    self.view.to_beats(max(0.0, onset)),
                    self.view.to_beats(max(0.0, offset - onset)),
                    number,
                )
                for onset, offset, pitch in found
            )
            placed += 1
        if placed == 0:
            self.statusBar().showMessage(i18n.tr("No free channel for the transcription"))
            return
        self.view.replace(channels, notes, "Transcribe")
        self.statusBar().showMessage(
            i18n.tr(
                "Transcription found {parts} parts{skipped}",
                parts=placed,
                skipped=i18n.tr(", {count} left out", count=skipped) if skipped else "",
            )
        )

    def _target_notes(self, channel: int = 0) -> list[tuple[float, float, int, int]]:
        """One channel's notes in seconds, ordered by time then pitch then stable id."""
        return list(self._notes_by_channel((channel,))[channel])

    def _notes_by_channel(self, channels) -> dict:
        found = {channel: [] for channel in channels}
        for note in self.view.notes():
            if note.channel in found:
                found[note.channel].append(
                    (self.view.to_seconds(note.start), self.view.to_seconds(note.end), note.pitch, note.id)
                )
        return {channel: tuple(sorted(notes)) for channel, notes in found.items()}

    def _open_audio(self, target: Path | None) -> str:
        """Load the audio a project names, or say why there is none: its notes are worth having either way."""
        if target is not None and target.exists():
            self.load_audio(str(target))
            return ""
        self._clear_audio()
        return i18n.tr(" (audio not found: {path})", path=target) if target is not None else ""

    def _clear_audio(self) -> None:
        """Forget the analysed file, for a project that names one this machine does not have."""
        self._audio_generation += 1
        self.position_timer.stop()
        self.audio_path = None
        self._harmonic = None
        self._harmonic_fit = None
        self.view.set_spectrum(None)
        self.song.unload()
        self.player.stop()
        self.transport.detect.setEnabled(False)
        self.edit.transcribe.setEnabled(False)
        self.edit.align.setEnabled(False)
        self.transport.suggestion.hide()
        self._show_position()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self._autosave()

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        # every timer this window owns must be disarmed here: a closed window is destroyed by the
        # cyclic collector sooner or later, and a collector run that lands inside Qt's timer dispatch
        # would free a timer that dispatch is still walking
        self.autosave_timer.stop()
        self.position_timer.stop()
        self.speed_timer.stop()
        for worker in tuple(self._workers):
            worker.wait()
        self._workers.clear()
        self._remember_configuration()
        self._remember_session()
        self.settings_store.flush()
        self._save_state()
        self.stop_ipc()
        self.mtc.stop()  # a port left open would outlive the window that owns it
        self.player.stop()  # a preview waiting on the delay would outlive the window that owns its timer
        self.song.close()  # the engine owns the audio device, so it leaves before the window does
        super().closeEvent(event)

    def _analysis_options(self) -> dict:
        """What the analysis runs with: the document's values, then whatever this run was told to use."""
        analysis = self.project_settings.analysis
        chosen = {
            "channels": analysis.channels,
            "t_num": analysis.t_num,
            "fft_points": analysis.fft_points,
            "a4": analysis.a4,
        }
        for name, value in self.overrides.items():
            if value is not None and name in chosen:
                chosen[name] = value
        return chosen

    def _start_loader(self, attribute: str, worker: QThread) -> None:
        """Start a background loader and keep it alive until it finishes.

        A loader the window has replaced still runs to the end, so the window holds every one of
        them: a `QThread` collected mid-flight takes its callbacks down with it. `_release_loader`
        drops it once it reports, and `closeEvent` waits for whatever is left.
        """
        setattr(self, attribute, worker)
        self._workers.add(worker)
        worker.finished.connect(partial(self._release_loader, attribute, worker))
        worker.start()

    def _release_loader(self, attribute: str, worker: QThread) -> None:
        self._workers.discard(worker)
        if getattr(self, attribute, None) is worker:
            setattr(self, attribute, None)
        worker.deleteLater()

    def load_audio(self, path: str) -> None:
        if not self._loading and str(path) != self.audio_path:
            self._harmonic = None
            self._harmonic_fit = None
            # the old song's spectrum must not sit under the new one while it is analysed
            self.view.set_spectrum(None)
            # another song brings its own tempo and offset; a re-analysis of the same one does not
            self.transport.bpm.setValue(project.FIELD_SPECS[("tempo", "bpm")].default)
            self.transport.grid_offset.setValue(project.FIELD_SPECS[("editor", "grid_offset_ms")].default)
        self.audio_path = path
        self._audio_generation += 1
        generation = self._audio_generation
        self.edit.transcribe.setEnabled(True)
        self.edit.align.setEnabled(True)
        self._mark_dirty()
        self.state.last_audio_dir = str(Path(path).parent)
        self._save_state()
        loader = SpectrumLoader(path, parent=self, **self._analysis_options())
        loader.progress.connect(partial(self._on_analysis_progress, generation))
        loader.loaded.connect(partial(self._on_spectrum_loaded, generation))
        loader.failed.connect(partial(self._on_spectrum_failed, generation))
        self.statusBar().showMessage(i18n.tr("Analysing {path} …", path=path))
        self._start_loader("loader", loader)
        song_loader = SongLoader(path, parent=self)
        song_loader.loaded.connect(partial(self._on_song_loaded, generation))
        song_loader.failed.connect(partial(self._on_playback_failed, generation))
        self._start_loader("song_loader", song_loader)
        self._start_tempo()

    def _on_song_loaded(self, generation: int, samples, sample_rate: int) -> None:
        if generation != self._audio_generation:
            return
        self.song.load(samples, sample_rate)
        self.song.gain = self.mix.audio_volume.value() / 100.0
        self.song.speed = self.transport.speed.value()

    def _on_spectrum_failed(self, generation: int, message: str) -> None:
        if generation != self._audio_generation:
            return
        self.statusBar().showMessage(i18n.tr("Spectrum failed: {error}", error=message))

    def _on_playback_failed(self, generation: int, message: str) -> None:
        if generation != self._audio_generation:
            return
        self.statusBar().showMessage(i18n.tr("Playback failed: {error}", error=message))

    def _on_song_failed(self, message: str) -> None:
        """The playback process has no generation of its own: a failure from it is always current."""
        self.statusBar().showMessage(i18n.tr("Playback failed: {error}", error=message))

    def _on_speed_changed(self, _value: int = 0) -> None:
        # the song retunes in place as the slider travels; the notes follow once it settles
        self.song.speed = self.transport.speed.value()
        self.speed_timer.start()

    def _apply_speed(self) -> None:
        """Take the settled speed for the note layer; the song already follows the slider."""
        self._set_note_speed()

    def _set_note_speed(self) -> None:
        """Hand the notes over at the settled speed, carrying on from where they are playing."""
        if not self.view.notes():
            return
        playing = self.player.is_playing
        position = self._position()
        self._send_program()
        if playing:
            self.player.play(position)

    def _on_channels_changed(self, *_args) -> None:
        """A channel edit is a document change, and a mute or instrument lands in the running sound."""
        self._mark_dirty()
        if self._is_playing():
            self._set_note_speed()

    def _on_active_channel_changed(self, number: int) -> None:
        channel = next(channel for channel in self.view.channels if channel.channel == number)
        self.statusBar().showMessage(i18n.tr("Drawing into {channel}", channel=channel.label), 2000)

    def _start_tempo(self, manual: bool = False) -> None:
        """Estimate the tempo of the loaded audio in the background, as a suggestion only.

        A run the user asked for offers its result whatever the BPM field holds; one that follows an
        audio load stays quiet over a tempo of the user's own.
        """
        if self.audio_path is None:
            return
        generation = self._audio_generation
        self._tempo_manual = manual
        self.transport.suggestion.hide()
        self.transport.detect.setEnabled(False)
        if self.view.spectrum is not None:
            # the spectrum is already on screen, so its onset envelope costs nothing to reuse
            tempo_loader = SpectrumTempoLoader(self.view.spectrum.spectrum, parent=self)
        else:
            tempo = self.settings.tempo
            tempo_loader = TempoLoader(
                self.audio_path,
                algorithm=tempo.estimator,
                window_seconds=tempo.window_seconds,
                window_hop_seconds=tempo.window_hop_seconds,
                parent=self,
            )
        tempo_loader.loaded.connect(partial(self._on_tempo_loaded, generation))
        tempo_loader.failed.connect(partial(self._on_tempo_failed, generation))
        self._start_loader("tempo_loader", tempo_loader)

    def _on_tempo_loaded(self, generation: int, result) -> None:
        """Offer what was estimated as a candidate, unless the field already holds that tempo."""
        if generation != self._audio_generation:
            return
        self.transport.detect.setEnabled(True)
        manual, self._tempo_manual = self._tempo_manual, False
        if not result.bpm:
            return
        if not manual and self.transport.bpm.value() != project.FIELD_SPECS[("tempo", "bpm")].default:
            return  # a tempo the user set, or took from an estimate, is not one to suggest over
        if round(result.bpm) == round(self.transport.bpm.value()):
            return  # the balloon would read what the field already says
        self.transport.suggestion.estimate(result.bpm, result.agreement, result.windows, result.source, result.residual)
        self.transport.suggestion.show_under(self.transport.bpm)

    def _on_tempo_failed(self, generation: int, message: str) -> None:
        if generation != self._audio_generation:
            return
        self.transport.detect.setEnabled(self.audio_path is not None)
        self.statusBar().showMessage(i18n.tr("Tempo estimation failed: {error}", error=message))

    def _apply_tempo(self, bpm: float) -> None:
        self.transport.suggestion.hide()
        self.transport.bpm.setValue(bpm)
        self.statusBar().showMessage(i18n.tr("Tempo set to {bpm} BPM from the audio", bpm=f"{bpm:.0f}"))

    def _play(self) -> None:
        """Send the notes to the synth and start the audio file, both from where the cursor sits."""
        notes, _channels = self._program()
        if not notes and not self.song.is_loaded:
            self.statusBar().showMessage(i18n.tr("Nothing to play: load a file or draw some notes"))
            return
        seconds = self._position()
        if seconds >= self._duration() - 1e-3:
            seconds = 0.0
        speed = self.transport.speed.value()
        if self.song.is_loaded:
            self.song.speed = speed
            self.song.play(seconds)
        self._send_program()
        self.player.play(seconds)
        if self.mtc_mode == "send":
            self.mtc.seek_to(seconds)
        self._show_position()
        if self._is_playing():
            self.position_timer.start()
        elif self.player.silent:
            self.statusBar().showMessage(i18n.tr("No MIDI output is available on this machine"))
        else:
            self.statusBar().showMessage(i18n.tr("{player} did not accept the notes", player=self.player_name))

    def _toggle_play(self) -> None:
        """One button for both, so it asks the players what they are doing right now."""
        if self._is_playing():
            self._pause()
        else:
            self._play()

    def _play_from_start(self) -> None:
        self._seek(0.0)
        self._play()

    def _pause(self) -> None:
        self.position_timer.stop()
        self.song.pause()
        self.player.pause()
        if self.mtc_mode == "send":
            self.mtc.seek_to(self._position())
        self._show_position()

    def _stop(self) -> None:
        self.position_timer.stop()
        self.song.stop()
        self.player.stop()
        if self.mtc_mode == "send":
            self.mtc.seek_to(self._position())
        self._show_position()

    def _seek(self, seconds: float) -> None:
        self.song.seek(seconds)
        self.player.seek(seconds)
        if self.mtc_mode == "send":
            self.mtc.seek_to(seconds)  # the code follows the transport, wherever it was put
        self._show_position()

    def _position(self) -> float:
        """Where the transport sits: the audio file leads when one is loaded, else the notes."""
        return self.song.position if self.song.is_loaded else self.player.position

    def _duration(self) -> float:
        return self.song.duration if self.song.is_loaded else self.player.duration

    def _is_playing(self) -> bool:
        return self.song.is_playing or self.player.is_playing

    def _show_position(self) -> None:
        seconds = self._position()
        playing = self._is_playing()
        if playing and self._on_loop_tick(seconds):
            return  # the span was reached and `_seek` already showed where the loop sent it
        self.transport.set_position(seconds)
        self.transport.set_playing(playing)
        if playing != self._reported_playing:
            self._reported_playing = playing
            self.transport_changed.emit(playing)
        self.view.playing = playing
        self.view.set_playhead(seconds)
        if playing and self.transport.auto_page.isChecked():
            self.view.follow_playhead(seconds)

    def _on_playback_finished(self) -> None:
        if self._is_playing():  # the other layer is still running
            return
        self.position_timer.stop()
        self._show_position()

    def _on_midi_volume(self, *_args) -> None:
        self.player.gain = self.mix.midi_volume.value() / 100.0

    def _on_audio_volume(self, *_args) -> None:
        self.song.gain = self.mix.audio_volume.value() / 100.0

    def _on_note_preview(self, pitch: int) -> None:
        """Audition a note the user clicked or drew."""
        self.player.preview(pitch)

    def _on_hover_changed(self, pitch: int | None) -> None:
        if pitch is None:
            self.cursor_note.clear()
            return
        self.cursor_note.setText(f"{note_name(pitch)}   {note_frequency(pitch):.2f} Hz")

    def _on_division_changed(self, *_args) -> None:
        self.view.division = self._division_value()
        self.view.refresh()

    def _on_overtone(self, *_args) -> None:
        self.view.overtone_highlight = self.transport.overtone.isChecked()
        self.view.refresh()

    def _on_snap_changed(self, *_args) -> None:
        self.view.snap = self._snap_value()

    def _on_bpm_changed(self, *_args) -> None:
        self.transport.suggestion.hide()  # a tempo the user typed wins over the suggestion
        self.view.bpm = self.transport.bpm.value()

    def _on_grid_offset_changed(self, *_args) -> None:
        """The offset only slides the drawn grid; the playhead and the notes keep their timestamps."""
        self.view.set_offset(self.transport.grid_offset.value() / 1000.0)

    def _harmonic_strength(self) -> float:
        """How much of the overtone estimate the drawing takes out: none while the switch is off."""
        return harmonic.STRENGTH if self.settings.editor.harmonic_removal else 0.0

    def _on_harmonic_toggle(self, *_args) -> None:
        self._on_spectrum_parameters()

    def _on_spectrum_parameters(self, *_args) -> None:
        self.view.gain = self.mix.gain.value()
        self.view.contrast = self.mix.contrast.value()
        self.view.set_harmonic(
            self._harmonic.table if self._harmonic is not None else None,
            self._harmonic_strength(),
        )
        self.view.refresh()
        self._ensure_harmonic()

    def _ensure_harmonic(self) -> None:
        """Fit the overtone estimate when the spectrum is up and one is wanted, once per fit."""
        strength = self._harmonic_strength()
        if strength <= 0.0 or self.view.spectrum is None or self.audio_path is None:
            self.view.set_harmonic(None, 0.0)
            return
        fit = (harmonic.HARMONICS, harmonic.DECAY)
        if self._harmonic is not None and self._harmonic_fit == fit:
            self.view.set_harmonic(self._harmonic.table, strength)
            return
        loader = HarmonicLoader(
            self.audio_path,
            self.view.spectrum.spectrum,
            harmonics=fit[0],
            decay=fit[1],
            parent=self,
        )
        loader.loaded.connect(partial(self._on_harmonic_loaded, self._audio_generation, fit))
        loader.failed.connect(partial(self._on_harmonic_failed, self._audio_generation))
        self._start_loader("harmonic_loader", loader)

    def _on_harmonic_loaded(self, generation: int, fit: tuple[int, float], result) -> None:
        if generation != self._audio_generation:
            return
        self._harmonic = result
        self._harmonic_fit = fit
        self.view.set_harmonic(result.table, self._harmonic_strength())

    def _on_harmonic_failed(self, generation: int, message: str) -> None:
        if generation != self._audio_generation:
            return
        self.statusBar().showMessage(i18n.tr("Overtone removal failed: {error}", error=message))

    def _on_analysis_progress(self, generation: int, done: int, total: int) -> None:
        if generation != self._audio_generation:
            return
        self.statusBar().showMessage(i18n.tr("Analysing … {percent}%", percent=done * 100 // max(1, total)))

    def _on_spectrum_loaded(self, generation: int, spectrum: NoteSpectrum) -> None:
        if generation != self._audio_generation:
            return
        self.view.set_spectrum(spectrum)
        self._ensure_harmonic()
        self.statusBar().showMessage(
            i18n.tr(
                "{frames} frames x {bands} bands, {ms} ms/frame, {seconds} s",
                frames=spectrum.frames,
                bands=spectrum.table.shape[1],
                ms=f"{spectrum.frame_ms:g}",
                seconds=f"{spectrum.duration:.1f}",
            )
        )

    def _update_status(self) -> None:
        self.setWindowTitle(
            i18n.tr(
                "{document} — Namioto — {notes} notes",
                document=self._document_name(),
                notes=len(self.view.notes()),
            )
        )

    # -- actions and menus -------------------------------------------------------------------

    def _register(self, name: str, action: QAction) -> QAction:
        self._actions[name] = action
        return action

    def _mirror(self, name: str, button, label: str) -> QAction:
        """A menu item that stands for a button the bar already has.

        The button stays the one place the state lives: a checkable action copies the button's state,
        follows it with `toggled`, and triggering it clicks the button rather than flipping a second
        copy, so a menu and a button can never disagree about whether a switch is on. No shortcut is
        set here: the window's `QShortcut`s are the one binding, and a second one on the action would
        fire the command twice.
        """
        action = QAction(i18n.tr(label), self)
        if button.isCheckable():
            action.setCheckable(True)
            action.setChecked(button.isChecked())
            button.toggled.connect(action.setChecked)
        action.triggered.connect(lambda *_args: button.click())
        return self._register(name, action)

    def _command(self, name: str, label: str, slot: Callable, checkable: bool = False) -> QAction:
        """A menu item for a command only the menu offers - Save As, Quit, meter editing.

        A checkable command's slot wants the state the user just chose, so it is handed the same
        `checked` Qt puts on `triggered`; every other command takes no argument.
        """
        action = QAction(i18n.tr(label), self)
        action.setCheckable(checkable)
        if checkable:
            action.triggered.connect(lambda checked=False: slot(checked))
        else:
            action.triggered.connect(lambda *_args: slot())
        return self._register(name, action)

    def _build_menus(self) -> None:
        """The window's menus, built from what the window already does.

        Every entry is either a button the bars already hold (mirrored, so one state) or a command
        this window already has. Nothing is listed that would do nothing: a dead menu item is worse
        than a missing one, and the grouping only follows WaveTone's where Namioto has the feature.
        """
        bars = self.menuBar()

        file_menu = bars.addMenu(i18n.tr("File"))
        file_menu.addAction(self._mirror("open", self.transport.open, "Open…"))
        file_menu.addAction(self._mirror("save", self.transport.save, "Save"))
        file_menu.addAction(self._command("save-as", "Save As…", self._on_save_as))
        file_menu.addSeparator()
        file_menu.addAction(self._command("import-midi", "Import MIDI…", self._import_midi_file))
        file_menu.addAction(self._mirror("export-midi", self.transport.export, "Export MIDI…"))
        file_menu.addSeparator()
        file_menu.addAction(self._register("properties", properties_action(self)))
        file_menu.addAction(self._command("quit", "Quit", self.close))

        edit_menu = bars.addMenu(i18n.tr("Edit"))
        edit_menu.addAction(self._command("undo", "Undo", self.view.undo))
        edit_menu.addAction(self._command("redo", "Redo", self.view.redo))
        edit_menu.addSeparator()
        edit_menu.addAction(self._command("cut", "Cut", self._cut_selection))
        edit_menu.addAction(self._command("copy", "Copy", self.view.copy_selection))
        edit_menu.addAction(self._command("paste", "Paste", self.view.paste_notes))
        edit_menu.addAction(self._command("delete", "Delete", self.view.delete_selection))

        view_menu = bars.addMenu(i18n.tr("View"))
        view_menu.addAction(self._command("zoom-in", "Zoom In", lambda: self._zoom_view(1.25)))
        view_menu.addAction(self._command("zoom-out", "Zoom Out", lambda: self._zoom_view(0.8)))
        view_menu.addAction(self._command("zoom-fit", "Fit", self._fit_view))
        view_menu.addAction(self._mirror("follow", self.transport.auto_page, "Follow Playhead"))

        transport_menu = bars.addMenu(i18n.tr("Transport"))
        transport_menu.addAction(self._mirror("play", self.transport.play_pause, "Play/Pause"))
        transport_menu.addAction(self._mirror("stop", self.transport.stop, "Stop"))
        transport_menu.addAction(self._register("record", record_action(self)))
        # the A-B loop is four entries rather than a card on the bar: WaveTone keeps it in the menu
        # too, and a row of its own made the control area taller than the window had room for
        transport_menu.addAction(self._command("loop", "A-B Loop", self._on_loop_toggled, checkable=True))
        transport_menu.addAction(self._command("loop-a", "Set A", self._set_loop_a))
        transport_menu.addAction(self._command("loop-b", "Set B", self._set_loop_b))
        transport_menu.addAction(self._command("loop-clear", "Clear A-B", self._clear_loop))

        tools_menu = bars.addMenu(i18n.tr("Tools"))
        tools_menu.addAction(self._register("tempo", tempo_action(self)))
        tools_menu.addAction(self._command("transcribe", "Transcription…", self._open_transcription))
        tools_menu.addAction(self._command("align", "Align Lyrics…", self._open_align))
        tools_menu.addSeparator()
        tools_menu.addAction(self._command("bookmark-add", "Add Bookmark at Playhead", self._add_bookmark))
        tools_menu.addAction(self._command("bookmark-remove", "Remove Nearest Bookmark", self._remove_bookmark))
        tools_menu.addAction(self._command("meter", "Meter…", self._edit_meter))

        sync_menu = bars.addMenu(i18n.tr("Sync"))
        send = QAction(i18n.tr("MTC Send…"), self)
        send.setCheckable(True)
        send.triggered.connect(lambda checked: self._set_mtc_mode("send" if checked else ""))
        sync_menu.addAction(self._register("mtc-send", send))
        receive = QAction(i18n.tr("MTC Receive…"), self)
        receive.setCheckable(True)
        receive.triggered.connect(lambda checked: self._set_mtc_mode("receive" if checked else ""))
        sync_menu.addAction(self._register("mtc-receive", receive))

        settings_menu = bars.addMenu(i18n.tr("Settings"))
        settings_menu.addAction(self._mirror("settings", self.transport.settings_button, "Settings…"))

        help_menu = bars.addMenu(i18n.tr("Help"))
        help_menu.addAction(self._register("shortcuts", self._shortcuts_action))

    # -- dragging a file in ------------------------------------------------------------------

    def _dragged_file(self, event) -> str:
        """The local file a drag carries, or "": the same rule the Open dialog's filter applies.

        A drag may carry several paths; this window opens one document, so only the first is taken,
        as WaveTone's `DragQueryFileW(0)` does. Anything that is not a local file - a URL from a
        browser - is left to the widget underneath rather than guessed at.
        """
        mime = event.mimeData()
        if not mime.hasUrls():
            return ""
        urls = mime.urls()
        if not urls:
            return ""
        first = urls[0]
        return first.toLocalFile() if first.isLocalFile() else ""

    def dragEnterEvent(self, event) -> None:
        if self._dragged_file(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        path = self._dragged_file(event)
        if not path:
            event.ignore()
            return
        event.acceptProposedAction()
        # `open_file` already decides by suffix and asks about unsaved work where it must, so a drop
        # behaves exactly like a file picked from the chooser - no second confirmation here
        self.open_file(path)

    def _cut_selection(self) -> None:
        """Cut is copy then delete: the roll owns those two and no third gesture.

        It is two undo steps rather than one, which is the honest cost of not adding a command to
        `roll.py` - a file this batch may not touch.
        """
        if self.view.copy_selection():
            self.view.delete_selection()

    def _import_midi_file(self) -> None:
        """The menu's Import MIDI: the chooser, then the same import the Open dialog would run."""
        if self.project_path is None:
            self.statusBar().showMessage(
                i18n.tr("Open a song first: a MIDI file is imported into a project, not on its own")
            )
            return
        chosen, _filter = QFileDialog.getOpenFileName(
            self, i18n.tr("Import MIDI"), self._start_directory(), _midi_filter()
        )
        if chosen:
            self.import_midi(chosen)

    # -- view zoom ---------------------------------------------------------------------------

    def _zoom_view(self, factor: float) -> None:
        """Zoom both axes around the middle: the menu's In and Out, on the roll's own setter."""
        zoom_x, zoom_y = self.view.zoom
        self.view.set_zoom(zoom_x * factor, zoom_y * factor)

    def _fit_view(self) -> None:
        """Fit the whole score: one bar of pixels per beat across, and the pitch range down the side."""
        beats = max(1.0, self.view.to_beats(self._duration()))
        viewport = self.view.viewport().size()
        rows = max(1, PITCH_MAX - PITCH_MIN + 1)
        self.view.set_zoom(viewport.width() / beats, viewport.height() / rows)
        self.view.center_on(beats / 2.0, (PITCH_MIN + PITCH_MAX) / 2.0)

    @property
    def meter_map(self) -> MeterMap:
        """The song's bar lengths, where `properties_dialog` looks for them."""
        return self.view.meter_map

    # -- the A-B loop ------------------------------------------------------------------------

    def _restore_loop(self) -> None:
        """Take up the span the last run left armed, if any, and show it."""
        span = self.state.loop
        if span is not None:
            self._loop_ends = [span[0], span[1]]
        self._show_loop()

    def _show_loop(self) -> None:
        """Show the span in force, filling the menu item in quietly: showing is not arming it.

        The item is the one place the armed state is visible now, so it is filled the way the toggle
        on the bar used to be - signals blocked, because a restored span is not the user switching
        the loop on. The times go to the status bar with a timeout, and only when they change: the
        window opens through here with no span at all, and there is nothing to announce then.
        """
        span = self.state.loop
        armed = span is not None
        action = self._actions["loop"]
        if action.isChecked() != armed:
            blocked = action.blockSignals(True)
            action.setChecked(armed)
            action.blockSignals(blocked)
        text = loop_text(span)
        if text != self._loop_text_shown:
            self._loop_text_shown = text
            self.statusBar().showMessage(text, LOOP_MESSAGE_MS)

    def _set_loop_a(self) -> None:
        self._store_loop_end(0, self._position())

    def _set_loop_b(self) -> None:
        self._store_loop_end(1, self._position())

    def _store_loop_end(self, index: int, seconds: float) -> None:
        """Place one end and, when both are placed, arm the loop.

        The span is session state and never touches the document: `docs/usage.md` keeps view and
        volume changes out of the dirty flag, and a loop is the same kind of thing - press A, press
        B, and the project is still saved. That is the opposite of the meter dialog, which edits the
        song and does mark it.
        """
        self._loop_ends[index] = seconds
        start, end = self._loop_ends
        if start is None or end is None or end <= start:
            self.state.loop = None
            self._show_loop()
            if start is not None and end is not None:
                self.statusBar().showMessage(i18n.tr("B must come after A"))
            return
        self.state.loop = (start, end)
        self._show_loop()
        self._save_state()

    def _clear_loop(self) -> None:
        self._loop_ends = [None, None]
        self.state.loop = None
        self._show_loop()
        self._save_state()

    def _on_loop_toggled(self, armed: bool) -> None:
        """The toggle arms the span the user placed; without both ends there is nothing to arm."""
        if not armed:
            self.state.loop = None
            self._show_loop()
            self._save_state()
            return
        start, end = self._loop_ends
        if start is None or end is None or end <= start:
            self._actions["loop"].setChecked(False)  # nothing to repeat: the item falls back
            self.statusBar().showMessage(i18n.tr("Set A and B on the transport first: A-B loop"))
            return
        self.state.loop = (start, end)
        self._show_loop()
        self._save_state()

    def _on_loop_tick(self, seconds: float) -> bool:
        """Send the transport back to A when it reaches B; True when it did."""
        target = loop_target(self.state.loop, seconds)
        if target is None:
            return False
        self._seek(target)
        return True

    # -- the audio tone: pitch and the equaliser ---------------------------------------------

    def _apply_tone(self) -> None:
        """Give the audio player the pitch and curve the settings hold.

        `audio.eq_bands` is a tuple either way: `namioto.params.coerce` builds it that way on the way
        in, and the engine takes a tuple, so nothing here has to guess at its length.
        """
        pitch = self.settings.audio.pitch_semitones
        bands = self.settings.audio.eq_bands
        self.mix.set_pitch(pitch)
        self.mix.set_eq_bands(bands)
        self._on_pitch(pitch)
        self._on_eq(self.mix.eq_bands)  # the engine always holds the curve the bar shows

    def _on_pitch(self, semitones: float) -> None:
        self.song.pitch_semitones = float(semitones)

    def _on_eq(self, bands) -> None:
        self.song.eq_bands = tuple(float(value) for value in bands)

    # -- MTC sync ----------------------------------------------------------------------------

    def _mtc_rate(self) -> Rate | None:
        """The frame rate the settings name, or None when they name none.

        A silent fall back to 25 fps would put a wrong timecode on the wire, which a receiving machine
        cannot detect, so an unset rate refuses to start instead.
        """
        return MTC_RATES.get(self.settings.mtc.framerate)

    def _choose_midi_port(self, direction: str) -> str:
        """The port to sync over: the one the settings name, or one picked from what this machine has.

        A name in the settings file is the whole point of the two `mtc` port fields - a machine set up
        once should not ask again on every start - so the dialog only appears when the field is empty
        or the port it names is gone. A port the user picks is kept, so the next start needs no
        answer either. `namioto.ui.midi_dialog` picks channels out of a file and
        `namioto.analysis.devices` picks ONNX providers, so neither is a port picker;
        `QInputDialog.getItem` is the smallest honest one, and the names come from
        `namioto.midi_port`, which imports its backend lazily, so a machine with no MIDI backend
        still starts.
        """
        try:
            names = list(midi_port.output_names() if direction == "output" else midi_port.input_names())
        except Exception as error:  # a backend that cannot enumerate ports is the same as none
            self.statusBar().showMessage(i18n.tr("MIDI ports could not be read: {error}", error=error))
            return ""
        if not names:
            message = (
                "No MIDI output is available on this machine"
                if direction == "output"
                else "No MIDI input is available on this machine"
            )
            self.statusBar().showMessage(i18n.tr(message))
            return ""
        current = self._stored_port(direction)
        if current in names:
            return current
        if current:
            self.statusBar().showMessage(
                i18n.tr("MTC port {port} is not on this machine; pick another", port=current)
            )
        chosen, accepted = QInputDialog.getItem(
            self, i18n.tr("MTC"), i18n.tr("Port"), names, 0, False
        )
        if not accepted or not chosen:
            return ""
        self._remember_port(direction, chosen)
        return chosen

    def _stored_port(self, direction: str) -> str:
        return getattr(self.settings.mtc, f"{direction}_port")

    def _remember_port(self, direction: str, port: str) -> None:
        """Keep the port that was picked, so the question is asked once rather than every start.

        `touch` rather than `apply`: this is one field of the settings already in force, and applying
        the whole lot back would re-enter `_on_settings_changed` in the middle of starting a port.
        """
        setattr(self.settings.mtc, f"{direction}_port", port)
        self.settings_store.touch()

    def _apply_mtc_settings(self, previous) -> None:
        """A frame rate or port the settings window changed reaches the port that is running.

        Sending on at the old rate after a new one was asked for would put a timecode on the wire that
        the receiving machine cannot tell is wrong, so a running port is stopped and started again;
        `_set_mtc_mode` then either opens it at the new settings or leaves it off and says why - an
        emptied frame rate refuses to start rather than quietly timing at the old one.
        """
        if not self.mtc_mode:
            return  # nothing is running, and the next start reads the settings anyway
        now, before = self.settings.mtc, previous.mtc
        if (now.framerate, now.input_port, now.output_port) == (
            before.framerate,
            before.input_port,
            before.output_port,
        ):
            return
        mode = self.mtc_mode
        self._set_mtc_mode("")  # the port open now speaks for the settings that have just been replaced
        self._set_mtc_mode(mode)

    def _set_mtc_mode(self, mode: str) -> None:
        """The one place the sync mode changes: stop what was running, start what was asked for.

        `mode` is "", "send" or "receive"; both menu items are switched to match, so a failed start
        cannot leave a checked item pointing at a port that never opened.
        """
        if mode == self.mtc_mode:
            self._sync_mtc_actions()
            return
        if self.mtc_mode:  # a port that was never started has nothing to close
            self.mtc.stop()
        self.mtc_mode = ""
        started = mode == "" or (self._start_mtc_send() if mode == "send" else self._start_mtc_receive())
        if started and mode:
            self.mtc_mode = mode
        self._sync_mtc_actions()
        self._apply_delay_policy()

    def _sync_mtc_actions(self) -> None:
        send, receive = self._actions.get("mtc-send"), self._actions.get("mtc-receive")
        if send is not None:
            send.setChecked(self.mtc_mode == "send")
        if receive is not None:
            receive.setChecked(self.mtc_mode == "receive")

    def _start_mtc_send(self) -> bool:
        rate = self._mtc_rate()
        if rate is None:
            self.statusBar().showMessage(
                i18n.tr("MTC needs a frame rate: set mtc.framerate in the settings file (24, 25, 29.97drop or 30)")
            )
            return False
        port = self._choose_midi_port("output")
        if not port:
            return False
        if not self.mtc.start_send(port, rate):
            return False
        self.mtc.seek_to(self._position())
        self.statusBar().showMessage(i18n.tr("MTC sent on {port}", port=port))
        return True

    def _start_mtc_receive(self) -> bool:
        port = self._choose_midi_port("input")
        if not port:
            return False
        if not self.mtc.start_receive(port):
            return False
        self.statusBar().showMessage(i18n.tr("MTC received on {port}", port=port))
        return True

    def _apply_delay_policy(self) -> None:
        """In receive mode the external clock says when the sound goes, so the playback delay must not
        move it: it is zeroed for the session and put back when sync ends.

        There is no delay slider among the bars - the delay is a settings field - so nothing can be
        disabled here; zeroing it is what removes the conflict, and the settings file keeps its value.
        """
        receiving = self.mtc_mode == "receive"
        self.player.delay_ms = 0 if receiving else self.settings.midi.playback_delay_ms
        if receiving:
            self.statusBar().showMessage(i18n.tr("MTC receive: the playback delay is off while the port drives"))

    def _on_mtc_position(self, seconds: float) -> None:
        """A position from the port: a jump, or a nudge of the rate. GUI thread only.

        `MtcPort` emits from RtMidi's own thread and never touches the window; this slot runs on the
        GUI thread because Qt queues a cross-thread connection, which is the whole reason the port
        only emits.
        """
        if self.mtc_mode != "receive":
            return
        drift = seconds - self._position()
        kind, value = mtc_correction(drift)
        if kind == "seek":
            self._seek(seconds)
        else:
            self._trim_playback_rate(value)

    def _trim_playback_rate(self, speed: float) -> None:
        """Nudge the note playback's rate towards the external clock, without moving the playhead.

        `MidiPortOut` reads its position as `_start + (now - _started) * _speed`, so writing a new rate
        alone would jump the playhead by `elapsed × Δspeed`. Re-anchoring `_started` on the position
        the transport already shows keeps it continuous, and the formula itself is untouched. The
        audio player is left alone: its rate lives in the engine, and its position mapping is not
        this window's to re-anchor.
        """
        player = self.player
        if not player.is_playing:
            return
        position = player.position
        player._speed = speed
        player._started = time.monotonic() - (position - player._start) / speed

    def _on_mtc_failed(self, message: str) -> None:
        self.statusBar().showMessage(i18n.tr("MTC failed: {error}", error=message))
        self._set_mtc_mode("")

    # -- bookmarks and meters ----------------------------------------------------------------

    def _add_bookmark(self) -> None:
        """A bookmark at the playhead, named for the beat it sits on until it is renamed."""
        beat = self.view.to_beats(max(0.0, self._position()))
        markers = MarkerList(self.view.markers)
        if any(abs(marker.beat - beat) < 1e-4 for marker in markers):
            self.statusBar().showMessage(i18n.tr("There is already a bookmark on this beat"))
            return
        markers.add(beat, "")
        self.view.set_markers(markers.markers)
        self._mark_dirty()  # a bookmark is song data: it goes into the project file
        self.statusBar().showMessage(i18n.tr("Bookmark added at {time}", time=format_time(self._position())))

    def _remove_bookmark(self) -> None:
        """Drop the bookmark nearest the playhead, if one is near at all."""
        markers = MarkerList(self.view.markers)
        if not markers:
            self.statusBar().showMessage(i18n.tr("This project has no bookmarks"))
            return
        beat = self.view.to_beats(max(0.0, self._position()))
        nearest = min(markers, key=lambda marker: abs(marker.beat - beat))
        markers.remove(nearest)
        self.view.set_markers(markers.markers)
        self._mark_dirty()
        self.statusBar().showMessage(i18n.tr("Bookmark removed"))

    def _edit_meter(self) -> None:
        """Edit the score's meter changes.

        Unlike the A-B loop, a meter is song data: it decides where every bar line and number falls,
        it is written into the project file and into the MIDI export, so a change here marks the
        project dirty. The dialog is built fresh each time because it edits a copy of whatever map is
        in force now.
        """
        dialog = MeterDialog(self.view.meter_map, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        meter = dialog.meter_map()
        if meter == self.view.meter_map:
            return
        self.view.set_meter_map(meter)
        self._mark_dirty()
        self.statusBar().showMessage(i18n.tr("Meter changed: {bars} bars", bars=len(meter.points)))


class MeterDialog(QDialog):
    """The score's meter changes, edited as rows of `(beat, beats_per_bar, bar_offset)`.

    A `MeterMap` normalises itself - sorted, one point per beat, a point on beat 0 - so the dialog
    does not sort or dedupe: it hands its rows over and the map decides. Same division as the
    bookmark list, and it keeps those rules in the one place that owns them.

    This is song data, unlike the A-B loop: a meter decides where every bar line and number falls,
    it is written to the project file and to the MIDI export, so the caller marks the project dirty
    after a change.
    """

    def __init__(self, meter: MeterMap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(i18n.tr("Meter"))
        self._rows: list[tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox, QSpinBox]] = []
        self._rows_layout = QVBoxLayout()
        for point in meter.points:
            self._append(point.beat, point.beats_per_bar, point.bar_offset)

        add = QPushButton(i18n.tr("Add a meter change"))
        add.clicked.connect(lambda: self._append(0.0, 4.0, 1))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(_label("Meter changes: the beat each takes effect on, its beats per bar, and the bar number"))
        layout.addLayout(self._rows_layout)
        layout.addWidget(add)
        layout.addWidget(buttons)

    def _append(self, beat: float, beats_per_bar: float, bar_offset: int) -> None:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        beat_box = QDoubleSpinBox(row)
        beat_box.setRange(0.0, 99999.0)
        beat_box.setDecimals(4)
        beat_box.setValue(beat)
        beats_box = QDoubleSpinBox(row)
        beats_box.setRange(0.25, 32.0)
        beats_box.setDecimals(2)
        beats_box.setValue(beats_per_bar)
        offset_box = QSpinBox(row)
        offset_box.setRange(1, 9999)
        offset_box.setValue(bar_offset)
        remove = QPushButton(i18n.tr("Remove"), row)
        remove.clicked.connect(lambda: self._remove(row))
        for widget in (
            _label("Beat"),
            beat_box,
            _label("Beats/bar"),
            beats_box,
            _label("Bar"),
            offset_box,
            remove,
        ):
            line.addWidget(widget)
        self._rows_layout.addWidget(row)
        self._rows.append((row, beat_box, beats_box, offset_box))

    def _remove(self, row: QWidget) -> None:
        for index, (widget, _beat, _beats, _offset) in enumerate(self._rows):
            if widget is not row:
                continue
            self._rows.pop(index)
            self._rows_layout.removeWidget(row)
            row.deleteLater()
            return

    def meter_map(self) -> MeterMap:
        """The rows as a map: one point on beat 0 is added or kept by `MeterMap` itself."""
        return MeterMap(
            tuple(
                MeterPoint(beat.value(), beats.value(), offset.value())
                for _row, beat, beats, offset in self._rows
            )
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="namioto", description="Namioto piano-roll MIDI editor")
    parser.add_argument("audio", nargs="?", help="audio file to analyse, or a .nto project or a .mid file to open")
    parser.add_argument("--channels", choices=CHANNEL_MODES, help="which channels to analyse, over the settings")
    parser.add_argument("--t-num", type=float, help="spectrum frames per second, over the settings")
    parser.add_argument("--gain", type=float, help="spectrum gain for this run")
    parser.add_argument("--contrast", type=float, help="spectrum contrast for this run")
    parser.add_argument(
        "--ipc",
        nargs="?",
        const="",
        metavar="ADDRESS",
        help="listen for remote control on ADDRESS (a socket path, tcp://host:port, or empty for the default)",
    )
    parser.add_argument("--no-ipc", action="store_true", help="do not listen for remote control")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    settings = store.load()
    added = qtplugins.system_dirs() if settings.general.system_qt_plugins else []
    if added:
        # Qt reads its plugin paths once, while the application is being built
        QCoreApplication.setLibraryPaths([*QCoreApplication.libraryPaths(), *(str(path) for path in added)])
    app = QApplication(sys.argv)
    window = MainWindow(settings, overrides={"channels": args.channels, "t_num": args.t_num})
    window.apply_overrides(gain=args.gain, contrast=args.contrast)
    if args.no_ipc:
        window.stop_ipc()
    elif args.ipc:
        window.stop_ipc()  # an address on the command line wins over the one already listening
        window.start_ipc(args.ipc)
    elif args.ipc is not None:
        window.start_ipc()
    window.show()
    if args.audio is not None:
        # a file waits for the window. Naming a project opens the platform's file chooser, and on
        # Linux that is the xdg-desktop-portal one, which is only ready once the event loop has run -
        # asking before that falls back to Qt's own dialog.
        QTimer.singleShot(0, lambda: window.open_file(args.audio))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
