# SPDX-License-Identifier: AGPL-3.0-only
"""The editor window and its wiring: the bars, the roll, the sidebar and the lyrics strip.

`MainWindow` owns the document, the players, the program's own preferences and the window's state,
and every path the program opens or writes, and runs in this one process - the analysis that must not
touch the GUI goes out to the `LoadingThread` loaders at the top of the file, or to a spawned
`namioto-transcription` child. Run it with `uv run namioto`.

Every bar value is one `_make_bindings` entry; `_remember_configuration` and the apply path both
walk that table, and `_seeding` keeps an apply's quiet writes from counting as user edits. A project
is the document: opening audio needs one (the sibling `.nto` when there is one, else a name from the
chooser), a MIDI is imported into the project already open and never on its own, and the Open dialog
offers its filter only inside a project; switching documents goes through `_confirm_discard` first.
The Export button is a menu: MIDI, the lyrics as a canonical `.krc` (`karaoke.rebuild`, gated by
`lyricmap.verify`), or a karaoke subtitle as `.ass` (`karaoke.generate_ass`).

Dirty state covers notes, channels, tempo, audio and the lyric times, and is asked about only once
the document has a file name. `general.auto_save` (off by default) writes the open project once an
edit settles (`AUTOSAVE_DELAY_MS`) and when the window loses focus; only a named document is written,
and a modal dialog holds the focus trigger back so the save-on-close prompt never answers itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PySide6.QtCore import QByteArray, QEvent, QLibraryInfo, QProcess, Qt, QThread, QTimer, QTranslator, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from namioto import i18n, ipc, lyrics, midi, params, project, subtitle
from namioto import settings as store
from namioto import state as window_state
from namioto.analysis import bpm
from namioto.analysis.spectrum import CHANNEL_MODES, NoteSpectrum
from namioto.channels import Channel, free_channel
from namioto.channels import audible as audible_channels
from namioto.channels import set_field as channel_set_field
from namioto.karaoke import AssSettings, KrcError
from namioto.karaoke.canonical import rebuild
from namioto.karaoke.operations import SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds
from namioto.karaoke.sounds import set_channel as set_sound_channel
from namioto.karaoke.writer import dumps
from namioto.lyricmap import constraints, faithful, faithful_gate
from namioto.lyricmap.channels import split as split_channels
from namioto.lyricmap.editing import detach, note_runs, operation_at
from namioto.lyricmap.notes import TimedNote, resolve
from namioto.lyricmap.problems import Problem
from namioto.lyricmap.verify import Gate
from namioto.lyrics import text_key
from namioto.playback import note_frequency
from namioto.ui import theme
from namioto.ui.align_dialog import AlignDialog
from namioto.ui.ass_dialog import AssDialog
from namioto.ui.audio import open_player
from namioto.ui.channel_panel import ChannelPanel
from namioto.ui.controls import ControlArea, EditBar, MixBar, TransportBar
from namioto.ui.ipc_commands import WindowBridge
from namioto.ui.ipc_server import IpcServer
from namioto.ui.loading import LoadingThread
from namioto.ui.lyric_editor import LyricEditor
from namioto.ui.lyric_map import LyricMapper, map_lyrics
from namioto.ui.lyrics_dialog import LyricsDialog, LyricsWatcher
from namioto.ui.midi_dialog import MidiImportDialog
from namioto.ui.open_dialog import OpenAudioDialog
from namioto.ui.roll import SNAP_CHOICES, PianoKeyboard, PianoRollView, TimelineRuler
from namioto.ui.settings_dialog import SettingsDialog, SettingsStore
from namioto.ui.song import SongPlayer, load_song
from namioto.ui.spectrogram import SpectrumLoader
from namioto.ui.strips import SoundStrip
from namioto.ui.text import note_name
from namioto.ui.transcription_dialog import TranscriptionDialog

POSITION_INTERVAL_MS = 40
SPEED_SETTLE_MS = 100
AUTOSAVE_DELAY_MS = 3000
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


def _lyrics_filter() -> str:
    return i18n.tr("Lyrics file (*{suffix})", suffix=lyrics.SUFFIX)


def _ass_filter() -> str:
    return i18n.tr("ASS subtitle (*.ass)")


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


class MainWindow(QMainWindow):
    transport_changed = Signal(bool)  # whether the transport is running, on every change
    subtitle_changed = Signal()

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
        theme.hints().colorSchemeChanged.connect(self._on_color_scheme)
        self.overrides = dict(overrides or {})  # values this run was asked for, never written back
        self._display_overrides: dict[str, float] = {}
        self._seeding = False
        self._loading = False
        self.project_path: Path | None = None
        self.project_dirty = False
        self.audio_path: str | None = None
        self.lyrics_text = ""
        self._stored_lyrics: project.Lyrics | None = None
        self._lyric_migration_error = ""
        self._lyrics_directory: TemporaryDirectory | None = None
        self._lyric_key = ""
        self._lyric_model = ""
        self._lyric_flags: tuple[bool, ...] = ()
        self._lyric_scores = None
        self._lyric_problems = None
        self._lyric_filtered: tuple = ()
        self._lyric_error = ""
        self._lyric_alignment = None
        self._lyric_map_thread: LyricMapper | None = None
        self._lyric_drag_request = None
        self._lyric_drag_result = None
        self._lyric_drag_valid = False
        self._lyric_map_pending = None
        self._lyric_map_revision = 0
        self._lyric_map_result = None
        self._lyric_map_status = QLabel(i18n.tr("Calculating lyric mapping…"), self)
        self._lyric_map_status.hide()
        self.statusBar().addPermanentWidget(self._lyric_map_status)
        self.loader: SpectrumLoader | None = None
        self.tempo_loader: TempoLoader | None = None
        self._tempo_manual = False
        self.song_loader: SongLoader | None = None
        # the audio a result belongs to: a loader the window replaced may still report its own
        self._audio_generation = 0
        self._workers: set[QThread] = set()
        self.ipc: IpcServer | None = None
        self._bridge: WindowBridge | None = None
        self._reported_playing: bool | None = None

        editor = self.project_settings.editor
        self.view = PianoRollView()
        self.view.set_zoom(editor.zoom_x, editor.zoom_y)
        self.view.initial_center = (self.project_settings.view.center_x, self.project_settings.view.center_y)
        self.view.overtone_highlight = self.settings.editor.overtone_highlight
        self.view.division = editor.division
        self.ruler = TimelineRuler(self.view)
        self.sound_strip = SoundStrip(self.view)
        self.sound_strip.setVisible(False)
        self.sound_strip.lyric_action_requested.connect(self._on_lyric_action)
        self.sound_strip.mapping_drag_started.connect(self._begin_lyric_drag)
        self.sound_strip.mapping_preview_requested.connect(self._preview_lyric_drag)
        self.sound_strip.mapping_drag_finished.connect(self._finish_lyric_drag)
        self.keyboard = PianoKeyboard(self.view)
        self.player, self.player_name = self._make_player()
        self._current_player_key = self._player_key()
        self.player.gain = self.project_settings.playback.midi_volume / 100.0
        self.song = SongPlayer(self)
        self.lyrics_watcher = LyricsWatcher(self)
        self.lyrics_watcher.changed.connect(self._on_lyrics_file_changed)
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
        layout.addWidget(self.sound_strip, 1, 1)
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
        self.transport.export_krc_requested.connect(self._on_export_krc)
        self.transport.export_ass_requested.connect(self._on_export_ass)
        self.edit.transcribe_requested.connect(self._open_transcription)
        self.edit.lyrics_requested.connect(self._open_lyrics)
        self.edit.align_requested.connect(self._open_align)
        self.sound_strip.channel_requested.connect(
            lambda line, index, channel: self._set_lyric_channel(line, channel, index=index)
        )
        self.lyric_editor = LyricEditor(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lyric_editor)
        self.sound_strip.editor_requested.connect(self.lyric_editor.open_at)
        self.view.lyric_text_changed.connect(self._restore_lyric_text)
        self.transport.note_lyrics.toggled.connect(self.view.set_note_lyrics_visible)
        self.transport.lyric_scope.toggled.connect(self.sound_strip.set_current_channel_only)
        self.view.lyrics_changed.connect(self._update_lyric_scope)
        self._update_lyric_scope()
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
        self.view.notes_changed.connect(self._remap_lyrics_async)
        self.view.gesture_started.connect(self._invalidate_lyric_mapping)
        self.view.gesture_finished.connect(self._remap_lyrics_async)
        self.view.channels_changed.connect(self._on_channels_changed)
        self.view.active_channel_changed.connect(self._on_active_channel_changed)
        self.transport.channels.toggled.connect(self.channel_panel.setVisible)
        self.edit.quantize_requested.connect(self.view.quantize_notes)
        self.transport.bpm.valueChanged.connect(self._mark_dirty)
        self.play_shortcut = QShortcut(QKeySequence("Space"), self)
        self.play_shortcut.activated.connect(self._toggle_play)
        for keys, slot in (
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
            QShortcut(QKeySequence(standard), self).activated.connect(slot)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.view.redo)
        self.cursor_note = QLabel()
        self.statusBar().addPermanentWidget(self.cursor_note)
        self._restore_session()
        self._show_hint()
        self._watch_lyrics()
        self._update_status()
        self.view.setFocus()  # the roll holds the keyboard, so the bar opens without a focus ring on its first button
        if self.settings.remote.enabled:
            self.start_ipc()

    def _make_player(self):
        return open_player(self)

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
        self._release_lyrics_copy()
        self._loading = True
        try:
            self.project_settings = project.default_settings(self.state.project)
            for name, value in (analysis or {}).items():
                if ("analysis", name) in project.FIELD_SPECS:
                    params.set_value(self.project_settings, "analysis", name, value)
            self.project_path = None
            self.project_dirty = False
            self.autosave_timer.stop()
            self._stored_lyrics = None
            self._lyric_migration_error = ""
            self.lyrics_text = ""
            self.transport.bpm.setValue(project.FIELD_SPECS[("tempo", "bpm")].default)
            self.transport.grid_offset.setValue(project.FIELD_SPECS[("editor", "grid_offset_ms")].default)
            # the notes go first: `set_channels` gives a channel back to every note it still finds,
            # so setting the channels over the old notes would leave their channels behind
            self.view.clear_notes()
            self.view.set_channels([Channel(channel=0)])
            self.view.set_spectrum(None)
            self.view.undo_stack.clear()
            self._apply_project_settings()
            self._watch_lyrics()
        finally:
            self._loading = False
        self._remap_lyrics()
        self._update_status()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.settings,
            parent=self,
        )
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
        self.settings = settings
        self.subtitle_changed.emit()
        self._seed_bindings(lambda binding: binding.model is None)
        self._apply_theme()
        self._update_status()
        if settings.remote.enabled and self.ipc is None:
            self.start_ipc()  # a switch flipped on reaches a running editor at once
        elif not settings.remote.enabled and self.ipc is not None:
            self.stop_ipc()

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
            (note.pitch, self.view.to_seconds(note.start), self.view.to_seconds(note.duration), note.channel)
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

    def _on_export_krc(self) -> bool:
        """Write the lyrics out as a `.krc`, with the mapping's groups and `.N` folded in."""
        if not self.lyrics_text:
            self.statusBar().showMessage(i18n.tr("There are no lyrics to export"))
            return False
        name = Path(self.audio_path).stem if self.audio_path is not None else "untitled"
        suggested = Path(self._start_directory()) / f"{name}{lyrics.SUFFIX}"
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            i18n.tr("Export lyrics"),
            str(suggested),
            _lyrics_filter(),
        )
        if not chosen:
            return False
        return self.export_krc(chosen)

    def export_krc(self, path: str | Path) -> bool:
        """Write the lyrics and the current mapping to `path`, adding the `.krc` suffix if missing."""
        target = Path(path)
        if target.suffix.lower() != lyrics.SUFFIX:
            target = target.with_name(target.name + lyrics.SUFFIX)
        gate = self._verification()
        if not gate.open():
            self._locate_export_problem(gate)
            self.statusBar().showMessage(self._gate_message(gate))
            return False
        try:
            lyrics.save(target, gate.canonical)
        except OSError as failure:
            self.statusBar().showMessage(i18n.tr("Lyrics file could not be written: {error}", error=failure))
            return False
        self.statusBar().showMessage(i18n.tr("Exported {name}", name=target.name))
        return True

    def _verification(self, subtitle: bool = False):
        """Run the one gate over the mapping the strip shows."""
        if self._lyric_migration_error:
            return Gate((Problem("invalid_lyrics", self._lyric_migration_error),))
        if self.view.lyric_mapping_pending:
            return Gate((Problem("mapping_pending", "the lyric mapping is still being calculated"),))
        lines = list(self.view.lyric_lines)
        if not lines:
            return Gate((Problem("no_lyric_sounds", "the lyrics have no sounds"),))
        if any(not 0 <= ref.line < len(lines) for op in self.view.lyric_operations for ref in op.sounds):
            return Gate((Problem("invalid_anchor", "an operation names a missing lyric line"),))
        issues = []
        try:
            parts = split_channels(self.lyrics_text)
        except KrcError as error:
            return Gate((Problem("invalid_lyrics", str(error)),))
        for channel, part in parts.items():
            text = part.text
            resolved = resolve(tuple(TimedNote(*note) for note in self._target_notes(channel)), include_branches=False)
            try:
                spans = faithful.read(text, resolved.stream)
            except (KrcError, ValueError) as error:
                issues.append(Problem("invalid_lyrics", str(error)))
                continue
            if subtitle:
                dropped = tuple(ref for ref, (_run, slots) in constraints.read(text).items() if slots == 0)
                gate = faithful_gate(
                    spans,
                    resolved.stream,
                    filtered=resolved.filtered,
                    subtitle=True,
                    dropped=dropped,
                )
                for problem in gate.problems:
                    if problem.code == "incomplete_alignment":
                        refs = tuple(part.refs[ref.line][ref.index] for ref in problem.refs)
                    elif problem.code == "filtered_note":
                        refs = problem.refs
                    else:
                        refs = tuple(ref for row in part.refs for ref in row)[:1]
                    issues.append(Problem(problem.code, problem.message, refs))
        if issues:
            return Gate(tuple(issues))
        return Gate((), self.lyrics_text)

    def _gate_message(self, gate) -> str:
        counts = gate.counts()
        summary = ", ".join(f"{code} ×{count}" for code, count in sorted(counts.items()))
        locations = [
            (ref.line, ref.index) for problem in gate.problems for ref in problem.refs if isinstance(ref, SoundRef)
        ]
        if locations:
            row, index = locations[0]
            summary += i18n.tr(" (line {line}, sound {sound})", line=row + 1, sound=index + 1)
        return i18n.tr("The mapping cannot be exported: {problems}", problems=summary)

    def _locate_export_problem(self, gate) -> None:
        for problem in gate.problems:
            for ref in problem.refs:
                if isinstance(ref, SoundRef):
                    self.sound_strip.reveal_sound(ref.line, ref.index)
                    return

    def _on_export_ass(self) -> bool:
        """Write the lyrics out as a karaoke subtitle, timed by the mapping the strip shows."""
        if not self.lyrics_text or not self.view.lyric_lines:
            message = self._lyric_error if self.lyrics_text else i18n.tr("There are no lyrics to export")
            self.statusBar().showMessage(message or i18n.tr("There are no lyrics to export"))
            return False
        if not self.view.notes():
            self.statusBar().showMessage(i18n.tr("There are no notes to time the subtitle with"))
            return False
        gate = self._verification(subtitle=True)
        if not gate.open():
            self._locate_export_problem(gate)
            self.statusBar().showMessage(self._gate_message(gate))
            return False
        dialog = AssDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        selected = dialog.values()
        name = Path(self.audio_path).stem if self.audio_path is not None else "untitled"
        suggested = Path(self._start_directory()) / f"{name}.ass"
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            i18n.tr("Export ASS subtitle"),
            str(suggested),
            _ass_filter(),
        )
        if not chosen:
            return False
        if not self.export_ass(chosen, settings=AssSettings(**vars(selected.ass))):
            return False
        self.settings.ass = selected.ass
        self.settings_store.flush()
        self.subtitle_changed.emit()
        return True

    def ass_source(self) -> tuple[str, bytes]:
        """Export the current KRC first, then MIDI, as one GUI-thread snapshot."""
        gate = self._verification(subtitle=True)
        if not gate.open():
            self._locate_export_problem(gate)
            raise ValueError(self._gate_message(gate))
        krc = gate.canonical
        data = midi.to_bytes(**self._midi_export())
        return krc, data

    def export_ass(
        self,
        path: str | Path,
        *,
        settings: AssSettings | None = None,
        save_krc: str | Path | None = None,
        save_midi: str | Path | None = None,
    ) -> bool:
        """Build the subtitle from the current KRC and MIDI exports, optionally saving both."""
        target = subtitle.ass_path(path)
        try:
            krc, data = self.ass_source()
            text = subtitle.build_ass(
                krc, data, settings=settings or self._ass_settings(), wavetone=self.settings.midi.wavetone
            )
            subtitle.save_export(target, krc, data, text, save_krc=save_krc, save_midi=save_midi)
        except (KrcError, ValueError, EOFError) as error:
            self.statusBar().showMessage(i18n.tr("The subtitle could not be built: {error}", error=error))
            return False
        except OSError as error:
            self.statusBar().showMessage(i18n.tr("Subtitle file could not be written: {error}", error=error))
            return False
        self.statusBar().showMessage(i18n.tr("Exported {name}", name=target.name))
        return True

    def _ass_settings(self) -> AssSettings:
        """The subtitle's own preferences, gathered for `karaoke.generate_ass`."""
        return AssSettings(**vars(self.settings.ass))

    def load_project(self, path: str | Path) -> bool:
        """Open a project: its values come over the running ones, and its notes replace the roll."""
        try:
            opened = project.load(path)
        except (OSError, ValueError) as error:
            self.statusBar().showMessage(i18n.tr("Project could not be opened: {error}", error=error))
            return False
        self._release_lyrics_copy()
        self._loading = True
        try:
            self.project_settings = opened.settings
            self.project_path = Path(path)
            self.project_dirty = False
            self.autosave_timer.stop()
            self._stored_lyrics = opened.lyrics
            self._lyric_alignment = None
            self._lyric_migration_error = ""
            self._lyric_key = ""
            self._watch_lyrics()
            missing = self._open_audio(project.resolve_audio(self.project_path, opened.audio))
            self.view.set_channels(opened.channels)
            loaded = (
                (note.pitch, self.view.to_beats(note.start), self.view.to_beats(note.duration), note.channel, note.id)
                for note in opened.notes
            )
            self.view.set_notes(loaded, opened.next_id)
            self._apply_project_settings()  # the bars, the zoom and the player take the document's values
            self.view.undo_stack.clear()  # another document starts its history over
        finally:
            self._loading = False
        migrated = self._migrate_lyrics()
        self._remap_lyrics()
        self._update_status()
        self.statusBar().showMessage(
            self._lyric_migration_error
            or (
                i18n.tr("Opened {name} — {notes} notes", name=self.project_path.name, notes=len(opened.notes)) + missing
            )
        )
        if migrated:
            self._mark_dirty()
        return True

    def save_project(self, path: str | Path) -> bool:
        """Write the notes and the values that belong to them out to `path`."""
        if self._lyric_migration_error:
            self.statusBar().showMessage(self._lyric_migration_error)
            return False
        self._remember_configuration()
        self._remember_session()
        target = Path(path)
        # scene order is not a file's order: sorted notes keep a saved project stable to diff
        notes = tuple(
            sorted(
                project.Note(
                    self.view.to_seconds(note.start),
                    self.view.to_seconds(note.duration),
                    note.pitch,
                    note.channel,
                    note.id,
                )
                for note in self.view.notes()
            )
        )
        lyrics_times = (
            project.Lyrics(
                text=self.lyrics_text,
                key=self._lyric_key,
                model=self._lyric_model,
                mode="read",
                raw=self.view.lyric_raw,
                flagged=self._lyric_flags,
                scores=tuple(tuple(row) for row in (self._lyric_scores or ())),
                problems=tuple(tuple(row) for row in (self._lyric_problems or ())),
            )
            if self.lyrics_text
            else None
        )
        payload = project.Project(
            settings=self.project_settings,
            audio=project.store_audio(target, self.audio_path),
            channels=tuple(self.view.channels),
            notes=notes,
            next_id=self.view.document.next_id,
            lyrics=lyrics_times,
        )
        try:
            project.save(payload, target)
        except OSError as error:
            self.statusBar().showMessage(i18n.tr("Project could not be saved: {error}", error=error))
            return False
        self.project_path = target
        self.project_dirty = False
        self.autosave_timer.stop()
        self._stored_lyrics = lyrics_times
        self._watch_lyrics()
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
                i18n.tr("{notes} tempo changes; the grid takes the first tempo", notes=imported.tempo_changes)
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
                self.transport.bpm.setValue(imported.bpm)  # scene units are beats, and the file sets them
                self.view.replace(
                    imported.channels,
                    [
                        (note.pitch, self.view.to_beats(note.start), self.view.to_beats(note.duration), note.channel)
                        for note in imported.notes
                    ],
                    "Import MIDI",
                )
        finally:
            self._loading = False
        self._remap_lyrics_async()
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
            (note.pitch, self.view.to_beats(note.start), self.view.to_beats(note.duration), landing[note.channel])
            for note in imported.notes
        ]
        self.view.replace(channels.values(), kept + arriving, "Merge MIDI")

    def _midi_export(self) -> dict:
        notes = tuple(
            project.Note(
                self.view.to_seconds(note.start),
                self.view.to_seconds(note.duration),
                note.pitch,
                note.channel,
            )
            for note in self.view.notes()
        )
        return {
            "channels": tuple(self.view.channels),
            "notes": notes,
            "bpm": self.project_settings.tempo.bpm,
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
        """Ask GAME for the singing voice's notes, over the audio the session is already listening to."""
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
            self._adopt_transcription(dialog.notes())

    def _adopt_transcription(self, notes) -> None:
        """Replace channel 0's notes with the transcription."""
        if not notes:
            self.statusBar().showMessage(i18n.tr("GAME found no notes in the loaded audio"))
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
        self.view.replace(channels, kept + [(*note, 0) for note in arriving], "Transcribe with GAME")
        self.statusBar().showMessage(
            i18n.tr("GAME found {notes} notes on channel {channel}", notes=len(arriving), channel=0)
        )

    def lyrics_path(self) -> Path | None:
        """The document's temporary KRC working copy, never a project-directory sidecar."""
        if self.project_path is None:
            return None
        if self._lyrics_directory is None:
            self._lyrics_directory = TemporaryDirectory(prefix="namioto-lyrics-")
        return Path(self._lyrics_directory.name) / "lyrics.krc"

    def _prepare_lyrics_copy(self) -> Path | None:
        try:
            path = self.lyrics_path()
            if path is None:
                return None
            lyrics.save(path, self.lyrics_text)
        except OSError as error:
            self.lyrics_watcher.watch(None)
            self.statusBar().showMessage(str(error))
            return None
        self.lyrics_watcher.watch(path)
        return path

    def _release_lyrics_copy(self) -> None:
        self.lyrics_watcher.watch(None)
        if self._lyrics_directory is not None:
            self._lyrics_directory.cleanup()
            self._lyrics_directory = None

    def _update_lyric_controls(self) -> None:
        named = self.project_path is not None
        self.edit.lyrics.setEnabled(named)
        self.edit.align.setEnabled(named)
        self.view.lyric_editable = True
        self.lyrics_watcher.watch(self.lyrics_path() if self._lyrics_directory is not None else None)

    def _lyrics_text_editable(self) -> bool:
        return not self.view.gesture_active and not self._lyric_migration_error

    def _migrate_lyrics(self) -> bool:
        stored = self._stored_lyrics
        if stored is None or stored.mode != "edit":
            return False
        try:
            operations = stored.operations
            try:
                rebuild(stored.text, operations)
            except KrcError:
                result = map_lyrics(
                    stored.text,
                    natural_sounds(stored.text),
                    stored.raw,
                    stored.scores,
                    stored.flagged,
                    mode="edit",
                    anchors=tuple(operation for operation in stored.operations if operation.confirmed),
                    use_constraints=stored.use_constraints,
                    notes_by_channel={channel: self._target_notes(channel) for channel in split_channels(stored.text)},
                )
                if result.error:
                    raise KrcError(result.error) from None
                operations = result.operations
            text = self._canonical_mapping(stored.text, operations)
            self._stored_lyrics = stored._replace(text=text, key=text_key(text), mode="read", operations=())
            self._lyric_key = ""
            self._watch_lyrics()
            return True
        except (KrcError, ValueError) as error:
            self._lyric_migration_error = i18n.tr(
                "The saved lyric mapping could not be recovered: {error}", error=error
            )
            self._invalidate_lyric_mapping()
            self._set_lyric_mapping_pending(False)
            self.view.lyric_editable = False
            return False

    def _watch_lyrics(self) -> None:
        """Restore project lyrics and watch only the temporary editing copy."""
        stored = self._stored_lyrics
        self.lyrics_text = stored.text if stored is not None else ""
        self._update_lyric_controls()
        if not self._load_sounds():
            self._remap_lyrics_async()

    def _load_sounds(self) -> bool:
        """Derive the sounds of the open `.krc` and lay them onto the notes the project kept."""
        text = self.lyrics_text
        self.view.lyric_text = text
        key = text_key(text) if text else ""
        if key == self._lyric_key and bool(self.view.lyric_lines) == bool(text):
            return False
        self._lyric_key = key
        self._lyric_model = ""
        lines = []
        if text:
            try:
                lines = natural_sounds(text)
            except KrcError as error:
                self._lyric_error = str(error)
                self.statusBar().showMessage(str(error))
                lines = []
        else:
            self._lyric_error = ""
        stored = self._stored_lyrics
        if (
            lines
            and stored is not None
            and stored.key == key
            and len(stored.raw) == len(lines)
            and all(len(row) == len(line.sounds) for row, line in zip(stored.raw, lines, strict=True))
        ):
            self._lyric_model = stored.model
            raw = [list(row) for row in stored.raw]
            self._lyric_flags = stored.flagged if len(stored.flagged) == len(lines) else ()
            self._lyric_scores = [list(row) for row in stored.scores] if len(stored.scores) == len(lines) else None
            problems = stored.problems if len(stored.problems) == len(lines) else ()
            self._lyric_problems = [list(row) for row in problems]
            self.view.lyric_operations = ()
        else:
            raw = [[(None, None)] * len(line.sounds) for line in lines]
            self._lyric_flags = ()
            self._lyric_scores = None
            self._lyric_problems = None
            self.view.lyric_operations = []
        self.view.lyric_evidence = (
            self._lyric_model,
            tuple(self._lyric_flags),
            tuple(tuple(row) for row in (self._lyric_scores or ())),
            tuple(tuple(row) for row in (self._lyric_problems or ())),
        )
        self.view.load_lyrics(lines, raw, raw=raw, editable=True)
        self._remap_lyrics()
        return True

    def _remap_lyrics(self) -> None:
        """Request a full mapping, including when the current channel is mapped again."""
        self._lyric_map_result = None
        self._remap_lyrics_async()

    def _mapping_request(self, lines) -> dict:
        """Everything the mapper thread needs, snapshotted so the GUI thread may move on."""
        lines = tuple(lines)
        raw = [list(row) for row in self.view.lyric_raw]
        if len(raw) != len(lines):
            raw = [[(None, None)] * len(line.sounds) for line in lines]
        return {
            "text": self.lyrics_text,
            "lines": lines,
            "raw": raw,
            "scores": tuple(tuple(row) for row in (self._lyric_scores or ())),
            "flagged": tuple(self._lyric_flags),
            "notes_by_channel": self._notes_by_channel({sound.channel for line in lines for sound in line.sounds}),
            "mode": "read",
        }

    def _set_lyric_mapping_pending(self, pending: bool) -> None:
        self.view.lyric_mapping_pending = pending
        self.transport.export_krc_action.setEnabled(not pending)
        self.transport.export_ass_action.setEnabled(not pending)
        self._lyric_map_status.setVisible(pending)
        self.subtitle_changed.emit()

    def _invalidate_lyric_mapping(self) -> None:
        self._lyric_alignment = None
        self._lyric_map_revision += 1
        self._lyric_map_pending = None
        self._set_lyric_mapping_pending(bool(self.view.lyric_lines))

    def _remap_lyrics_async(self) -> None:
        if self._loading or self.view.gesture_active:
            return
        if self._lyric_migration_error:
            self._lyric_error = self._lyric_migration_error
            self._set_lyric_mapping_pending(False)
            return
        if not self.view.lyric_lines:
            self._invalidate_lyric_mapping()
            self._lyric_map_result = None
            self._lyric_filtered = ()
            if not self.lyrics_text:
                self._lyric_error = ""
            self.sound_strip.hide()
            self.view.set_filtered_notes(())
            return
        snapshot = self._mapping_request(self.view.lyric_lines)
        if self._lyric_alignment is not None:
            _revision, _model, request, _refs = self._lyric_alignment
            if all(request[key] == snapshot[key] for key in ("text", "lines", "notes_by_channel")):
                return
            self._lyric_alignment = None
        if self._lyric_map_pending is not None and self._lyric_map_pending[1] == snapshot:
            return
        thread = self._lyric_map_thread
        if thread is not None and thread.revision == self._lyric_map_revision and thread.request == snapshot:
            return
        self._lyric_map_revision += 1
        request = (self._lyric_map_revision, snapshot)
        self._set_lyric_mapping_pending(bool(self.view.lyric_lines))
        if thread is not None:
            self._lyric_map_pending = request
            return
        self._start_lyric_mapper(request)

    def _start_lyric_mapper(self, request) -> None:
        revision, request = request
        thread = LyricMapper(revision, request, self)
        if request.get("mode") == "read":
            thread.previous = self._lyric_map_result
        thread.mapped.connect(self._on_lyric_mapping)
        thread.failed.connect(partial(self._on_lyric_mapping_failed, revision))
        thread.finished.connect(partial(self._on_lyric_mapper_finished, thread))
        thread.finished.connect(thread.deleteLater)
        self._lyric_map_thread = thread
        thread.start()

    def _on_lyric_mapping(self, result) -> None:
        revision, mapping = result
        if revision != self._lyric_map_revision:
            return
        if self._lyric_alignment is not None and revision == self._lyric_alignment[0]:
            self._commit_alignment(mapping)
            return
        if self.view.gesture_active:
            return
        thread = self._lyric_map_thread
        snapshot = self._mapping_request(self.view.lyric_lines)
        if thread is None or any(thread.request[key] != value for key, value in snapshot.items() if key != "previous"):
            return
        self._lyric_map_result = mapping
        self._apply_lyric_mapping(mapping)
        self._set_lyric_mapping_pending(False)
        if mapping.error:
            self.statusBar().showMessage(mapping.error)

    def _on_lyric_mapping_failed(self, revision: int, message: str) -> None:
        if revision != self._lyric_map_revision:
            return
        self._lyric_alignment = None
        self._lyric_error = message
        self._set_lyric_mapping_pending(False)
        self.statusBar().showMessage(message)

    def _on_lyric_mapper_finished(self, thread) -> None:
        if thread is not self._lyric_map_thread:
            return
        self._lyric_map_thread = None
        request = self._lyric_map_pending
        self._lyric_map_pending = None
        if request is not None:
            self._start_lyric_mapper(request)

    def _apply_lyric_mapping(self, result) -> None:
        self.view.lyric_text = self.lyrics_text
        self._lyric_error = result.error
        self._lyric_filtered = result.filtered
        self.view.load_lyrics(
            result.lines,
            result.spans,
            result.red,
            raw=[list(row) for row in result.raw],
            zero=result.zero,
            group=result.group,
            mapped=result.mapped,
            editable=True,
            operations=result.operations,
            filtered=(note.id for note in result.filtered),
        )
        self.sound_strip.setVisible(bool(result.lines))
        self.edit.align.setEnabled(self.project_path is not None)

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

    def _open_lyrics(self) -> None:
        """Read a text into a `.krc` with a model, or by pasting what a web model answered."""
        if not self._lyrics_text_editable():
            return
        path = self._prepare_lyrics_copy()
        if path is None:
            return
        dialog = LyricsDialog(
            path,
            self.settings.lyrics,
            parent=self,
            source_directory=self.project_path.parent if self.project_path else None,
        )
        dialog.saved.connect(self._on_lyrics_saved)
        dialog.open_requested.connect(self._open_lyrics_editor)
        dialog.exec()

    def _on_lyrics_saved(self, text: str) -> bool:
        """Validate and adopt a text edit without starting alignment."""
        if not self._lyrics_text_editable():
            return False
        if text == self.lyrics_text:
            return True
        try:
            faithful.read(text, ())
        except (KrcError, ValueError) as error:
            self.statusBar().showMessage(i18n.tr("The lyrics could not be read: {error}", error=error))
            return False
        lines = natural_sounds(text)
        same_sounds = tuple(tuple(sound.token for sound in line.sounds) for line in lines) == tuple(
            tuple(sound.token for sound in line.sounds) for line in self.view.lyric_lines
        )
        with self.view.lyric_edit(i18n.tr("Edit lyrics")):
            self.lyrics_text = text
            self._stored_lyrics = None
            if same_sounds:
                self._lyric_key = text_key(text)
                self.view.lyric_text = text
                self.view.load_lyrics(lines, self.view.lyric_times, raw=self.view.lyric_raw)
            else:
                self._load_sounds()
        self._remap_lyrics()
        self._mark_dirty()
        return True

    def _on_lyrics_file_changed(self) -> None:
        path = self.lyrics_path()
        if path is None or not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            self.statusBar().showMessage(i18n.tr("The lyrics could not be read: {error}", error=error))
            return
        if text == self.lyrics_text:
            return
        if self._on_lyrics_saved(text):
            self.statusBar().showMessage(i18n.tr("Lyrics reloaded from {name}", name=path.name))

    def _open_lyrics_editor(self) -> None:
        if not self._lyrics_text_editable():
            return
        path = self._prepare_lyrics_copy()
        if path is None:
            return
        command = lyrics.editor_command(self.settings.lyrics.editor)
        started, _pid = QProcess.startDetached(command[0], [*command[1:], str(path)])
        if not started:
            self.statusBar().showMessage(i18n.tr("The lyrics editor could not be started"))

    def _open_align(self) -> None:
        """Ask the forced aligner for a time on every sound of the project lyrics."""
        if self.view.gesture_active or self._lyric_migration_error:
            return
        path = self.project_path
        if path is None:
            self.statusBar().showMessage(i18n.tr("Save the project before aligning the lyrics"))
            return
        if not self.lyrics_text:
            self.statusBar().showMessage(i18n.tr("There are no lyrics in {name} to align", name=path.name))
            return
        if not self.view.lyric_lines:
            self.statusBar().showMessage(
                i18n.tr("The lyrics could not be read: {error}", error=self._lyric_error or path.name)
            )
            return
        if self.audio_path is None:
            self.statusBar().showMessage(i18n.tr("Load the audio before aligning the lyrics"))
            return
        channel = self.view.active_channel
        try:
            part = split_channels(self.lyrics_text).get(channel)
        except KrcError as error:
            self.statusBar().showMessage(str(error))
            return
        if part is None:
            self.statusBar().showMessage(i18n.tr("The selected channel has no lyric sounds"))
            return
        dialog = AlignDialog(
            self.audio_path, self.lyrics_text, self.view.bpm, self, offset=self.view.offset, channel=channel
        )
        generation = self._audio_generation

        def adopt(raw, model, flagged):
            if dialog.text != self.lyrics_text or generation != self._audio_generation:
                return
            self._adopt_alignment(
                raw, model, flagged, use_constraints=dialog.parameters()["use_constraints"], channel=channel
            )

        dialog.aligned.connect(adopt)
        dialog.exec()

    def _adopt_alignment(
        self, raw, model: str, flagged=(), *, use_constraints: bool = True, channel: int | None = None
    ) -> None:
        if self.view.gesture_active or self._lyric_migration_error:
            return
        lines = self.view.lyric_lines
        if channel is None:
            refs = tuple(tuple(SoundRef(row, at) for at in range(len(line.sounds))) for row, line in enumerate(lines))
        else:
            part = split_channels(self.lyrics_text).get(channel)
            refs = part.refs if part is not None else ()
        if (
            not refs
            or len(raw) != len(refs)
            or any(len(row) != len(owned) for row, owned in zip(raw, refs, strict=True))
        ):
            return
        combined = [list(row) for row in self.view.lyric_raw]
        flags = list(self._lyric_flags) if len(self._lyric_flags) == len(lines) else [False] * len(lines)
        row_flags = {}
        for position, owned in enumerate(refs):
            for ref, span in zip(owned, raw[position], strict=True):
                combined[ref.line][ref.index] = span
                row_flags[ref.line] = row_flags.get(ref.line, False) or (
                    flagged[position] if position < len(flagged) else False
                )
        for row, flagged_row in row_flags.items():
            flags[row] = (
                flagged_row
                if channel is None or all(sound.channel == channel for sound in lines[row].sounds)
                else flags[row] or flagged_row
            )
        snapshot = dict(
            self._mapping_request(lines),
            raw=combined,
            flagged=tuple(flags),
            scores=(),
            mode="edit",
            use_constraints=use_constraints,
            solve_channels=(channel,) if channel is not None else None,
        )
        self._lyric_map_revision += 1
        revision = self._lyric_map_revision
        self._lyric_alignment = revision, model, snapshot, refs
        request = revision, snapshot
        self._set_lyric_mapping_pending(True)
        if self._lyric_map_thread is not None:
            self._lyric_map_pending = request
        else:
            self._start_lyric_mapper(request)

    def _canonical_mapping(self, text, operations, *, channels=None, line_indices=None, preserve_groups=False) -> str:
        rebuilt = rebuild(text, operations, line_indices=line_indices)
        lines = natural_sounds(text)
        groups = [
            run
            for run in note_runs(text)
            if len(run.sounds) > 1
            and (
                preserve_groups
                or (
                    channels is not None
                    and lines[run.sounds[0].line].sounds[run.sounds[0].index].channel not in channels
                )
            )
        ]
        if groups:
            plan = list(note_runs(rebuilt))
            for group in groups:
                refs = set(group.sounds)
                covered = [run for run in plan if refs.intersection(run.sounds)]
                if {ref for run in covered for ref in run.sounds} != refs:
                    raise KrcError("the mapping does not preserve the existing group boundary")
                plan = [run for run in plan if run not in covered] + [group]
            rebuilt = rebuild(rebuilt, plan, line_indices=line_indices)
        for channel, part in split_channels(rebuilt).items():
            if channels is not None and channel not in channels:
                continue
            resolved = resolve([TimedNote(*note) for note in self._target_notes(channel)], include_branches=False)
            expected = sorted(
                part.operations(operations), key=lambda operation: (operation.sounds[0].line, operation.sounds[0].index)
            )
            if faithful.operations(part.text, resolved.stream) != tuple(
                replace(operation, confirmed=True) for operation in expected
            ):
                raise KrcError("the mapping cannot be represented by sequential KRC NOTE slots")
        return rebuilt

    def _commit_alignment(self, mapping) -> None:
        _revision, model, snapshot, refs = self._lyric_alignment
        self._lyric_alignment = None
        current = self._mapping_request(self.view.lyric_lines)
        if self.view.gesture_active or any(
            snapshot[key] != current[key] for key in ("text", "lines", "notes_by_channel")
        ):
            self._remap_lyrics()
            return
        selected = snapshot["solve_channels"]
        errors = [error for channel, error in mapping.channel_errors if selected is None or channel in selected]
        try:
            if errors:
                raise KrcError("; ".join(errors))
            text = self._canonical_mapping(
                snapshot["text"],
                mapping.operations,
                channels=selected,
                line_indices=sorted({ref.line for row in refs for ref in row}),
                preserve_groups=snapshot["use_constraints"],
            )
        except (KrcError, ValueError) as error:
            self._remap_lyrics()
            self.statusBar().showMessage(str(error))
            return
        with self.view.lyric_edit(i18n.tr("Align lyrics")):
            self.lyrics_text = text
            self.view.lyric_text = text
            self._lyric_key = text_key(text)
            self._lyric_model = model
            self._lyric_flags = snapshot["flagged"]
            self._lyric_scores = None
            self._lyric_problems = None
            self.view.lyric_evidence = (model, snapshot["flagged"], (), ())
            self.view.load_lyrics(natural_sounds(text), self.view.lyric_times, raw=snapshot["raw"])
        self._prepare_lyrics_copy()
        self._remap_lyrics()
        self._mark_dirty()
        self.statusBar().showMessage(i18n.tr("Aligned {lines} lines", lines=len(refs)))

    def _restore_lyric_text(self, text: str) -> None:
        self.lyrics_text = text
        self._lyric_key = text_key(text) if text else ""
        self._lyric_model, self._lyric_flags, self._lyric_scores, self._lyric_problems = self.view.lyric_evidence
        if self._lyrics_directory is not None:
            self._prepare_lyrics_copy()

    def _apply_lyric_editor(self, source, text, note_ids, channel):
        if (
            source != self.lyrics_text
            or self._lyric_migration_error
            or self.view.lyric_mapping_pending
            or self.view.gesture_active
        ):
            QMessageBox.warning(
                self, i18n.tr("Edit lyrics"), i18n.tr("Lyrics changed or are busy. Reopen the editor and try again.")
            )
            return False
        try:
            lines = natural_sounds(text)
            faithful.structure(text)
        except ValueError as error:
            QMessageBox.warning(self, i18n.tr("Edit lyrics"), str(error))
            return False
        notes = [item for item in self.view.notes() if item.note.id in note_ids]
        if note_ids and (self.view._locked(channel) or any(self.view._locked(item.channel) for item in notes)):
            QMessageBox.warning(self, i18n.tr("Edit lyrics"), i18n.tr("A source or destination channel is locked."))
            return False
        with self.view.lyric_edit(i18n.tr("Edit lyrics")):
            for item in notes:
                item.note.channel = channel
            self.lyrics_text = text
            self.view.lyric_text = text
            self._lyric_key = text_key(text)
            same_sounds = [[sound.reading for sound in line.sounds] for line in lines] == [
                [sound.reading for sound in line.sounds] for line in self.view.lyric_lines
            ]
            raw = self.view.lyric_raw if same_sounds else ()
            self.view.load_lyrics(lines, (), raw=raw, operations=())
            if notes:
                self.view._sync_channel_visuals()
                self.view.notes_changed.emit()
        self._mark_dirty()
        self._remap_lyrics()
        return True

    def _set_lyric_channel(self, line: int, channel: int, *, index: int | None = None) -> None:
        if (
            self._lyric_migration_error
            or self.view.lyric_mapping_pending
            or self.view.gesture_active
            or not 0 <= line < len(self.view.lyric_lines)
            or channel not in {item.channel for item in self.view.channels}
        ):
            return
        old = self.view.lyric_lines[line].track if index is None else self.view.lyric_lines[line].sounds[index].channel
        if old == channel:
            return
        text = self.lyrics_text
        parsed = parse(text)
        rows = [row for chapter in parsed.chapters for row in chapter.lines]
        if index is None:
            rows[line].track = channel
        else:
            set_sound_channel(rows[line], index, channel)
        self._apply_lyric_editor(text, dumps(parsed, dotted=True), set(), None)

    def _begin_lyric_drag(self) -> None:
        if self._lyric_map_result is None:
            return
        self._lyric_drag_request = self._mapping_request(self.view.lyric_lines)
        self._lyric_drag_result = self._lyric_map_result
        self._lyric_drag_valid = False
        self.view.begin_gesture(i18n.tr("Move lyrics"))
        self._set_lyric_mapping_pending(False)

    def _preview_lyric_drag(self, runs) -> None:
        if self._lyric_drag_request is None:
            return
        self._lyric_map_revision += 1
        self._lyric_map_pending = None
        self._lyric_drag_valid = False
        if runs is None:
            self.lyrics_text = self._lyric_drag_request["text"]
            self._lyric_key = text_key(self.lyrics_text)
            self._apply_lyric_mapping(self._lyric_drag_result)
            self._lyric_map_result = self._lyric_drag_result
            self._set_lyric_mapping_pending(False)
            self.statusBar().showMessage(i18n.tr("No valid mapping at this position"))
            return
        try:
            baseline = note_runs(self._lyric_drag_request["text"])
            changed = set(baseline).symmetric_difference(runs)
            affected = sorted({ref.line for operation in changed for ref in operation.sounds})
            text = rebuild(self._lyric_drag_request["text"], runs, line_indices=affected)
            snapshot = dict(
                self._lyric_drag_request,
                text=text,
                lines=faithful.structure(text).lines,
                previous=self._lyric_map_result,
            )
            mapping = map_lyrics(**snapshot)
        except (KrcError, ValueError) as error:
            self.statusBar().showMessage(str(error))
            return
        self.lyrics_text = text
        self._lyric_key = text_key(text)
        self._apply_lyric_mapping(mapping)
        self._lyric_map_result = mapping
        self._lyric_drag_valid = True
        self._set_lyric_mapping_pending(False)

    def _finish_lyric_drag(self, commit: bool) -> None:
        if self._lyric_drag_request is None:
            return
        valid = commit and self._lyric_drag_valid and self.lyrics_text != self._lyric_drag_request["text"]
        baseline = self._lyric_drag_result
        self._lyric_drag_request = None
        self._lyric_drag_result = None
        if valid:
            self.view.commit_gesture()
            self._prepare_lyrics_copy()
            self._mark_dirty()
        else:
            self._invalidate_lyric_mapping()
            self._lyric_map_result = baseline
            self._apply_lyric_mapping(baseline)
            self.view.cancel_gesture()

    def _on_lyric_action(self, kind: str, line: int, index: int) -> None:
        """A Sound's context menu changes the KRC in one undo step."""
        if self._lyric_migration_error or self.view.gesture_active or not 0 <= line < len(self.view.lyric_lines):
            return
        if self.view.lyric_mapping_pending:
            self.statusBar().showMessage(i18n.tr("Calculating lyric mapping…"))
            return
        ref = SoundRef(line, index)
        runs = note_runs(self.lyrics_text)
        run = operation_at(runs, ref)
        if kind not in ("drop", "keep") or run is None or (kind == "keep" and run.slots != 0):
            return
        candidate = detach(runs, ref, 0 if kind == "drop" else 1)
        if candidate is not None:
            self._begin_lyric_drag()
            self._preview_lyric_drag(candidate)
            self._finish_lyric_drag(True)

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
        self.view.set_spectrum(None)
        self.song.unload()
        self.player.stop()
        self.transport.detect.setEnabled(False)
        self.edit.transcribe.setEnabled(False)
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
        self.autosave_timer.stop()
        self._lyric_map_revision += 1
        self._lyric_map_pending = None
        if self._lyric_map_thread is not None:
            self._lyric_map_thread.wait()
            self._lyric_map_thread = None
        for worker in tuple(self._workers):
            worker.wait()
        self._workers.clear()
        self._remember_configuration()
        self._remember_session()
        self.settings_store.flush()
        self._save_state()
        self.stop_ipc()
        self._release_lyrics_copy()
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
            self._lyric_alignment = None
            # another song brings its own tempo and offset; a re-analysis of the same one does not
            self.transport.bpm.setValue(project.FIELD_SPECS[("tempo", "bpm")].default)
            self.transport.grid_offset.setValue(project.FIELD_SPECS[("editor", "grid_offset_ms")].default)
            self._lyric_model = ""
            self._lyric_flags = ()
            self._lyric_scores = None
            self._lyric_problems = None
            self.view.lyric_evidence = ("", (), (), ())
            self.view.load_lyrics(
                self.view.lyric_lines,
                self.view.lyric_times,
                raw=[[(None, None)] * len(line.sounds) for line in self.view.lyric_lines],
            )
            self._remap_lyrics()
        self.audio_path = path
        self._audio_generation += 1
        generation = self._audio_generation
        self.edit.transcribe.setEnabled(True)
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

    def _update_lyric_scope(self) -> None:
        channels = {sound.channel for line in self.view.lyric_lines for sound in line.sounds}
        enabled = len(channels) > 1
        if not enabled:
            self.transport.lyric_scope.setChecked(False)
        self.transport.lyric_scope.setEnabled(enabled)

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
        self._show_position()

    def _stop(self) -> None:
        self.position_timer.stop()
        self.song.stop()
        self.player.stop()
        self._show_position()

    def _seek(self, seconds: float) -> None:
        self.song.seek(seconds)
        self.player.seek(seconds)
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

    def _show_hint(self) -> None:
        self.statusBar().showMessage(
            i18n.tr(
                "space: play or pause  |  click (outside edit mode): move the playhead  |  "
                "pen: drag an empty row to draw  |  select: drag a box, ctrl-click to add  |  "
                "shift drag a note: trim its start (left half) or end (right half)  |  "
                "right click: move to a channel  |  "
                "ctrl C: copy the selection, ctrl V: paste it at the playhead, ctrl D: delete it  |  "
                "ctrl Z: undo, ctrl shift Z: redo  |  "
                "middle drag: pan  |  ctrl wheel: zoom x, ctrl shift wheel: zoom y  |  gear: settings"
            )
        )

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

    def _on_spectrum_parameters(self, *_args) -> None:
        self.view.gain = self.mix.gain.value()
        self.view.contrast = self.mix.contrast.value()
        self.view.refresh()

    def _on_analysis_progress(self, generation: int, done: int, total: int) -> None:
        if generation != self._audio_generation:
            return
        self.statusBar().showMessage(i18n.tr("Analysing … {percent}%", percent=done * 100 // max(1, total)))

    def _on_spectrum_loaded(self, generation: int, spectrum: NoteSpectrum) -> None:
        if generation != self._audio_generation:
            return
        self.view.set_spectrum(spectrum)
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
    app = QApplication(sys.argv)
    window = MainWindow(overrides={"channels": args.channels, "t_num": args.t_num})
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
