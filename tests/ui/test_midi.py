# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFileDialog

from namioto import midi, project
from namioto import settings as store
from namioto.channels import Channel
from namioto.ui.app import MainWindow
from namioto.ui.midi_dialog import MidiImportDialog
from namioto.ui.open_dialog import OpenAudioDialog

from .helpers import fake_loaders


def accept_import(monkeypatch, mode: str, mapping=()) -> None:
    """Answer the import dialog without one: the flow under test is the import, not the widget."""
    monkeypatch.setattr(MidiImportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(MidiImportDialog, "mode", lambda self: mode)
    monkeypatch.setattr(MidiImportDialog, "mapping", lambda self: list(mapping))


def test_importing_a_midi_brings_in_its_notes_and_channels(midi_window, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=0, program=52), Channel(channel=1)),
        (project.Note(1.0, 0.5, 60, 0), project.Note(1.0, 0.5, 48, 1)),
        120.0,
    )
    assert midi_window.import_midi(path) is True

    assert [channel.channel for channel in midi_window.view.channels] == [0, 1]
    assert [channel.program for channel in midi_window.view.channels] == [52, 0]
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(48, 1), (60, 0)]
    assert midi_window.transport.bpm.value() == 120.0
    assert midi_window.project_dirty is True
    assert "Imported 2 notes" in midi_window.statusBar().currentMessage()


def test_a_midi_without_a_project_is_refused(own_window, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 60, 0),), 120.0)

    assert own_window.import_midi(path) is False
    assert own_window.view.notes() == []
    assert "Open a song first" in own_window.statusBar().currentMessage()


def test_importing_over_notes_asks_before_replacing(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_notes([(60, 0.0, 1.0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    monkeypatch.setattr(MidiImportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

    assert midi_window.import_midi(path) is False
    assert [note.pitch for note in midi_window.view.notes()] == [60]  # nothing was touched


def test_replacing_is_what_the_dialog_can_choose(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Old"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    accept_import(monkeypatch, "replace")

    assert midi_window.import_midi(path) is True
    # a replacement is the file's channels whole: a name the project gave the old one does not stay
    assert [channel.channel for channel in midi_window.view.channels] == [0]
    assert [channel.name for channel in midi_window.view.channels] == [""]
    assert [note.pitch for note in midi_window.view.notes()] == [62]


def test_merging_adds_the_file_channels_to_the_roll(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=0), Channel(channel=2, program=33, volume=90)),
        (project.Note(1.0, 0.5, 64, 0), project.Note(1.0, 0.5, 40, 2)),
        140.0,
    )
    accept_import(monkeypatch, "merge", [0, -1])

    assert midi_window.import_midi(path) is True
    assert [channel.channel for channel in midi_window.view.channels] == [0, 2]
    assert (midi_window.view.channels[1].program, midi_window.view.channels[1].volume) == (33, 90)
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(40, 2), (60, 0), (64, 0)]
    # the file's 140 BPM does not touch the grid: a second stays a second on the audio (120 BPM here)
    assert sorted((note.pitch, round(note.start, 3)) for note in midi_window.view.notes()) == [
        (40, 2.0),
        (60, 0.0),
        (64, 2.0),
    ]
    assert midi_window.transport.bpm.value() == 120.0  # the grid stays on the audio, not on the file


def test_merging_onto_an_empty_channel_takes_the_file_channel_over(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=5, name="Placeholder"), Channel(channel=0, name="Used")))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])  # channel 5 carries nothing
    path = tmp_path / "song.mid"
    midi.write(
        path,
        (Channel(channel=2, program=81, volume=90),),
        (project.Note(1.0, 0.5, 62, 2),),
        120.0,
    )
    accept_import(monkeypatch, "merge", [5])

    assert midi_window.import_midi(path) is True
    # an empty channel is a free place: the file's channel takes it over, the way a brand new one would
    assert [channel.channel for channel in midi_window.view.channels] == [0, 5]
    landed = next(channel for channel in midi_window.view.channels if channel.channel == 5)
    assert (landed.program, landed.volume) == (81, 90)
    assert landed.name == "Placeholder"  # the name belongs to the project, and it stays
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 5)]


def test_a_new_channel_keeps_the_number_the_file_played_on(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=6),), (project.Note(1.0, 0.5, 62, 6),), 120.0)
    accept_import(monkeypatch, "merge", [-1])

    assert midi_window.import_midi(path) is True
    # nothing else plays on channel 6, so the file's new channel keeps that number
    assert [channel.channel for channel in midi_window.view.channels] == [0, 6]
    assert midi_window.view.channels[1].name == ""  # a MIDI channel carries no name
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 6)]


def test_a_new_channel_takes_a_free_number_when_the_files_own_is_taken(midi_window, monkeypatch, tmp_path) -> None:
    midi_window.view.set_channels((Channel(channel=0, name="Voice"),))
    midi_window.view.set_notes([(60, 0.0, 1.0, 0)])
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 62, 0),), 120.0)
    accept_import(monkeypatch, "merge", [-1])

    assert midi_window.import_midi(path) is True
    # channel 0 already carries the roll's notes, so the file's new channel takes the lowest free one
    assert [channel.channel for channel in midi_window.view.channels] == [0, 1]
    assert sorted((note.pitch, note.channel) for note in midi_window.view.notes()) == [(60, 0), (62, 1)]


def test_the_import_dialog_prefills_the_mapping_by_position() -> None:
    imported = midi.Imported(channels=(Channel(channel=2), Channel(channel=7)), notes=(), bpm=120.0)
    dialog = MidiImportDialog(imported, (Channel(channel=0), Channel(channel=2)), "song.mid")
    assert dialog.mapping() == [0, 2]  # both roll channels carry no notes, so the file lands on them in order
    assert dialog.mode() == "merge"  # the additive choice is what a bare accept takes
    dialog.close()


def test_a_file_channel_moves_past_a_roll_channel_that_already_has_notes() -> None:
    imported = midi.Imported(channels=(Channel(channel=0), Channel(channel=1), Channel(channel=2)), notes=(), bpm=120.0)
    channels = tuple(Channel(name=name, channel=index) for index, name in enumerate("ABCD"))
    # A and C carry notes, so the file's first channel takes the empty place after the second's own
    dialog = MidiImportDialog(imported, channels, "song.mid", occupied={0, 2})
    assert dialog.mapping() == [3, 1, -1]
    dialog.close()


def test_the_import_dialog_keeps_the_roll_within_sixteen_channels() -> None:
    imported = midi.Imported(channels=(Channel(channel=15),), notes=(), bpm=120.0)
    # every channel carries notes, so nothing is free to land on and the mapping reaches for a new one
    channels = tuple(Channel(name=f"T{index}", channel=channel) for index, channel in enumerate([0, *range(15)]))
    dialog = MidiImportDialog(imported, channels, "song.mid", occupied=range(16))
    assert dialog.mapping() == [-1]
    assert dialog.merge_button.isEnabled() is False  # a new channel would be the seventeenth
    dialog._targets[0].setCurrentIndex(0)  # pointed at an existing channel instead
    assert dialog.merge_button.isEnabled() is True
    dialog.close()


def test_a_wavetone_file_loses_its_lead_in_only_when_the_setting_says_so(midi_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "wavetone.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(1.0, 0.5, 60, 0),), 120.0, wavetone=True)

    midi_window.import_midi(path)
    assert [round(note.start, 3) for note in midi_window.view.notes()] == [2.0]

    store.set_value(midi_window.settings, "midi", "wavetone", False)
    accept_import(monkeypatch, "replace")  # the roll holds a note now, so the dialog would stand in the way
    midi_window.import_midi(path)
    assert [round(note.start, 3) for note in midi_window.view.notes()] == [6.0]


def test_a_midi_that_cannot_be_read_says_so(midi_window, tmp_path) -> None:
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"not a MIDI file at all")
    assert midi_window.import_midi(broken) is False
    assert "could not be read" in midi_window.statusBar().currentMessage()


def test_exporting_writes_the_roll_out_as_midi(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=2, program=81),))
    own_window.view.set_notes(((60, 0.0, 1.0, 2), (64, 1.0, 0.5, 2)))
    path = tmp_path / "out.mid"
    assert own_window.export_midi(path) is True

    imported = midi.read(path, wavetone=True)  # the WaveTone compatibility the settings start with
    assert [(note.pitch, round(note.start, 3)) for note in imported.notes] == [(60, 0.0), (64, 0.5)]
    assert [(channel.channel, channel.program) for channel in imported.channels] == [(2, 81)]
    assert [round(note.start, 3) for note in midi.read(path).notes] == [2.0, 2.5]  # the same, a bar late
    assert own_window.project_path is None  # exporting is not saving
    assert "Exported out.mid" in own_window.statusBar().currentMessage()


def test_a_hidden_channel_is_exported_like_any_other(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1, visible=False)))
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 0.0, 1.0, 1)))
    path = tmp_path / "hidden.mid"
    own_window.export_midi(path)

    assert [note.pitch for note in midi.read(path).notes] == [60, 62]


def test_opening_a_midi_file_imports_it(midi_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "song.mid"
    midi.write(path, (Channel(channel=0),), (project.Note(0.5, 0.5, 60, 0),), 120.0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

    midi_window._on_open()
    assert [note.pitch for note in midi_window.view.notes()] == [60]


def test_the_open_dialog_shows_every_file_type_by_default(own_window, monkeypatch) -> None:
    shown: list[str] = []

    def fake(_parent, _caption, _directory, filters, *_args, **_kwargs):
        shown.append(filters)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)
    own_window._on_open()

    assert shown[0].split(";;")[0] == "All files (*)"
    # a MIDI is a project's, so without one the chooser does not offer it
    assert "MIDI" not in shown[0]


def test_the_open_dialog_offers_a_midi_once_a_project_is_open(midi_window, monkeypatch) -> None:
    shown: list[str] = []

    def fake(_parent, _caption, _directory, filters, *_args, **_kwargs):
        shown.append(filters)
        return ("", "")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)

    midi_window._on_open()

    assert "MIDI" in shown[0]


def test_opening_an_audio_file_analyses_it(own_window, monkeypatch, tmp_path) -> None:
    path = tmp_path / "song.flac"
    path.write_bytes(b"")
    fake_loaders(monkeypatch)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    own_window._on_open()

    assert own_window.audio_path == str(path)
    assert own_window.project_path == tmp_path / "song.nto"  # a sound is a project, and it has a file


def open_editor(dialog, name: str):
    """The widget one row of the open dialog edits, to change it the way a user would."""
    return next(widget for field, widget, _read in dialog._rows if field == name)


def test_the_open_dialog_starts_from_the_document_and_the_audio(own_window, tmp_path) -> None:
    document = project.default_settings()
    document.analysis.channels = "left"
    document.analysis.fft_points = 4096
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), document, parent=own_window)
    try:
        assert dialog.path.text() == str(tmp_path / "song.nto")
        assert dialog.document().analysis.channels == "left"
        assert dialog.document().analysis.fft_points == 4096
    finally:
        dialog.close()


def test_the_open_dialog_reads_an_edited_widget_back(own_window, tmp_path) -> None:
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), project.default_settings(), parent=own_window)
    try:
        open_editor(dialog, "a4").setValue(442.0)
        assert dialog.document().analysis.a4 == 442.0
    finally:
        dialog.close()


def test_the_open_dialog_adds_the_project_suffix(own_window, tmp_path) -> None:
    dialog = OpenAudioDialog(str(tmp_path / "song.flac"), project.default_settings(), parent=own_window)
    try:
        dialog.path.setText(str(tmp_path / "mysong"))
        assert dialog.target() == tmp_path / "mysong.nto"
    finally:
        dialog.close()


def test_opening_audio_keeps_the_analysis_habits_the_user_changed(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)

    def accept(self):
        open_editor(self, "fft_points").setValue(4096)
        open_editor(self, "a4").setValue(442.0)  # the song's tuning is not a habit
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(OpenAudioDialog, "exec", accept)

    assert own_window.open_audio(str(audio)) is True

    assert own_window.state.project["analysis"] == {"fft_points": 4096}
    assert own_window.project_settings.analysis.fft_points == 4096
    assert own_window.project_settings.analysis.a4 == 442.0


def test_the_command_line_analysis_values_start_the_dialog_and_stop_there(own_window, monkeypatch, tmp_path) -> None:
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"")
    fake_loaders(monkeypatch)
    own_window.overrides["channels"] = "left"
    seen: list[str] = []

    def accept(self):
        seen.append(self.document().analysis.channels)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(OpenAudioDialog, "exec", accept)

    assert own_window.open_audio(str(audio)) is True

    assert seen == ["left"]  # the window opened on the value the command line asked for
    assert own_window.overrides == {}  # and the document owns it from there
    assert own_window.project_settings.analysis.channels == "left"


def test_the_command_line_takes_audio_and_projects_but_not_a_midi_alone(qt_app, monkeypatch, tmp_path) -> None:
    fake_loaders(monkeypatch)
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"")
    midi_path = tmp_path / "tune.mid"
    midi.write(midi_path, (Channel(channel=0),), (project.Note(0.5, 0.5, 60, 0),), 120.0)
    project_path = tmp_path / "work.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        project_path,
    )
    monkeypatch.setattr(OpenAudioDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    opened = [MainWindow(), MainWindow(), MainWindow()]
    for window, path in zip(opened, (audio, midi_path, project_path), strict=True):
        window.open_file(str(path))
    try:
        assert opened[0].audio_path == str(audio)
        assert opened[0].project_path == tmp_path / "clip.nto"  # opening a sound names its project
        # a MIDI is part of a project, so it is not a document the command line can open on its own
        assert opened[1].project_path is None
        assert [note.pitch for note in opened[1].view.notes()] == []
        assert opened[2].project_path == project_path
        assert [note.pitch for note in opened[2].view.notes()] == [62]
    finally:
        for window in opened:
            window.project_dirty = False
            window.close()


def test_the_export_button_writes_a_midi_file(own_window, monkeypatch, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.set_notes(((60, 0.0, 1.0, 0),))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "MIDI file (*.mid)")
    )

    assert own_window._on_export_midi() is True
    target = tmp_path / "exported.mid"  # the export adds its own suffix
    assert [note.pitch for note in midi.read(target, wavetone=True).notes] == [60]
    assert own_window.project_path is None  # an export leaves the document where it was
