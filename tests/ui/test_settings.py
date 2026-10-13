# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QLabel, QTabWidget

from namioto import project
from namioto import settings as store
from namioto import state as window_state
from namioto.analysis import devices
from namioto.interaction import Interaction, Tool
from namioto.ui.app import MainWindow, TempoLoader
from namioto.ui.audio import find_port
from namioto.ui.controls import EditBar, TransportBar
from namioto.ui.form import WrappedLabel, _show_device_status, field_editor
from namioto.ui.roll import SNAP_CHOICES
from namioto.ui.settings_dialog import PAGES, SettingsDialog
from namioto.ui.spectrogram import SpectrumLoader

from .helpers import fake_loaders


def row_writer(dialog, section: str, name: str):
    """The write half of one row of the settings window, to change it the way a widget would."""
    for row_section, field, _read, write in dialog._rows:
        if (row_section, field.name) == (section, name):
            return write
    raise AssertionError(f"the settings window has no row for {section}.{name}")


def test_the_gear_button_opens_the_settings_window(own_window, monkeypatch) -> None:
    opened: list[SettingsDialog] = []
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: opened.append(self) or 0)
    own_window.transport.settings_button.click()
    assert len(opened) == 1
    assert opened[0].parent() is own_window


def test_the_settings_window_lists_every_visible_field(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    names = {(section, field.name) for section, field, _read, _write in dialog._rows}
    expected = {
        (section.name, field.name) for section in store.SECTIONS if section.name in PAGES for field in section.fields
    }
    assert names == expected  # `editor` has no page: its switches are what the program remembers itself
    pages = [dialog.findChild(QTabWidget).tabText(index) for index in range(dialog.findChild(QTabWidget).count())]
    assert pages == ["General", "Devices", "Tempo", "Lyrics", "Network", "MIDI", "Remote"]
    dialog.close()


def test_applying_the_settings_window_reaches_the_window_and_the_file(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    dialog.applied.connect(own_window.settings_store.apply)  # the window wires this up when it opens it
    row_writer(dialog, "tempo", "window_seconds")(20.0)
    row_writer(dialog, "midi", "wavetone")(False)
    dialog.apply()

    assert own_window.settings.tempo.window_seconds == 20.0
    assert own_window.settings.midi.wavetone is False
    saved = json.loads(store.default_path().read_text())
    assert saved["tempo"]["window_seconds"] == 20.0
    assert saved["midi"]["wavetone"] is False
    dialog.close()


def test_the_analysis_options_are_a_project_s_so_the_window_has_no_row(own_window) -> None:
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    assert not [row for row in dialog._rows if row[0] == "analysis"]
    dialog.close()


def test_a_wrapped_status_line_grows_to_fit_its_text(qt_app) -> None:
    label = WrappedLabel()
    label.setText("没有可用的 GPU 后端；选择 GPU 时会回退到 CPU")
    label.setFixedWidth(200)
    label.show()
    qt_app.processEvents()

    needed = label.heightForWidth(200)
    assert needed > label.fontMetrics().lineSpacing()  # the text really does wrap
    assert label.minimumHeight() == needed  # and the row is forced tall enough for it
    label.close()


def test_the_gpu_row_says_whether_its_runtime_is_installed(qt_app, monkeypatch) -> None:
    monkeypatch.setattr(devices, "installed", lambda: ("CPUExecutionProvider",))
    field = store.FIELD_SPECS[("devices", "gpu")]
    combo, _read, write = field_editor(store.Settings().devices.gpu, field)
    status = QLabel()

    _show_device_status(combo, status)
    assert "No GPU backend is available" in status.text()

    write("cuda")
    _show_device_status(combo, status)
    assert "NVIDIA CUDA is not available" in status.text()

    monkeypatch.setattr(devices, "installed", lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"))
    _show_device_status(combo, status)
    assert "NVIDIA CUDA is available" in status.text()


def test_restoring_defaults_puts_every_widget_back(own_window) -> None:
    own_window.settings.ass.font_size = 72
    dialog = SettingsDialog(own_window.settings, parent=own_window)
    row_writer(dialog, "tempo", "window_seconds")(8.0)
    row_writer(dialog, "midi", "wavetone")(False)
    assert store.get_value(dialog.values(), "midi", "wavetone") is False

    dialog.restore_defaults()
    values = dialog.values()
    assert store.get_value(values, "midi", "wavetone") is True
    assert store.get_value(values, "tempo", "window_seconds") == 12.0
    assert values.ass.font_size == 72
    dialog.close()


def test_closing_the_window_remembers_its_state_and_the_document(own_window) -> None:
    own_window.view.set_zoom(96.0, 20.0)
    own_window.close()

    saved = window_state.load()
    assert saved.geometry
    assert own_window.project_settings.editor.zoom_x == 96.0
    assert own_window.project_settings.editor.zoom_y == 20.0
    assert own_window.project_settings.view.center_x > 0.0


def test_a_new_window_starts_from_the_remembered_defaults() -> None:
    window_state.save(window_state.State(project={"spectrum": {"gain": 300.0}, "editor": {"zoom_x": 96.0}}))
    opened = MainWindow()
    assert opened.mix.gain.value() == 300.0
    assert opened.view.zoom == (96.0, 16.0)
    opened.project_dirty = False
    opened.close()


def test_an_explicit_change_is_what_the_next_project_inherits(own_window) -> None:
    own_window.mix.gain.set_value(300.0)
    own_window.transport.grid_offset.setValue(120)
    own_window.view._zoom(1.5, 1.5, QPoint(0, 0))  # a wheel zoom is the user's; set_zoom is not
    own_window.close()

    saved = window_state.load().project
    assert saved["spectrum"]["gain"] == 300.0
    assert saved["editor"]["zoom_x"] == 72.0  # 48 * 1.5
    assert saved["editor"]["zoom_y"] == 24.0  # 16 * 1.5
    assert "grid_offset_ms" not in saved["editor"]  # the grid offset belongs to the song too
    assert "view" not in saved


def test_opening_a_project_leaves_the_remembered_defaults_alone(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.spectrum.gain = 300.0
    other.editor.zoom_x = 96.0
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)
    assert own_window.load_project(path) is True
    own_window.close()

    assert window_state.load().project == {}


def test_the_preferences_are_read_when_the_window_starts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(path))
    saved = store.Settings()
    store.set_value(saved, "editor", "auto_page", True)
    store.save(saved)

    opened = MainWindow()
    assert opened.transport.auto_page.isChecked() is True
    opened.close()


def test_a_new_document_starts_from_the_document_defaults(own_window) -> None:
    document = project.default_settings()
    assert own_window.view.contrast == document.spectrum.contrast
    assert own_window.view.snap == document.editor.snap
    assert own_window.transport.bpm.value() == document.tempo.bpm
    assert own_window.view.zoom == (document.editor.zoom_x, document.editor.zoom_y)


def test_the_command_line_seeds_the_run_without_writing_itself_back(own_window) -> None:
    own_window.apply_overrides(gain=300.0, contrast=2.0)
    assert own_window.view.gain == 300.0
    own_window.close()

    document = project.default_settings()
    assert own_window.project_settings.spectrum.gain == document.spectrum.gain
    assert own_window.project_settings.spectrum.contrast == document.spectrum.contrast


def test_every_bar_setting_has_one_binding(own_window) -> None:
    bound = {(binding.section, binding.name) for binding in own_window._bindings}
    # the document's values are all on a bar, save the ones the roll, the window and the command line own
    elsewhere = {
        ("editor", "zoom_x"),
        ("editor", "zoom_y"),
        ("view", "center_x"),
        ("view", "center_y"),
        ("analysis", "channels"),
        ("analysis", "t_num"),
        ("analysis", "fft_points"),
        ("analysis", "a4"),
    }
    assert set(project.FIELD_SPECS) - bound == elsewhere
    # the two switches with no page are the program's, and are still on the bars
    assert {("editor", "auto_page"), ("editor", "overtone_highlight")} <= bound


def test_a_command_line_channel_beats_the_document(own_window) -> None:
    own_window.project_settings.analysis.channels = "side"
    own_window.overrides["channels"] = "left"
    own_window.overrides["t_num"] = None
    options = own_window._analysis_options()
    assert options["channels"] == "left"
    assert options["t_num"] == 40.0
    assert options["fft_points"] == 8192
    assert options["a4"] == 440.0


def test_the_spectrum_loader_takes_every_analysis_parameter(tmp_path) -> None:
    loader = SpectrumLoader(tmp_path / "song.wav", channels="side", t_num=25.0, fft_points=2048, a4=442.0)
    assert (loader.channels, loader.t_num, loader.fft_points, loader.a4) == ("side", 25.0, 2048, 442.0)
    assert loader.path == tmp_path / "song.wav"


def test_loading_a_file_hands_the_document_to_the_analysers(own_window, monkeypatch) -> None:
    captured = fake_loaders(monkeypatch)
    own_window.project_settings.analysis.fft_points = 4096
    own_window.project_settings.analysis.a4 = 441.0
    store.set_value(own_window.settings, "tempo", "window_seconds", 8.0)
    own_window.overrides["channels"] = "left"

    own_window.load_audio("/tmp/song.wav")

    analysis, _song, tempo = captured
    assert analysis == {"channels": "left", "t_num": 40.0, "fft_points": 4096, "a4": 441.0}
    assert tempo == {"algorithm": "wavetone", "window_seconds": 8.0, "window_hop_seconds": 6.0}
    assert own_window.state.last_audio_dir == "/tmp"
    assert own_window.edit.transcribe.isEnabled()


def test_the_tempo_loader_follows_the_settings(own_window, monkeypatch) -> None:
    own_window.audio_path = "song.wav"
    store.set_value(own_window.settings, "tempo", "estimator", "librosa")
    store.set_value(own_window.settings, "tempo", "window_seconds", 8.0)
    store.set_value(own_window.settings, "tempo", "window_hop_seconds", 3.0)
    monkeypatch.setattr(TempoLoader, "start", lambda self: None)
    own_window._start_tempo()
    assert own_window.tempo_loader.algorithm == "librosa"
    assert own_window.tempo_loader.window_seconds == 8.0
    assert own_window.tempo_loader.window_hop_seconds == 3.0


def test_a_song_tempo_never_reaches_the_preferences(own_window) -> None:
    own_window.transport.bpm.setValue(93.0)
    own_window.transport.grid_offset.setValue(120)
    own_window.close()

    saved = store.load()
    assert not hasattr(saved, "playback")  # the offset is the document's, and it is not written here
    assert not hasattr(saved.tempo, "bpm")
    assert own_window.project_settings.tempo.bpm == 93.0


def test_another_song_starts_from_the_default_tempo_and_grid_offset(own_window, monkeypatch) -> None:
    fake_loaders(monkeypatch)
    own_window.transport.bpm.setValue(93.0)
    own_window.transport.grid_offset.setValue(120)

    own_window.load_audio("/tmp/other.wav")
    assert own_window.transport.bpm.value() == 120.0
    assert own_window.transport.grid_offset.value() == 0

    own_window.transport.bpm.setValue(93.0)
    own_window.load_audio("/tmp/other.wav")  # the same file again: a re-analysis keeps what was typed
    assert own_window.transport.bpm.value() == 93.0


def test_a_project_brings_its_tempo_and_grid_offset_back(own_window, tmp_path) -> None:
    other = project.default_settings()
    other.tempo.bpm = 93.0
    other.editor.grid_offset_ms = 120
    path = tmp_path / "song.nto"
    project.save(project.Project(settings=other), path)

    assert own_window.load_project(path) is True
    assert own_window.transport.bpm.value() == 93.0
    assert own_window.transport.grid_offset.value() == 120
    own_window.settings_store.flush()
    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert not hasattr(own_window.settings, "playback")
    assert "bpm" not in written["tempo"]  # the song's tempo stays in the document


def test_the_overtone_highlight_can_be_turned_off(window) -> None:
    window.edit.set_interaction(Interaction.viewing())
    window.view.overtone_highlight = True
    window.view.set_hover_pitch(60)
    assert window.view.highlight_pitches() == [60, 72, 79, 84]  # no edit mode needed

    window.view.overtone_highlight = False
    assert window.view.highlight_pitches() == [60]

    window.edit.set_interaction(Interaction.editing_with(Tool.PEN))
    assert window.view.highlight_pitches() == [60]  # and the switch still governs it while editing
    window.edit.set_interaction(Interaction.viewing())
    window.view.set_hover_pitch(None)


def test_the_zoom_can_be_set_from_outside(window) -> None:
    window.view.set_zoom(96.0, 32.0)
    assert window.view.zoom == (96.0, 32.0)
    window.view.set_zoom(1.0, 500.0)  # both ends are held to what the roll can draw
    assert window.view.zoom == (window.view.MIN_ZOOM_X, window.view.MAX_ZOOM_Y)
    window.view.set_zoom(48.0, 16.0)


def test_a_midi_port_is_matched_by_name() -> None:
    ports = ("Midi Through:Midi Through Port-0 14:0", "TiMidity:TiMidity port 0 128:0")
    assert find_port(ports, "TiMidity:TiMidity port 0 128:0") == 1
    assert find_port(ports, "TiMidity") == 1  # the client number changes between sessions
    assert find_port(ports, "Nonesuch") == 1  # nothing matches, so the first synth is used
    assert find_port(("Midi Through:0",), "") is None


def test_the_transport_carries_the_project_buttons() -> None:
    bar = TransportBar()
    seen: list[str] = []
    bar.open_requested.connect(lambda: seen.append("open"))
    bar.save_requested.connect(lambda: seen.append("save"))
    bar.export_midi_requested.connect(lambda: seen.append("export midi"))
    bar.export_krc_requested.connect(lambda: seen.append("export lyrics"))
    bar.export_ass_requested.connect(lambda: seen.append("export subtitle"))
    bar.open.click()
    bar.save.click()
    bar.export_midi_action.trigger()
    bar.export_krc_action.trigger()
    bar.export_ass_action.trigger()
    assert seen == ["open", "save", "export midi", "export lyrics", "export subtitle"]


def test_the_edit_bar_carries_the_wand() -> None:
    bar = EditBar(SNAP_CHOICES)
    bar.transcribe.setEnabled(True)  # no audio in this bar, so it starts off
    seen: list[str] = []
    bar.transcribe_requested.connect(lambda: seen.append("transcribe"))
    bar.transcribe.click()
    assert seen == ["transcribe"]
