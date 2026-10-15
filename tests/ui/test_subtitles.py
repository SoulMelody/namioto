# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QFileDialog

from namioto import settings as store
from namioto.karaoke import AssSettings
from namioto.ui.ass_dialog import GROUPS as ASS_GROUPS
from namioto.ui.ass_dialog import AssDialog


def test_edit_mode_ass_matches_its_persisted_krc_and_midi(subtitle_window, tmp_path):
    from namioto import subtitle

    window = subtitle_window
    assert not hasattr(window, "_lyric_mode")
    krc_path, midi_path = tmp_path / "current.krc", tmp_path / "current.mid"
    assert window.export_ass(tmp_path / "current.ass", save_krc=krc_path, save_midi=midi_path)
    assert window.export_krc(tmp_path / "separate.krc")
    assert window.export_midi(tmp_path / "separate.mid")
    assert krc_path.read_bytes() == (tmp_path / "separate.krc").read_bytes()
    assert midi_path.read_bytes() == (tmp_path / "separate.mid").read_bytes()
    expected = subtitle.build_ass(
        krc_path.read_text(),
        midi_path.read_bytes(),
        settings=window._ass_settings(),
        wavetone=window.settings.midi.wavetone,
    )
    assert (tmp_path / "current.ass").read_text() == expected


def test_the_export_button_writes_an_ass_subtitle(subtitle_window, monkeypatch, tmp_path) -> None:
    own_window = subtitle_window

    def options(dialog):
        dialog.editors["font_size"].setValue(72)
        dialog.editors["offset_ms"].setValue(1000)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AssDialog, "exec", options)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "exported"), "ASS subtitle (*.ass)")
    )
    before = own_window.view.notes()
    assert own_window._on_export_ass() is True
    target = tmp_path / "exported.ass"  # the export adds its own suffix
    text = target.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "Dialogue: 0," in text
    assert "Style: K1,sans-serif,72," in text
    assert "0:00:03.00,K1" in text
    assert own_window.view.notes() == before
    assert own_window.settings.ass.font_size == 72
    assert store.load().ass.font_size == 72
    assert store.load().ass.offset_ms == 1000
    assert own_window.project_path == tmp_path / "song.nto"  # an export leaves the document where it was


@pytest.mark.parametrize("stop", ("options", "path", "write", "build"))
def test_cancelled_or_failed_subtitle_export_does_not_remember_options(subtitle_window, monkeypatch, tmp_path, stop):
    original = store.to_dict(subtitle_window.settings)
    saved = store.default_path().read_bytes() if store.default_path().exists() else None
    seen = []

    def options(dialog):
        seen.append("options")
        dialog.editors["font_size"].setValue(24)
        return QDialog.DialogCode.Rejected if stop == "options" else QDialog.DialogCode.Accepted

    def path(*args, **kwargs):
        seen.append("path")
        return ("" if stop == "path" else str(tmp_path / "out.ass"), "")

    def fail(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(AssDialog, "exec", options)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", path)
    if stop == "write":
        monkeypatch.setattr("namioto.subtitle.write_text", fail)
    if stop == "build":

        def fail_build(*args, **kwargs):
            raise ValueError("invalid subtitle")

        monkeypatch.setattr("namioto.subtitle.generate_ass", fail_build)
    assert not subtitle_window._on_export_ass()
    assert seen == (["options"] if stop == "options" else ["options", "path"])
    assert store.to_dict(subtitle_window.settings) == original
    assert (store.default_path().read_bytes() if store.default_path().exists() else None) == saved
    assert not (tmp_path / "out.ass").exists()


def test_ass_options_cover_the_single_track_settings_without_mutating_them(own_window):
    original = store.to_dict(own_window.settings)
    dialog = AssDialog(own_window.settings, own_window)
    assert set(dialog.editors) == set(vars(own_window.settings.ass)) - {"overlay_color", "channel_colors"}
    assert "overlay_color" not in dialog.editors
    assert "channel_colors" not in dialog.editors
    assert set(dialog.editors) == {name for _, names in ASS_GROUPS for name in names}
    assert dialog.export_button.isEnabled()
    dialog.editors["font_size"].setValue(40)
    dialog.reject()
    assert store.to_dict(own_window.settings) == original


def test_ass_automatic_colours_follow_the_overlay_and_manual_blur(own_window):

    colors = ("#FF0000",) * 16
    dialog = AssDialog(own_window.settings, own_window, lyric_colors=colors)
    expected_overlay = AssSettings(overlay_color="FF0000").blur_colours()[0]
    assert dialog.editors["overlay_blur_color"].text() == expected_overlay
    assert dialog.values().ass.overlay_blur_color == ""
    assert not dialog.editors["overlay_blur_color"].isEnabled()
    dialog.automatic["overlay_blur_color"].setChecked(False)
    assert dialog.editors["overlay_blur_color"].isEnabled()
    dialog.editors["overlay_blur_color"].setText("123456")
    expected = AssSettings(overlay_color="FF0000", overlay_blur_color="123456").blur_colours()[1]
    assert dialog.editors["base_blur_color"].text() == expected
    assert dialog.values().ass.overlay_blur_color == "123456"
    assert dialog.values().ass.base_blur_color == ""
    dialog.restore_defaults()
    assert vars(dialog.values().ass) == vars(store.Settings().ass)
    assert all(check.isChecked() for check in dialog.automatic.values())


def test_ass_numeric_options_enforce_their_lower_bounds(own_window):
    dialog = AssDialog(own_window.settings, own_window)
    for name in ("font_size", "guide_dot_duration_ms"):
        dialog.editors[name].setValue(0)
        assert dialog.editors[name].value() == 1
    for name in ("border", "border_furi", "margin_h", "margin_v", "blur", "blur_scale", "clip_size"):
        dialog.editors[name].setValue(-1)
        assert dialog.editors[name].value() == 0
    for name in ("ruby_offset", "offset_ms"):
        dialog.editors[name].setValue(-10)
        assert dialog.editors[name].value() == -10


def test_ass_options_validate_and_pick_rgb_colours(own_window, monkeypatch):
    from PySide6.QtWidgets import QColorDialog

    dialog = AssDialog(own_window.settings, own_window)
    dialog.editors["lead_time_ms"].setValue(0)
    assert not dialog.export_button.isEnabled()
    assert "Fade in" in dialog.hint.text()
    dialog._accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.editors["fade_in_ms"].setValue(0)
    assert dialog.export_button.isEnabled()
    dialog.editors["base_outline_color"].setText("invalid")
    assert not dialog.export_button.isEnabled()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#12ABEF"))
    dialog._choose_colour(dialog.editors["base_outline_color"])
    assert dialog.editors["base_outline_color"].text() == "12ABEF"
    assert dialog.export_button.isEnabled()
    dialog.editors["font"].setText("a,b")
    assert not dialog.export_button.isEnabled()
    dialog.restore_defaults()
    assert dialog.export_button.isEnabled()


def test_a_subtitle_needs_lyrics_and_notes(own_window) -> None:
    assert own_window._on_export_ass() is False
    assert "no lyrics" in own_window.statusBar().currentMessage()
    own_window.lyrics_text = "あ"
    own_window._load_sounds()  # the lyrics are readable; now the subtitle needs a note to sit on
    assert own_window._on_export_ass() is False
    assert "no notes" in own_window.statusBar().currentMessage()
