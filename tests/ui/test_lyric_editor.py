from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QCheckBox, QInputDialog, QMenu, QMessageBox, QPushButton

from namioto.karaoke.sounds import natural_sounds
from namioto.ui.lyric_editor import reading, unit_rects

from .helpers import wait_for_lyric_mapping
from .test_lyric_channels import _mixed_lyrics


def test_panel_moves_lines_and_notes_in_one_undo_step(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)
    before = own_window.lyrics_text
    panel.apply("channel", 2, sync=True)
    wait_for_lyric_mapping(own_window)
    assert natural_sounds(own_window.lyrics_text)[0].track == 2
    assert next(item for item in own_window.view.notes() if item.note.id == 10).channel == 2
    own_window.view.undo_stack.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == before
    assert next(item for item in own_window.view.notes() if item.note.id == 10).channel == 0


def test_panel_color_does_not_move_lyrics(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)
    panel.apply("color", 2)
    wait_for_lyric_mapping(own_window)
    sound = natural_sounds(own_window.lyrics_text)[0].sounds[0]
    assert sound.channel == 0
    assert sound.color_channel == 2


def test_partial_selection_moves_only_selected_text(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "あいう", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    panel.canvas.selected = {1}
    panel.apply("channel", 2)
    wait_for_lyric_mapping(own_window)
    assert [sound.channel for sound in natural_sounds(own_window.lyrics_text)[0].sounds] == [0, 2, 0]


def test_grouped_ruby_remains_visible_and_moves_with_base(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "今日[きょう]あ", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    panel.canvas.selected = {0}
    assert reading(panel.canvas.units()[0]) == "きょう"
    assert panel.targets() == [(0, 0)]
    assert panel.canvas.height() == 64
    assert not panel.findChildren(QPushButton)
    panel.apply("channel", 2)
    wait_for_lyric_mapping(own_window)
    assert [sound.channel for sound in natural_sounds(own_window.lyrics_text)[0].sounds] == [2, 2, 0]


def test_shared_note_prevents_sync_move(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "あ.1い.+", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    panel.canvas.selected = {0}
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    before = own_window.lyrics_text
    panel.apply("channel", 2, sync=True)
    assert own_window.lyrics_text == before
    assert len(warnings) == 1


def test_ruby_edit_is_inline_and_undoable(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    own_window.show()
    panel = own_window.lyric_editor
    panel.open_at(0)
    before = own_window.lyrics_text
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: questions.append(args) or QMessageBox.StandardButton.Yes)
    panel.canvas.begin_edit(0, True)
    assert panel.canvas.editor.isVisible()
    panel.canvas.editor.setText("あい")
    panel.canvas.finish_edit()
    wait_for_lyric_mapping(own_window)
    assert reading(panel.canvas.units()[0]) == "あい"
    assert questions
    own_window.view.undo_stack.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == before


def test_inline_edit_escape_discards_change(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)
    before = own_window.lyrics_text
    panel.canvas.begin_edit(0, True)
    panel.canvas.editor.setText("あい")
    panel.canvas.cancel_edit()
    panel.canvas.finish_edit()
    assert own_window.lyrics_text == before


def test_double_click_ruby_in_list_edits_at_clicked_row(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)
    row_rect = panel.lines.visualItemRect(panel.lines.item(0))
    rect = unit_rects(panel.rows[0].units, panel.lines.font())[0]
    pos = QPointF(row_rect.x() + rect.center().x(), row_rect.y() + 24 + 12)
    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.lines.mouseDoubleClickEvent(event)
    assert panel.canvas.editor.parent() is panel.lines.viewport()
    assert panel.canvas.edit_target[-1] is True


def test_context_menu_moves_selected_content_and_sync_defaults_off(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)

    def choose_channel(menu, position):
        move = menu.actions()[0].menu()
        assert not move.findChild(QCheckBox).isChecked()
        move.actions()[-1].trigger()

    monkeypatch.setattr(QMenu, "exec", choose_channel)
    panel.context_menu(panel.mapToGlobal(panel.rect().center()))
    wait_for_lyric_mapping(own_window)
    assert natural_sounds(own_window.lyrics_text)[0].track == 2
    assert next(item for item in own_window.view.notes() if item.note.id == 10).channel == 0


def test_multiline_move_preserves_color_override(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "@c2{あ}\nい", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    panel.lines.item(1).setSelected(True)
    panel.apply("channel", 1)
    wait_for_lyric_mapping(own_window)
    lines = natural_sounds(own_window.lyrics_text)
    assert [line.track for line in lines] == [1, 1]
    assert lines[0].sounds[0].color_channel == 2
    assert lines[1].sounds[0].color_channel is None


def test_split_merge_preserves_mapping_and_undo_selection(own_window, tmp_path, monkeypatch):
    def fail_warning(*args):
        raise AssertionError(args[-1])

    monkeypatch.setattr(QMessageBox, "warning", fail_warning)
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "@c2{(今日)[きょう].2}あ.1\nい", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    before = own_window.lyrics_text
    mapped = [ids for line in own_window.view.lyric_mapped for ids in line]
    boundary = unit_rects(panel.canvas.units(), panel.canvas.font())[1].left()
    position = QPointF(boundary, 30)
    panel.canvas.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert panel.canvas.cut == 1
    panel.canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    wait_for_lyric_mapping(own_window)
    assert len(panel.rows) == 3
    assert reading(panel.rows[0].units[0]) == "きょう"
    assert panel.rows[0].units[0].base.words[0].color_channel == 2
    assert [ids for line in own_window.view.lyric_mapped for ids in line] == mapped
    assert panel.single_row() == 1
    own_window.view.undo_stack.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == before
    assert panel.single_row() == 0
    own_window.view.undo_stack.redo()
    wait_for_lyric_mapping(own_window)
    assert panel.single_row() == 1
    panel.merge_line()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == before


def test_chapter_boundaries_preserve_rows_and_mapping(own_window, tmp_path):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(1)
    before = own_window.lyrics_text
    mapped = own_window.view.lyric_mapped
    panel.start_chapter()
    wait_for_lyric_mapping(own_window)
    assert list(panel.chapter_starts) == [0, 1]
    assert own_window.view.lyric_mapped == mapped
    assert len(panel.rows) == 3
    panel.merge_line()
    assert len(panel.rows) == 3
    panel.merge_chapter()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == before
    panel.start_chapter()
    wait_for_lyric_mapping(own_window)
    panel.lines.item(2).setSelected(True)
    panel.delete_lines()
    wait_for_lyric_mapping(own_window)
    assert len(panel.model.chapters) == 1
    assert len(panel.rows) == 1


def test_insert_delete_and_empty_document(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    panel = own_window.lyric_editor
    panel.open_at(0)
    notes = [(item.note.id, item.note.channel) for item in own_window.view.notes()]
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("う", True))
    panel.insert_line()
    wait_for_lyric_mapping(own_window)
    assert [line.units[0].text for line in panel.rows] == ["あ", "う", "い", "う"]
    assert panel.single_row() == 1
    panel.lines.selectAll()
    panel.delete_lines()
    wait_for_lyric_mapping(own_window)
    assert not panel.rows
    assert not panel.model.chapters
    assert [(item.note.id, item.note.channel) for item in own_window.view.notes()] == notes
    panel.insert_line()
    wait_for_lyric_mapping(own_window)
    assert len(panel.rows) == 1
    assert panel.rows[0].units[0].text == "う"
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("今日[きょう]", True))
    panel.insert_line(above=True)
    wait_for_lyric_mapping(own_window)
    assert panel.single_row() == 0
    assert reading(panel.rows[0].units[0]) == "きょう"
    before = own_window.lyrics_text
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("あ", False))
    panel.insert_line()
    assert own_window.lyrics_text == before


def test_split_shared_note_is_rejected(own_window, tmp_path, monkeypatch):
    _mixed_lyrics(own_window, tmp_path)
    assert own_window._apply_lyric_editor(own_window.lyrics_text, "あ.1い.+", set(), None)
    wait_for_lyric_mapping(own_window)
    panel = own_window.lyric_editor
    panel.open_at(0)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    before = own_window.lyrics_text
    panel.canvas.cut = 1
    panel.split_line()
    assert own_window.lyrics_text == before
    assert len(warnings) == 1
