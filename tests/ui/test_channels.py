# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QContextMenuEvent, QKeyEvent, QPalette
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu

from namioto import settings as store
from namioto.channels import Channel
from namioto.ui import theme
from namioto.ui.roll import PITCH_MAX

from .helpers import draw_note, reset_channels


def test_notes_land_on_the_active_channel_and_wear_its_colour(window) -> None:
    window.edit.pen.click()
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.set_active_channel(1)
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    drawn = window.view.notes()[0]
    assert drawn.channel == 1
    on_first = window.view.add_note(60, 0.0, 1.0, 0)
    assert on_first.fill != drawn.fill  # each channel paints its own colour
    reset_channels(window)


def test_new_channels_use_piano_instead_of_the_active_instrument(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0, program=53),))
    added = own_window.view.add_channel()
    assert added.program == 0
    assert own_window.view.channels[0].program == 53


def test_a_locked_channel_cannot_be_edited(window) -> None:
    window.edit.pen.click()
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    scene_pos = QPointF(3.0, PITCH_MAX - 60 + 0.5)
    window.view.set_channel_field(0, lock=True)

    assert window.view.channel_menu(scene_pos) is None  # the lock leaves nothing to move
    assert window.view.notes() == [note]

    draw_note(window, QPointF(6.0, 30.0))  # and the pen stays silent on it too
    assert window.view.notes() == [note]
    window.view.set_channel_field(0, lock=False)
    window.view.clear_notes()


def test_a_locked_channel_survives_select_all_and_delete(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0, lock=True), Channel(channel=1)))
    window.view.clear_notes()
    locked = window.view.add_note(60, 2.0, 1.0, 0)
    window.view.add_note(64, 4.0, 1.0, 1)
    window.view.undo_stack.clear()

    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier))
    assert not locked.isSelected()  # select-all skips a locked channel
    window.view.delete_selection()
    assert [note.pitch for note in window.view.notes()] == [60]
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 64]
    reset_channels(window)


def test_pasting_leaves_a_locked_channel_alone(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 1.0, 0).setSelected(True)
    window.view.set_playhead(8.0)
    assert window.view.copy_selection()
    window.view.set_channel_field(0, lock=True)

    assert window.view.paste_notes() is False  # a locked channel takes no new notes
    assert len(window.view.notes()) == 1
    window.view.set_channel_field(0, lock=False)
    window.view.clear_notes()


def test_an_invisible_channel_hides_its_notes(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(60, 2.0, 2.0)
    assert note.isVisible()
    window.view.set_channel_field(0, visible=False, lock=True)  # hiding also locks, noteDigger style
    assert not note.isVisible()
    window.view.set_channel_field(0, visible=True, lock=False)
    assert note.isVisible()
    window.view.clear_notes()


def test_removing_a_channel_takes_its_notes_and_leaves_the_numbers_alone(window) -> None:
    window.view.clear_notes()
    window.view.add_channel()
    window.view.add_note(60, 0.0, 1.0, 0)
    survivor = window.view.add_note(64, 1.0, 1.0, 1)
    assert window.view.remove_channel(0)
    assert [note.pitch for note in window.view.notes()] == [64]
    assert survivor.channel == 1
    reset_channels(window)


def test_a_muted_channel_is_left_out_of_the_program(window) -> None:
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.add_note(60, 0.0, 1.0, 0)
    window.view.add_note(64, 0.0, 1.0, 1)
    notes, channels = window._program()
    assert len(notes) == 2 and len(channels) == 2

    window.view.set_channel_field(1, mute=True)
    notes, channels = window._program()
    assert [note[0] for note in notes] == [60]
    assert [channel[0] for channel in channels] == [0]
    reset_channels(window)


def test_the_channels_button_toggles_the_sidebar(window) -> None:
    window.channel_panel.setVisible(False)
    window.transport.channels.click()
    assert window.channel_panel.isVisible()
    window.transport.channels.click()
    assert not window.channel_panel.isVisible()


def test_a_channel_switch_marks_the_exceptional_state(own_window) -> None:
    own_window.view.set_channels(
        (
            Channel(channel=0, visible=True),
            Channel(channel=1, lock=True, mute=True),
            Channel(channel=2, visible=False, lock=True),
        )
    )
    cards = own_window.channel_panel._cards
    assert cards[0].lock_button.isChecked() is False
    assert cards[0].eye_button.isChecked() is False  # visible is the normal state
    assert cards[1].lock_button.isChecked() is True
    assert cards[1].mute_button.isChecked() is True
    assert cards[1].eye_button.isChecked() is False
    assert cards[2].lock_button.isChecked() is True
    assert cards[2].eye_button.isChecked() is True  # hidden


def test_a_channel_change_refreshes_its_card_in_place(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    panel = own_window.channel_panel
    card = panel._cards[0]

    view.set_channel_field(0, lock=True)
    assert panel._cards[0] is card  # a field edit refreshes the card instead of rebuilding it
    assert card.lock_button.isChecked() is True

    view.set_active_channel(1)
    assert panel._cards[0] is card  # and so does making another channel active
    assert card.property("active") is False
    assert panel._cards[1].property("active") is True

    view.add_channel()
    assert panel._cards[0] is not card  # the set of channels changed, so the cards are rebuilt


def test_a_channel_id_shows_before_its_name(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0, name="Lead"), Channel(channel=2)))
    cards = own_window.channel_panel._cards
    assert cards[0].id.text() == "(0)"
    assert cards[0].name.text() == "Lead"
    assert cards[2].id.text() == "(2)"
    assert cards[2].name.text() != ""  # an unnamed channel keeps a word to read


def test_the_channel_id_wears_the_note_colour(own_window) -> None:
    own_window.view.set_channels((Channel(channel=0, color="#ff8800"),))
    card = own_window.channel_panel._cards[0]
    assert "#ff8800" in card.id.styleSheet()


def test_a_channel_moves_to_another_id_with_its_notes(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=3)))
    view.set_notes([(60, 0.0, 1.0, 3)])
    assert view.set_channel_number(3, 5) is True
    assert [channel.channel for channel in view.channels] == [0, 5]
    assert [note.channel for note in view.notes()] == [5]
    assert view.set_channel_number(5, 0) is False  # 1 is already taken
    assert view.set_channel_number(5, 5) is True  # the same number is a quiet no-op


def test_moving_a_channel_moves_its_colour_too(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    assert view.channel_color(1).name() == QColor(theme.NOTE_PALETTE[1]).name()
    assert view.set_channel_number(1, 4) is True
    assert view.channel_color(4).name() == QColor(theme.NOTE_PALETTE[4]).name()
    assert own_window.channel_panel._cards[4].id.styleSheet() == f"color: {QColor(theme.NOTE_PALETTE[4]).name()};"


def test_the_card_menu_can_change_the_channel_id(own_window, monkeypatch) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    card = own_window.channel_panel._cards[0]
    monkeypatch.setattr(
        QMenu, "exec", lambda self, *args, **kwargs: next(a for a in self.actions() if a.text() == "Channel ID…")
    )
    monkeypatch.setattr(QInputDialog, "getInt", lambda *args, **kwargs: (5, True))
    card.contextMenuEvent(
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(0, 0), card.mapToGlobal(QPoint(0, 0)))
    )
    assert [channel.channel for channel in view.channels] == [1, 5]


def test_the_active_channel_wears_the_selection_colour(own_window) -> None:
    view = own_window.view
    view.set_channels((Channel(channel=0), Channel(channel=1)))
    panel = own_window.channel_panel
    card = panel._cards[0]
    card.resize(288, 62)  # an unshown window never lays the cards out on its own
    highlight = panel.palette().color(QPalette.ColorRole.Highlight)

    def edge_distance() -> int:
        """How far the card's top edge is from the palette's selection colour, at its middle."""
        colour = card.grab().toImage().pixelColor(card.width() // 2, 1)
        return sum((getattr(colour, part)() - getattr(highlight, part)()) ** 2 for part in ("red", "green", "blue"))

    view.set_active_channel(1)
    plain = edge_distance()
    view.set_active_channel(0)
    assert edge_distance() < plain  # the outline is drawn only while the channel is active


def test_the_auto_page_and_overtone_toggles_start_off_and_reach_the_settings(own_window) -> None:
    assert own_window.transport.auto_page.isChecked() is False
    assert own_window.transport.overtone.isChecked() is False
    assert own_window.view.overtone_highlight is False

    own_window.transport.overtone.click()
    assert own_window.view.overtone_highlight is True
    own_window.transport.auto_page.click()
    assert store.get_value(own_window.settings, "editor", "overtone_highlight") is True
    assert store.get_value(own_window.settings, "editor", "auto_page") is True


def test_the_auto_page_turn_follows_the_playhead(own_window, monkeypatch) -> None:
    own_window.show()
    QApplication.processEvents()
    view = own_window.view
    view.set_zoom(48.0, 16.0)
    view.clear_notes()
    view.add_note(60, 0.0, 900.0)  # the page can only turn as far as the sound goes
    monkeypatch.setattr(own_window, "_is_playing", lambda: True)
    own_window.transport.auto_page.setChecked(True)

    def page() -> tuple[float, float]:
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        return rect.left(), rect.right()

    monkeypatch.setattr(own_window, "_position", lambda: 1.0)
    own_window._show_position()
    left, right = page()
    assert left <= 1.0 * view.bpm / 60.0 <= right

    monkeypatch.setattr(own_window, "_position", lambda: 300.0)
    own_window._show_position()
    left, right = page()
    playhead = 300.0 * view.bpm / 60.0
    assert left <= playhead <= right
    assert playhead - left < (right - left) * 0.5  # the playhead lands near the left of the fresh page

    own_window.transport.auto_page.setChecked(False)
    monkeypatch.setattr(own_window, "_position", lambda: 500.0)
    own_window._show_position()
    assert page() == pytest.approx((left, right))  # with the toggle off nobody turns the page
