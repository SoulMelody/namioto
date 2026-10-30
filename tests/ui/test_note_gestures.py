# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from namioto.channels import Channel
from namioto.interaction import Tool
from namioto.ui.roll import PITCH_MAX

from .helpers import FakeOutput, draw_note, focus, reset_channels, roll_mouse, roll_mouse_here, wheel


def roll_context_menu(window, scene_pos: QPointF) -> None:
    """Open the roll's context menu at a scene position, as a real right click would."""
    position = window.view.mapFromScene(scene_pos)
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        position,
        window.view.viewport().mapToGlobal(position),
    )
    QApplication.sendEvent(window.view.viewport(), event)


def pick_menu(monkeypatch, data) -> None:
    """Make the next context menu answer with the entry carrying `data`."""
    monkeypatch.setattr(
        QMenu, "exec", lambda self, *args, **kwargs: next((a for a in self.actions() if a.data() == data), None)
    )


def test_clicking_the_roll_moves_the_playhead(window) -> None:
    if window.view.edit_mode:
        window.edit.mode.click()  # outside edit mode the roll is a seek bar
    assert not window.view.edit_mode
    scene_pos = QPointF(2.0, 40.0)  # scene units: two beats across, forty rows down

    roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos)
    assert window.view.playhead == pytest.approx(1.0)  # two beats at 120 BPM
    assert window.player.position == pytest.approx(1.0)

    window.edit.pen.click()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_the_pen_draws_where_the_playhead_lands(window) -> None:
    window.edit.pen.click()  # picking a tool enters edit mode
    assert window.view.edit_mode and window.view.tool is Tool.PEN
    window.view.set_playhead(None)
    window.view.clear_notes()
    left = QPointF(2.0, 40.0)
    right = QPointF(4.0, 40.0)

    draw_note(window, left, right)
    assert len(window.view.notes()) == 1
    assert window.view.playhead == pytest.approx(2.0)  # the drag carried the cursor to 4 beats
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_clicking_a_key_previews_the_pitch_under_it(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    keyboard = window.keyboard
    for y in (keyboard.height() // 4, keyboard.height() // 2, keyboard.height() * 3 // 4):
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(20.0, float(y)),
            keyboard.mapToGlobal(QPointF(20.0, float(y))),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.keyboard.mousePressEvent(event)
    assert len(fake.previews) == 3
    assert fake.previews == sorted(fake.previews, reverse=True)  # lower on screen is a lower note


def test_clicking_a_note_previews_it(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 1.0)
    window.view.centerOn(QPointF(2.5, float(PITCH_MAX - 69) + 0.5))
    position = window.view.mapFromScene(QPointF(2.5, float(PITCH_MAX - 69) + 0.5))
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(position),
        window.view.viewport().mapToGlobal(QPointF(position)),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.view.mousePressEvent(event)
    assert fake.previews == [69]
    window.view.mouseReleaseEvent(event)
    window.view.clear_notes()


def test_the_second_of_two_quick_clicks_still_sounds_and_seeks(window, monkeypatch) -> None:
    """Qt hands the second quick click over as a double-click, and that press counts as well."""
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.clear_notes()
    row = float(PITCH_MAX - 60) + 0.5

    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
        window.view.set_playhead(None)
        roll_mouse(window, kind, QPointF(2.5, row))
        assert fake.previews[-1] == 60
        assert window.view.playhead == pytest.approx(1.25)  # 2.5 beats at 120 BPM
        roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))

    assert fake.previews == [60, 60]
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_drag_in_view_mode_scrubs_the_playhead(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()
    window.view.clear_notes()
    window.view.set_playhead(None)
    window.view.centerOn(QPointF(3.0, float(PITCH_MAX - 60) + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, float(PITCH_MAX - 60) + 0.5))
    for beats, pitch in ((3.0, 62), (4.0, 64)):
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(float(beats), float(PITCH_MAX - pitch) + 0.5))
        assert window.view.playhead == pytest.approx(beats / 2.0)  # two beats a second at 120 BPM
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, float(PITCH_MAX - 64) + 0.5))

    assert fake.previews == [60, 62, 64]  # the drag sounds the rows it crosses
    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_drag_glides_the_playhead_and_sounds_every_row_it_crosses(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.edit.pen.click()  # the pen draws and auditions while it slides
    window.view.clear_notes()
    window.view.set_playhead(None)
    window.view.centerOn(QPointF(2.5, float(PITCH_MAX - 60) + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, float(PITCH_MAX - 60) + 0.5))
    for pitch in (62, 64, 64, 65):  # two moves inside one row must not sound it twice
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.5 + (pitch - 60) * 0.5, float(PITCH_MAX - pitch) + 0.5))
    assert fake.previews == [60, 62, 64, 65]  # a glissando up the rows
    assert window.view.playhead == pytest.approx(2.5)  # the drag reached 5 beats at 120 BPM
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, float(PITCH_MAX - 65) + 0.5))

    assert fake.previews == [60, 62, 64, 65]  # no stray preview after the release
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(7.0, float(PITCH_MAX - 70) + 0.5))
    assert fake.previews == [60, 62, 64, 65]  # and none while merely hovering either
    assert window.view.playhead == pytest.approx(2.5)

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_click_sounds_the_row_it_lands_on_in_either_mode(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    if window.view.edit_mode:
        window.edit.mode.click()  # nobody has picked a tool: this is the listening mode
    window.view.clear_notes()
    window.view.add_note(64, 2.0, 1.0)
    empty_row = float(PITCH_MAX - 60) + 0.5
    note_row = float(PITCH_MAX - 64) + 0.5

    draw_note(window, QPointF(2.5, empty_row))
    assert fake.previews == [60]  # an empty row sounds the pitch it is on
    draw_note(window, QPointF(2.5, note_row))
    assert fake.previews == [60, 64]  # a note sounds its own

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_press_on_a_note_moves_the_playhead_as_well(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.edit.pen.click()
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))
    window.view.set_playhead(None)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    assert window.view.playhead == pytest.approx(1.25)  # 2.5 beats at 120 BPM, note or not
    assert fake.previews == [69]  # and the note is still the one being edited
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(6.0, row))
    assert (note.start, note.end) == (5.5, 6.5)
    assert window.view.playhead == pytest.approx(3.0)  # the playhead travels with the drag
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(6.0, row))

    window._stop()
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_drawing_a_note_covers_the_cells_it_passed_through(window, monkeypatch) -> None:
    fake = FakeOutput()
    monkeypatch.setattr(window, "player", fake)
    window.view.clear_notes()
    row = float(PITCH_MAX - 69) + 0.5
    press = QPointF(8.3, row)  # a third of the way into the cell that starts at 8.0
    draw_note(window, press, QPointF(9.6, row))
    notes = window.view.notes()
    assert len(notes) == 1
    note = notes[0]
    assert (note.pitch, note.start, note.end) == (69, 8.0, 10.0)
    assert note.start <= press.x() <= note.end  # the note is where the click was
    assert fake.previews == [69]
    window.view.clear_notes()


def test_dragging_left_draws_the_note_to_the_left(window) -> None:
    window.view.clear_notes()
    row = float(PITCH_MAX - 69) + 0.5
    draw_note(window, QPointF(9.4, row), QPointF(7.6, row))
    note = window.view.notes()[0]
    assert (note.start, note.end) == (7.5, 9.5)
    window.view.clear_notes()


def test_a_click_without_a_drag_draws_one_snap_cell(window) -> None:
    row = float(PITCH_MAX - 69) + 0.5
    for with_move in (False, True):  # Qt hands out move events even when the mouse barely moved
        window.view.clear_notes()
        press = QPointF(8.3, row)
        draw_note(window, press, press if with_move else None)
        note = window.view.notes()[0]
        assert (note.start, note.end) == (8.0, 8.5)
    window.view.clear_notes()


def test_drawing_slides_the_note_to_the_row_the_pointer_moves_to(window) -> None:
    window.view.clear_notes()
    draw_note(
        window,
        QPointF(8.3, float(PITCH_MAX - 69) + 0.5),
        QPointF(9.6, float(PITCH_MAX - 76) + 0.5),  # slid seven rows up while drawing
    )
    note = window.view.notes()[0]
    assert (note.pitch, note.start, note.end) == (76, 8.0, 10.0)
    window.view.clear_notes()


def test_dragging_either_edge_of_a_note_changes_its_duration(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.05, row))  # within the left grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.0, row))
    assert (note.start, note.end) == (1.0, 4.0)  # the start moved, the end stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # within the right grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    assert (note.start, note.end) == (1.0, 5.0)  # the end moved, the start stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    window.view.clear_notes()


def test_shift_dragging_a_note_moves_the_edge_that_was_grabbed(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0, so it is split at 3.0
    row = float(PITCH_MAX - 69) + 0.5
    start = Qt.KeyboardModifier.ShiftModifier

    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row), start)  # left half
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.6, row), start)
    assert (note.start, note.end) == (1.5, 4.0)  # the start moved, the end stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.6, row), start)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.5, row), start)  # right half
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.9, row), start)
    assert (note.start, note.end) == (1.5, 5.0)  # the end moved, the start stayed
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.9, row), start)

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.9, row), start)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(9.0, row), start)  # drag the start past the end
    assert note.duration == pytest.approx(window.view.snap)  # one snap cell, never nothing
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(9.0, row), start)
    window.view.clear_notes()


@pytest.mark.parametrize(
    ("press_x", "first_x", "next_x"),
    [(2.5, 3.5, 4.5), (2.05, 1.5, 1.0), (2.95, 3.5, 4.0), (5.0, 6.0, 7.0)],
)
def test_note_gestures_reuse_collisions_and_skip_repeated_targets(
    own_window, monkeypatch, press_x, first_x, next_x
) -> None:
    view = own_window.view
    own_window.edit.pen.click()
    view.snap = 0.5
    view.add_note(69, 2.0, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    view.centerOn(QPointF(4.0, row))
    checks = []
    indices = []
    original_check = view._would_collide
    original_index = view.document.index_by_voice
    monkeypatch.setattr(
        view, "_would_collide", lambda *args, **kwargs: (checks.append(args), original_check(*args, **kwargs))[1]
    )
    monkeypatch.setattr(view.document, "index_by_voice", lambda: (indices.append(True), original_index())[1])
    roll_mouse(own_window, QEvent.Type.MouseButtonPress, QPointF(press_x, row))
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(first_x, row))
    geometry = [(note.start, note.pitch, note.duration) for note in view.notes()]
    assert len(checks) == 1
    assert len(indices) == 1
    gesture = view._gesture
    assert gesture is not None
    assert all(note not in gesture.collision_ignored for notes in gesture.collision_index.values() for note in notes)
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(first_x, row))
    assert len(checks) == 1
    assert [(note.start, note.pitch, note.duration) for note in view.notes()] == geometry
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(next_x, row))
    assert len(checks) == 2
    assert len(indices) == 1
    roll_mouse(own_window, QEvent.Type.MouseButtonRelease, QPointF(next_x, row))
    assert view._gesture is None


def test_drag_index_checks_new_pitches_and_skips_repeated_collisions(own_window, monkeypatch) -> None:
    view = own_window.view
    own_window.edit.pen.click()
    view.snap = 0.5
    moving = view.add_note(69, 2.0, 1.0)
    view.add_note(70, 4.0, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    view.centerOn(QPointF(4.0, row))
    checks = []
    original = view._would_collide
    monkeypatch.setattr(
        view, "_would_collide", lambda *args, **kwargs: (checks.append(args), original(*args, **kwargs))[1]
    )
    roll_mouse(own_window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(3.5, row - 1))
    assert (moving.start, moving.pitch) == (3.0, 70)
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(4.5, row - 1))
    assert (moving.start, moving.pitch) == (3.0, 70)
    assert len(checks) == 2
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(4.5, row - 1))
    assert len(checks) == 2
    roll_mouse(own_window, QEvent.Type.MouseMove, QPointF(4.5, row))
    assert (moving.start, moving.pitch) == (4.0, 69)
    roll_mouse(own_window, QEvent.Type.MouseButtonRelease, QPointF(4.5, row))


def test_a_note_drag_snaps_the_movement_and_keeps_the_note_off_the_grid(window) -> None:
    window.edit.pen.click()  # picking a tool enters edit mode
    window.view.clear_notes()
    note = window.view.add_note(69, 2.2, 1.0)  # a fifth of a cell off the 1/2-beat grid
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    body = QPointF(2.7, row)

    roll_mouse(window, QEvent.Type.MouseButtonPress, body)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.9, row))  # under half a cell: nothing at all
    assert (note.start, note.end) == (2.2, 3.2)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.0, row))  # past it: one whole cell
    assert (note.start, note.end) == (2.7, 3.7)  # and the note keeps its own offset
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.1, row))  # still one cell further along
    assert (note.start, note.end) == (2.7, 3.7)
    roll_mouse(window, QEvent.Type.MouseMove, body)  # and back again
    assert (note.start, note.end) == (2.2, 3.2)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, body)
    window.view.clear_notes()


def test_a_note_trim_that_comes_back_leaves_the_note_as_it_was(window) -> None:
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    before = window.view.undo_stack.count()

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # the right edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(6.0, row))
    assert (note.start, note.end) == (2.0, 6.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.95, row))  # and the drag hands it back
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(3.95, row))
    assert (note.start, note.end) == (2.0, 4.0)
    assert window.view.undo_stack.count() == before
    window.view.clear_notes()


def test_a_drag_cannot_drop_a_note_onto_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 4.0, 1.0)  # spans 4.0 to 5.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.5, row))  # one cell left: 3.0 to 4.0, touching
    assert (right.start, right.end) == (3.0, 4.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.0, row))  # another: 2.5 to 3.5 would overlap
    assert (right.start, right.end) == (3.0, 4.0)  # the drag stays at the last free place
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(1.5, row))  # far enough left to clear it again
    assert (right.start, right.end) == (1.0, 2.0)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(1.5, row))
    window.view.clear_notes()


def test_a_note_may_be_dragged_onto_another_channel(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    first = window.view.add_note(69, 2.0, 1.0, 0)
    window.view.add_note(69, 4.0, 1.0, 1)  # the same pitch over the same time, another channel
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.5, row))
    assert (first.start, first.end) == (4.0, 5.0)  # another channel is no obstacle
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.5, row))
    window.view.clear_notes()


def test_a_trim_cannot_reach_over_a_note_of_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 4.0, 2.0)  # spans 4.0 to 6.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(4.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.05, row))  # the left edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(3.5, row))  # 3.5 to 6.0, still clear
    assert (right.start, right.end) == (3.5, 6.0)
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.0, row))  # 2.0 to 6.0 would cross the first
    assert (right.start, right.end) == (3.5, 6.0)  # the edge stops where it last fit
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.0, row))
    window.view.clear_notes()


def test_a_trim_reaches_a_grid_line_a_hair_past_its_neighbour(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    left = window.view.add_note(69, 2.0, 0.5)  # spans 2.0 to 2.5
    # the neighbour lands a hair past the 3.0 line, the way a note comes back from a project file
    # that rounded its seconds; the snap line the grid offers must still be reachable
    window.view.add_note(69, 2.9999, 1.0)
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.45, row))  # the right edge
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(2.9, row))  # 2.0 to 3.0, the next snap line
    assert (left.start, left.end) == (2.0, 3.0)  # the hair it overlaps is below the model's slack
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.9, row))
    window.view.clear_notes()


def test_the_pen_opens_a_cell_that_overlaps_its_neighbour_by_a_hair(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 0.5)  # spans 2.0 to 2.5
    window.view.add_note(69, 2.9999, 1.0)  # a hair off the 3.0 line, as a project file leaves it
    row = float(PITCH_MAX - 69) + 0.5

    draw_note(window, QPointF(2.7, row), QPointF(2.7, row))  # one cell: 2.5 to 3.0
    assert (window.view.notes()[-1].start, window.view.notes()[-1].end) == (2.5, 3.0)
    window.view.clear_notes()


def test_the_pen_cannot_draw_over_a_note_of_its_own_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    row = float(PITCH_MAX - 69) + 0.5

    draw_note(window, QPointF(5.0, row), QPointF(2.0, row))  # back over the first note
    drawn = window.view.notes()[-1]
    assert (drawn.start, drawn.end) == (5.0, 5.5)  # it never left its first cell
    window.view.clear_notes()


def test_the_pen_cannot_open_a_note_over_one_of_its_pitch(window) -> None:
    window.view.clear_notes()
    window.edit.pen.click()
    window.view.snap = 0.5
    window.view.set_channels((Channel(channel=0),))
    window.view.add_note(69, 2.1, 1.0)  # spans 2.1 to 3.1, off the grid
    row = float(PITCH_MAX - 69) + 0.5  # the note's row, so a click misses the note but reads its pitch
    window.view.centerOn(QPointF(2.1, row))

    draw_note(window, QPointF(2.0, row))  # off the note, yet the cell it would open reaches into it
    assert len(window.view.notes()) == 1  # the pen refuses instead of doubling the pitch
    draw_note(window, QPointF(1.5, row))  # a cell that stops at the note's edge
    assert len(window.view.notes()) == 2
    window.view.clear_notes()


def test_clicking_a_selected_note_leaves_only_it_selected(window) -> None:
    window.view.clear_notes()
    left = window.view.add_note(69, 2.0, 1.0)  # spans 2.0 to 3.0
    right = window.view.add_note(69, 5.0, 1.0)  # spans 5.0 to 6.0
    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(4.0, row))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    assert window.view.selected_notes() == [left, right]

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))  # a click, no drag
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    assert window.view.selected_notes() == [left]

    # and a press on one of them that does drag still carries the whole selection
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.5, row), Qt.KeyboardModifier.ControlModifier)
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(5.5, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    assert (left.start, left.end) == (1.5, 2.5)  # the whole selection moved by the one cell
    assert (right.start, right.end) == (4.5, 5.5)

    window.view.undo()
    window.view.clear_notes()


def test_ctrl_click_is_what_adds_to_the_selection(window) -> None:
    window.view.clear_notes()
    first = window.view.add_note(69, 2.0, 1.0)
    second = window.view.add_note(72, 4.0, 1.0)
    first_row = float(PITCH_MAX - 69) + 0.5
    second_row = float(PITCH_MAX - 72) + 0.5
    window.view.centerOn(QPointF(4.0, first_row))
    for scene_pos, modifiers in (
        (QPointF(2.5, first_row), Qt.KeyboardModifier.NoModifier),
        (QPointF(4.5, second_row), Qt.KeyboardModifier.ControlModifier),
    ):
        roll_mouse(window, QEvent.Type.MouseButtonPress, scene_pos, modifiers)
        roll_mouse(window, QEvent.Type.MouseButtonRelease, scene_pos, modifiers)
    assert first.isSelected() and second.isSelected()
    window.view.clear_notes()


def test_clicking_a_note_makes_its_channel_active_and_drops_the_others(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    window.view.clear_notes()
    on_zero = window.view.add_note(60, 2.0, 1.0, 0)
    on_one = window.view.add_note(64, 4.0, 1.0, 1)
    on_zero.setSelected(True)
    window.view.set_active_channel(0)
    window.view.centerOn(QPointF(4.5, PITCH_MAX - 64 + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(4.5, PITCH_MAX - 64 + 0.5))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.5, PITCH_MAX - 64 + 0.5))
    assert window.view.active_channel == 1
    assert on_one.isSelected() and not on_zero.isSelected()
    reset_channels(window)


def test_the_marquee_follows_the_pointer_across_a_scroll(window) -> None:
    window.edit.select.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    framed = [window.view.add_note(60 + row, 2.0, 1.0) for row in range(3)]
    swept = window.view.add_note(60, 6.0, 1.0)  # the scroll carries the box's far corner over it
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 61 + 0.5))
    start = window.view.mapFromScene(QPointF(1.5, PITCH_MAX - 64 + 0.5))
    end = window.view.mapFromScene(QPointF(4.0, PITCH_MAX - 58 + 0.5))

    roll_mouse_here(window, QEvent.Type.MouseButtonPress, start)
    roll_mouse_here(window, QEvent.Type.MouseMove, end)
    assert window.view.selected_notes() == framed
    seeks = []
    window.view.seek_requested.connect(seeks.append)

    wheel(window.view, -240)  # a wheel down scrolls the roll to the right
    scrolled = window.view.horizontalScrollBar().value()
    assert scrolled > 0
    assert window.view._rubber.rect().topLeft().x() == pytest.approx(1.5, abs=0.05), "the press point holds"
    assert window.view.selected_notes() == framed + [swept], "the far corner stays under the pointer"

    roll_mouse_here(window, QEvent.Type.MouseMove, end - QPoint(scrolled, 0))  # back over the same beats
    assert window.view.selected_notes() == framed, "a note the box leaves is let go"
    assert seeks == [], "a box drag frames notes; it does not scrub the song"
    roll_mouse_here(window, QEvent.Type.MouseButtonRelease, end - QPoint(scrolled, 0))
    assert window.view.selected_notes() == framed
    window.view.clear_notes()
    window.edit.mode.click()


def test_a_marquee_selects_only_the_active_channel(window) -> None:
    window.edit.select.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    window.view.clear_notes()
    on_zero = window.view.add_note(60, 2.0, 2.0, 0)
    on_one = window.view.add_note(62, 2.0, 2.0, 1)
    window.view.set_active_channel(0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 61 + 0.5))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(1.0, PITCH_MAX - 64 + 0.5))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, PITCH_MAX - 58 + 0.5))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, PITCH_MAX - 58 + 0.5))
    assert on_zero.isSelected() and not on_one.isSelected()
    reset_channels(window)
    window.edit.mode.click()


def test_a_drawn_note_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    assert len(window.view.notes()) == 1
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert window.view.notes() == []
    assert window.view.undo_stack.canRedo()
    window.view.redo()
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_moving_notes_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row - 2))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row - 2))
    assert window.view.undo_stack.count() == 1
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (4.0, 71)

    window.view.undo()
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (2.0, 69)
    window.view.redo()
    assert (window.view.notes()[0].start, window.view.notes()[0].pitch) == (4.0, 71)
    window.view.clear_notes()


def test_trimming_a_note_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)  # spans 2.0 to 4.0
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.95, row))  # within the right grab band
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(5.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(5.0, row))
    assert window.view.undo_stack.count() == 1
    assert window.view.notes()[0].end == 5.0

    window.view.undo()
    assert window.view.notes()[0].end == 4.0
    window.view.clear_notes()


def test_deleting_the_selection_is_one_undo_step(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 0.0, 1.0)
    window.view.add_note(62, 1.0, 1.0)
    window.view.undo_stack.clear()

    for item in window.view.notes():
        item.setSelected(True)
    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert window.view.notes() == []
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 62]
    window.view.clear_notes()


def test_escape_takes_back_a_note_that_was_just_drawn(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    window.view.centerOn(QPointF(3.0, 40.0))

    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, 40.0))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.0, 40.0))
    assert len(window.view.notes()) == 1  # the gesture opened a note
    window.view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, 40.0))

    assert window.view.notes() == []  # and Escape took it back again
    assert window.view.undo_stack.count() == 0  # with no undo step
    window.view.clear_notes()


def test_ctrl_d_deletes_the_selection(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    assert len(window.view.notes()) == 1

    focus(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []

    window.view.undo()
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_the_roll_shortcuts_need_the_roll(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 1.0).setSelected(True)
    focus(window)

    window.transport.bpm.setFocus()
    QTest.keyClick(window.transport.bpm, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert len(window.view.notes()) == 1  # a field keeps its own Ctrl+D

    window.view.setFocus()
    QTest.keyClick(window.view, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []  # in the roll, Ctrl+D is the notes'
    window.view.clear_notes()


def test_the_note_menu_lists_the_other_channels_and_a_new_one(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_channel()
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    seen: dict[str, list] = {}

    def fake_exec(self, *args, **kwargs):
        seen["entries"] = [(action.text(), action.data()) for action in self.actions() if not action.isSeparator()]
        return None

    monkeypatch.setattr(QMenu, "exec", fake_exec)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert seen["entries"] == [(Channel(channel=1).label, 1), ("New channel", -1)]
    assert [item.channel for item in window.view.notes()] == [0]  # and nothing moves until an entry is picked
    window.view.clear_notes()


def test_the_note_menu_leaves_a_locked_channel_out(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0), Channel(channel=1, lock=True)))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))

    menu = window.view.channel_menu(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [action.data() for action in menu.actions() if not action.isSeparator()] == [-1]
    window.view.clear_notes()


def test_a_cancelled_note_menu_leaves_the_selection(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_note(64, 8.0, 1.0, 0)
    window.view.centerOn(QPointF(8.5, PITCH_MAX - 64 + 0.5))
    keeper = window.view.notes()[0]
    keeper.setSelected(True)

    pick_menu(monkeypatch, None)  # the menu opens over the other note, but nothing is picked
    roll_context_menu(window, QPointF(8.5, PITCH_MAX - 64 + 0.5))

    assert [item.start for item in window.view.selected_notes()] == [2.0]
    window.view.clear_notes()


def test_a_note_moves_to_another_channel_from_the_menu(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.add_channel()
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    window.view.undo_stack.clear()

    pick_menu(monkeypatch, 1)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [item.channel for item in window.view.notes()] == [1]
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert [item.channel for item in window.view.notes()] == [0]
    window.view.clear_notes()


def test_a_note_moves_to_a_fresh_channel_in_one_step(window, monkeypatch) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 2.0, 2.0, 0)
    window.view.centerOn(QPointF(3.0, PITCH_MAX - 60 + 0.5))
    window.view.undo_stack.clear()

    pick_menu(monkeypatch, -1)
    roll_context_menu(window, QPointF(3.0, PITCH_MAX - 60 + 0.5))
    assert [channel.channel for channel in window.view.channels] == [0, 1]
    assert [item.channel for item in window.view.notes()] == [1]
    assert window.view.undo_stack.count() == 1  # creating the channel and the move are one step
    window.view.undo()
    assert [channel.channel for channel in window.view.channels] == [0]
    assert [item.channel for item in window.view.notes()] == [0]
    window.view.clear_notes()


def test_clearing_every_note_can_be_undone(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(60, 0.0, 1.0)
    window.view.add_note(62, 1.0, 1.0)
    window.view.undo_stack.clear()

    window.view.clear_notes()
    assert window.view.notes() == []
    window.view.undo()
    assert [note.pitch for note in window.view.notes()] == [60, 62]
    window.view.clear_notes()


def test_selecting_a_note_is_not_an_undo_step(window) -> None:
    window.edit.select.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    note = window.view.add_note(69, 2.0, 1.0)
    window.view.undo_stack.clear()

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.5, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(2.5, row))
    assert note.isSelected()
    assert window.view.undo_stack.count() == 0
    window.view.clear_notes()
